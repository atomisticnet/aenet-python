"""
Read and write the Tinker coordinates format.

"""

import sys
import numpy as np
import re

from ..geometry import AtomicStructure
from ..util import cellmatrix_from_params, standard_cell
from ..staticdata import atomic_species
from .. import units
from .parser_abc import ParserABC

__author__ = "Nongnuch Artrith, Alexander Urban"
__date__ = "2013-12-24"


class TinkerParser(ParserABC):
    def __init__(self):
        self.name = 'tinker'
        self.description = 'Tinker XYZ format'
        self.extensions = ['tinker', 'arc']
        self.default_file_names = []

    def read(self, infile, datafile=None, frcfile=None, keyfile=None,
             first=1, last=None, skip=1, **kwargs):
        """
        Parse file in the Tinker XYZ atomic coordinates format.
        It is assumed that the atom names correspond to the atomic symbols.

        Arguments:
          infile     name of the input file in Tinker XYZ format or instance
                     of file
          datafile   path to the Tinker output file (for energies)
                     note that the output file can only be used, if energies
                     were written out at each MD step
          frcfile    path to the Tinker forces trajectory
          keyfile    path to the Tinker input file
          first      number of the first frame to be read (1...N)
          last       number of the last frame to be read (1...N)
          skip       read every 'skip' frames

        Returns:
          an instance of the AtomicStructure class
        """

        self._check_amend_args(**kwargs)

        if hasattr(infile, 'readline'):
            f = infile
            close_file = False
        else:
            f = open(infile, 'r')
            close_file = True
            if keyfile is None:
                keyfile = ".".join(infile.split(".")[:-1]+["key"])
            if frcfile is None:
                frcfile = ".".join(infile.split(".")[:-1]+["frc"])

        if datafile is not None:
            frameinfo = self.read_Tinker_output(datafile)
            hasenergies = True
        else:
            hasenergies = False

        first = int(first) - 1
        last = None if last is None else int(last)
        skip = int(skip)

        energy = None
        forces = None
        avec = None

        # try to open forces file
        if frcfile is not None:
            try:
                frc = open(frcfile, 'r')
                hasforces = True
            except IOError:
                hasforces = False
        else:
            hasforces = False

        # try to parse key file (key file)
        if keyfile is not None:
            try:
                avec = self.read_Tinker_key_file(keyfile)
            except IOError:
                avec = None

        struc = None
        iframe = 0
        while True:
            frame = self.read_Tinker_XYZ_frame(f)
            if (frame is None) or (last is not None and iframe > last):
                break
            if hasforces:
                (forces, types2, avec2) = self.read_Tinker_XYZ_frame(frc)
                forces *= units.kcal_mol2eV
            if hasenergies:
                energy = (frameinfo[iframe]['potential_energy'] *
                          units.kcal_mol2eV)
            if (iframe >= first) and ((iframe - first) % skip == 0):
                (coords, types, avec2) = frame
                if avec2 is not None:
                    avec = avec2
                if struc is None:
                    struc = AtomicStructure(coords, types, energy=energy,
                                            forces=forces, avec=avec)
                else:
                    struc.add_frame(coords, energy=energy, forces=forces,
                                    avec=avec)
            iframe += 1

        if close_file:
            f.close()
        if hasforces:
            frc.close()

        self._amend(struc, **kwargs)
        return struc

    def write(self, struc, outfile=None, frame=None, **kwargs):
        """
        Write atomic coordinates of AtomicStructure instance to output in
        the Tinker XYZ format.  The atomic numbers will be used as atom
        types.

        Arguments:
          struc    instance of AtomicStructure
          outfile  name of the output file; if None, write to stdout

        """

        for kw in kwargs:
            sys.stderr.write("Warning: unsupported argument: {}\n".format(kw))

        if outfile is not None:
            f = open(outfile, 'w')
        else:
            f = sys.stdout

        if frame is not None:
            frame_range = [frame]
        else:
            frame_range = range(struc.nframes)

        for frame in frame_range:
            f.write("{:6d}  Tinker XYZ format\n".format(struc.natoms))

            # if periodic, rotate structure to standard orientation
            if struc.pbc:
                frac_coords = struc.cart2frac(struc.coords[frame], frame=frame)
                (struc.avec[frame], a, b, c, alpha, beta, gamma
                 ) = standard_cell(struc.avec[frame], angles=True)
                struc.bvec[frame] = np.linalg.inv(struc.avec[frame])
                struc.coords[frame] = struc.frac2cart(frac_coords, frame=frame)
                f.write((" " + 6*" {:11.6f}" + "\n").format(
                    a, b, c, alpha, beta, gamma))

            for i in range(struc.natoms):
                x = struc.coords[frame][i][0]
                y = struc.coords[frame][i][1]
                z = struc.coords[frame][i][2]
                t = struc.types[i]
                j = [atomic_species.index(el) for el in atomic_species
                     if el['symbol'] == t][0] + 1
                f.write(("{:6d}  {:4s} {:10.6f}  {:10.6f} "
                         " {:10.6f} {:5d}\n").format(i+1, t, x, y, z, j))

        if outfile:
            f.close()

    def read_Tinker_key_file(self, infile):
        """
        Search for simulation cell in Tinker key file.

        Arguments:
          infile    name of the Tinker key file

        Returns:
          matrix of lattice vectors, if cell was found, None else
        """

        re_a = re.compile("^A-AXIS *([0-9.-]*)", re.IGNORECASE)
        re_b = re.compile("^B-AXIS *([0-9.-]*)", re.IGNORECASE)
        re_c = re.compile("^C-AXIS *([0-9.-]*)", re.IGNORECASE)
        re_alpha = re.compile("^ALPHA *([0-9.-]*)", re.IGNORECASE)
        re_beta = re.compile("^BETA *([0-9.-]*)", re.IGNORECASE)
        re_gamma = re.compile("^GAMMA *([0-9.-]*)", re.IGNORECASE)

        pbc = False
        a = b = c = 1.0
        alpha = beta = gamma = 90.0
        with open(infile, 'r') as f:
            for line in f:
                m = re_a.match(line)
                if m:
                    a = float(m.group(1))
                    pbc = True
                    continue
                m = re_b.match(line)
                if m:
                    b = float(m.group(1))
                    pbc = True
                    continue
                m = re_c.match(line)
                if m:
                    c = float(m.group(1))
                    pbc = True
                    continue
                m = re_alpha.match(line)
                if m:
                    alpha = float(m.group(1))
                    pbc = True
                    continue
                m = re_beta.match(line)
                if m:
                    beta = float(m.group(1))
                    pbc = True
                    continue
                m = re_gamma.match(line)
                if m:
                    gamma = float(m.group(1))
                    pbc = True
                    continue

        if pbc:
            avec = cellmatrix_from_params(a, b, c, alpha, beta, gamma)
        else:
            avec = None

        return avec

    def read_Tinker_XYZ_frame(self, filep):
        """
        Read a single frame from a Tinker trajectory (positions or forces)
        in Tinker XYZ format.

        Arguments:
          filep (file)   opened trajectory file

        Returns:
          tuple (coords, types, avec) with
          coords     coordinates (positions or forces)
          types      list of atomic species
          avec       lattice vector matrix or None
        """

        line = filep.readline()
        if line:
            natoms = int(line.split()[0])
            coords = []
            types = []
            avec = None
            i = 0
            while i < natoms:
                line = filep.readline().split()
                if (i == 0) and (len(line) == 6):
                    try:
                        (a, b, c, alpha, beta, gamma
                         ) = [float(el) for el in line]
                        avec = cellmatrix_from_params(a, b, c, alpha,
                                                      beta, gamma)
                        line = filep.readline().split()
                    except:
                        pass
                coords.append([float(s.replace("D", "E")) for s in line[2:5]])
                types.append(line[1])
                i += 1
            coords = np.array(coords)
            types = np.array(types)
            return (coords, types, avec)
        else:
            return None

    def read_Tinker_output(self, filename_or_fp):
        """
        Parse Tinker output file.

        Arguments:
          filename_or_fp     file name (str) or file object

        Returns:
          List of dictionaries with information about each MD step.
          Note: all data is in the original Tinker units !

          Keys:  time (ps)
                 total_energy (kcal/mol)
                 potential_energy (kcal/mol)
                 kinetic_energy (kcal/mol)
                 temperature (K)
                 pressure (atm)
                 density (g/ml)

          Depending on the run type and ensemble, not all keys may be
          present.
        """

        re_time = re.compile("^ Simulation Time")
        re_total_energy = re.compile("^ Total Energy")
        re_potential_energy = re.compile("^ Potential Energy")
        re_kinetic_energy = re.compile("^ Kinetic Energy")
        re_temperature = re.compile("^ Temperature")
        re_pressure = re.compile("^ Pressure")
        re_density = re.compile("^ Density")

        re_newframe = re.compile("^ Frame Number")

        if hasattr(filename_or_fp, 'readline'):
            fp = filename_or_fp
            close_file = False
        else:
            fp = open(filename_or_fp, 'r')
            close_file = True

        trajec = []
        frame = {}
        for line in fp:
            if re_newframe.search(line):
                trajec.append(frame)
                frame = {}
            elif re_time.search(line):
                frame['time'] = float(line.split()[2])
            elif re_total_energy.search(line):
                frame['total_energy'] = float(line.split()[2])
            elif re_potential_energy.search(line):
                frame['potential_energy'] = float(line.split()[2])
            elif re_kinetic_energy.search(line):
                frame['kinetic_energy'] = float(line.split()[2])
            elif re_temperature.search(line):
                frame['temperature'] = float(line.split()[1])
            elif re_pressure.search(line):
                frame['pressure'] = float(line.split()[1])
            elif re_density.search(line):
                frame['density'] = float(line.split()[1])

        if close_file:
            fp.close

        return trajec
