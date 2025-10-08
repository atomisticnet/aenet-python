"""
Read and write FHI-aims file formats.

"""

import sys
import re
import numpy as np

from ..geometry import AtomicStructure
from .parser_abc import ParserABC

__author__ = "Nongnuch Artrith"
__date__ = "2013-03-05"


class FHIaimsParser(ParserABC):
    def __init__(self):
        self.name = 'aims'
        self.description = 'FHI-aims geometry.in format'
        self.extensions = ['in', 'aims']
        self.default_file_names = ['geometry.in']

    def read(self, infile, **kwargs):
        """
        Parse file with atomic coordinates in the FHI-aims `geometry.in'
        format.

        Arguments:
          infile   name of the input file

        Returns:
          an instance of the AtomicStructure class
        """

        self._check_amend_args(**kwargs)

        lattice = re.compile('^ *lattice_vector ')
        atom = re.compile('^ *atom ')

        avec = []
        coords = []
        types = []

        if hasattr(infile, 'readline'):
            f = infile
            close_file = False
        else:
            f = open(infile, 'r')
            close_file = True

        for line in f:
            if lattice.match(line):
                vec = [float(el) for el in line.split()[1:4]]
                avec.append(vec)
            elif atom.match(line):
                coo = [float(el) for el in line.split()[1:4]]
                name = line.split()[4]
                coords.append(coo)
                types.append(name)

        if close_file:
            f.close()

        if len(avec) > 0:
            avec = np.array(avec)

        coords = np.array(coords)
        types = np.array(types)

        struc = AtomicStructure(coords, types, avec=avec)

        self._amend(struc, **kwargs)
        return struc

    def write(self, struc, outfile=None, frame=-1, **kwargs):
        """
        Write structure information in the FHI-aims 'geometry.in' format.
        """

        for kw in kwargs:
            sys.stderr.write("Warning: unsupported argument: {}\n".format(kw))

        if outfile is not None:
            f = open(outfile, 'w')
        else:
            f = sys.stdout

        if struc.pbc:
            for v in struc.avec[frame]:
                f.write("lattice_vector "
                        "{0:20.12f} {1:20.12f} {2:20.12f}\n".format(*v))

        for i in range(struc.natoms):
            x = struc.coords[frame][i][0]
            y = struc.coords[frame][i][1]
            z = struc.coords[frame][i][2]
            f.write("atom {0:20.12f} {1:20.12f} {2:20.12f} {3:2s}\n".format(
                    x, y, z, struc.types[i]))
            if all(struc.fixed[i]):
                f.write("constrain_relaxation .true.\n")

        if outfile:
            f.close()


class FHIaimsOutputParser(ParserABC):
    def __init__(self):
        self.name = 'aimsout'
        self.description = 'FHI-aims output file format'
        self.extensions = ['out']
        self.default_file_names = ['fhiaims.out']

    def read(self, infile, step=None, align_frames=False, **kwargs):
        """
        Read information from an FHI-aims output file.

        Arguments:
          infile   name of the input file
          step     optimization/MD step to be read;
          --> if step == None, the entire trajectory will be returned
          --> if step == -1, only the final step will be returned

        Returns:
          instance of the class AtomicStructure
        """

        self._check_amend_args(**kwargs)

        for kw in kwargs:
            sys.stderr.write("Warning: unsupported argument: {}\n".format(kw))

        types = []
        coords = []
        avec = []
        energy = 0.0
        MDrun = False

        if hasattr(infile, 'readline'):
            f = infile
            close_file = False
        else:
            f = open(infile, 'r')
            close_file = True

        # only search for initial geometry
        while True:
            line = f.readline()
            if not line:
                break
            if re.search("Molecular dynamics time step", line):
                MDrun = True
            if re.search("Input geometry:", line):
                line = f.readline()
                if not re.search("No unit cell", line):
                    avec.append([float(e) for e in f.readline().split()[-3:]])
                    avec.append([float(e) for e in f.readline().split()[-3:]])
                    avec.append([float(e) for e in f.readline().split()[-3:]])
                f.readline()
                f.readline()
                line = f.readline()
                while re.search("Species", line):
                    coords.append([float(el) for el in line.split()[-3:]])
                    types.append(line.split()[3])
                    line = f.readline()
                coords = np.array(coords)
                types = np.array(types)
                energy = 0.0
                forces = np.zeros(np.shape(coords))
                struc = AtomicStructure(coords.copy(), types, avec=avec[:],
                                        energy=energy, forces=[])
                break

        if MDrun:
            struc = self._parse_aims_MD(struc, energy, forces, coords,
                                        types, avec, f, step, align_frames)
        else:
            struc = self._parse_aims_optimize(struc, energy, forces, coords,
                                              types, avec, f, step,
                                              align_frames)

        if close_file:
            f.close()

        return struc

    def _parse_aims_optimize(self, struc, energy, forces, coords, types,
                             avec, f, step, align_frames):
        """
        Parse output from geometry optimization.
        """

        # Update geometry at every optimization/MD step.  The FHI-aims
        # output format for relaxations and single point calculations is
        # a little bit tricky.  The order in which quantities are
        # reported in the output file is:
        #
        #   (1) Begin self-consistency loop: Initialization.
        #   (1) Self-consistency cycle converged.
        #   (1) | Total energy corrected        :
        #   (1) Total atomic forces (unitary forces cleaned) [eV/Ang]:
        #   (1) Updated atomic structure:
        #   (2) Begin self-consistency loop: Re-initialization.
        #   (2) Self-consistency cycle converged.
        #   (2) | Total energy corrected        :
        #   (2) Total atomic forces (unitary forces cleaned) [eV/Ang]:
        #   (2) Updated atomic structure:
        #       ...
        #   (N) Have a nice day.
        #
        # in relaxations, but no updated atomic structure is reported in
        # single point runs.  We therefore store all information either
        # at the beginning of an SCF loop or at the very end of the
        # calculation.

        istep = 0
        converged = False
        re_scfbegin = re.compile("^ *Begin self-consistency loop *:")
        re_scfconv = re.compile("^ *Self-consistency cycle converged. *$")
        re_energy = re.compile(" Total energy corrected *: ")
        re_forces = re.compile("^ *Total atomic forces")
        re_update = re.compile("^ *Updated atomic structure:")
        re_nice = re.compile("^ *Have a nice day.")
        while True:
            line = f.readline()
            if not line:
                break
            if re_energy.search(line):
                # energy in eV
                energy = float(line.split()[5])  # T-->0 corrected
            elif re_update.search(line):
                re_lattice = re.compile("^ *lattice_vector ")
                re_atom = re.compile("^ *atom ")
                iatom = 0
                ivec = 0
                while (iatom < struc.natoms):
                    line = f.readline()
                    if re_lattice.search(line):
                        avec[ivec] = [float(el) for el in line.split()[1:4]]
                        ivec += 1
                    elif re_atom.search(line):
                        coords[iatom] = [float(el) for el in line.split()[1:4]]
                        iatom += 1
            elif re_forces.search(line):
                for iat in range(struc.natoms):
                    line = f.readline()
                    forces[iat] = [float(el) for el in line.split()[-3:]]
            elif re_scfconv.search(line):
                converged = True
            elif ((re_scfbegin.search(line) or (re_nice.search(line)))
                  and converged):
                istep += 1
                if ((step is None) and (istep == 1)):
                    struc = AtomicStructure(
                        coords.copy(), types, avec=avec[:],
                        energy=energy, forces=forces.copy())
                elif (step is None):
                    struc.add_frame(coords.copy(), avec=avec[:],
                                    energy=energy, forces=forces.copy())
                    if align_frames:
                        struc.align_frames(istep-1, 0)
                elif (step == istep):
                    struc = AtomicStructure(
                        coords.copy(), types, avec=avec[:],
                        energy=energy, forces=forces.copy())
                    break

        return struc

    def _parse_aims_MD(self, struc, energy, forces, coords, types, avec,
                       f, step, align_frames):
        """
        Parse output from geometry optimization.
        """
        # update geometry at every optimization/MD step
        istep = 0
        # re_energy = re.compile(" Total energy *: ")
        re_energy = re.compile(" Total energy, T -> 0 *: ")
        re_forces = re.compile("^ *Total atomic forces ")
        re_coords = re.compile("^ *Atomic structure .* time step:")
        re_atom = re.compile("^ *atom ")
        while True:
            line = f.readline()
            if not line:
                break
            if re_energy.search(line):
                # energy in eV
                energy = float(line.split()[9])  # T-->0 corrected
            elif re_coords.search(line):
                iatom = 0
                while (iatom < struc.natoms):
                    line = f.readline()
                    if re_atom.search(line):
                        coords[iatom] = [float(el) for el in line.split()[1:4]]
                        iatom += 1
                istep += 1
                if ((step is None) and (istep == 1)):
                    struc = AtomicStructure(
                        coords.copy(), types, avec=avec[:],
                        energy=energy, forces=forces.copy())
                elif (step is None):
                    struc.add_frame(coords.copy(), avec=avec[:],
                                    energy=energy, forces=forces.copy())
                    if align_frames:
                        struc.align_frames(istep-1, 0)
                elif (step == istep):
                    struc = AtomicStructure(
                        coords.copy(), types, avec=avec[:],
                        energy=energy, forces=forces.copy())
                    break
            elif re_forces.search(line):
                iat = 0
                while (iat < struc.natoms):
                    line = f.readline()
                    forces[iat] = [float(el) for el in line.split()[-3:]]
                    iat += 1

        return struc
