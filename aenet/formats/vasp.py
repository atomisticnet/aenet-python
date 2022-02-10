#!/usr/bin/env python

"""
Read and write files in VAPS's POSCAR format.

"""

import numpy as np
import sys

try:
    from lxml import etree
    has_lxml = True
except ImportError:
    import xml.etree.cElementTree as ET

from ..geometry import AtomicStructure
from ..exceptions import ArgumentError
from .parser_abc import ParserABC

__author__ = "Alexander Urban"
__date__ = "2013-03-29"


class VaspParser(ParserABC):
    def __init__(self):
        self.name = 'vasp'
        self.description = "VASP's POSCAR/CONTCAR format"
        self.extensions = ['vasp']
        self.default_file_names = ['POSCAR', 'CONTCAR']

    def read(self, infile, **kwargs):
        """
        Parse atomic structure file in the VASP POSCAR format.

        Arguments:
          infile   name of the input file

        Rerturns:
          instance of the AtomicStructure class
        """

        self._check_amend_args(**kwargs)

        if hasattr(infile, 'readline'):
            f = infile
            close_file = False
        else:
            f = open(infile, 'r')
            close_file = True

        comment = f.readline()
        a = float(f.readline().split()[0])
        avec = np.zeros((3, 3))
        for i in range(3):
            avec[i] = [float(el) for el in f.readline().split()]
        avec *= a
        line = f.readline()
        # try to read type names (modern VASP format)
        try:
            natoms = [int(el) for el in line.split()]
            ntypes = len(natoms)
            typenames = []
        except ValueError:
            typenames = line.split()
            line = f.readline()
            natoms = [int(el) for el in line.split()]
            ntypes = len(natoms)
        line = f.readline().strip()
        # selective dynamics (only first letter relevant)
        if (line[0] in "Ss"):
            seldyn = True
            line = f.readline().strip()
        else:
            seldyn = False
        # Cartesian or fractional coordinates:
        if (line[0] in "CcKk"):
            frac = False
        else:
            frac = True
        # read atomic coordinates
        types = []
        coords = []
        fixed = []
        def isfix(x): return x[0] in 'Ff'
        for t in range(ntypes):
            for i in range(natoms[t]):
                types.append(t)
                line = f.readline().split()
                coords.append([float(el) for el in line[0:3]])
                if seldyn:
                    fixed.append([isfix(el) for el in line[3:6]])
                if (len(typenames) == t):
                    if seldyn:
                        if (len(line) > 6) and (len(line[6]) <= 2):
                            typenames.append(line[6])
                    else:
                        if (len(line) > 3) and (len(line[3]) <= 2):
                            typenames.append(line[3])

        if close_file:
            f.close()

        struc = AtomicStructure(coords, types=types, typenames=typenames,
                                avec=avec, fractional=frac)
        struc.add_comment(comment.strip())
        if seldyn:
            struc.set_fixed_atoms(np.array(fixed, dtype=bool))

        self._amend(struc, **kwargs)
        return struc

    def write(self, struc, outfile=None, frame=-1, cartesian=False,
              wrap=True, **kwargs):
        """
        Write atomic structure to file in the VASP POSCAR format.

        Arguments:
          struc       instance of the AtomicStructure class
          outfile     name of the output file; if None, the contents
                      will be written to stdout
        """

        if not struc.pbc:
            raise ArgumentError(
                "Error: the VASP POSCAR format requires a periodic lattice")

        for kw in kwargs:
            sys.stderr.write("Warning: unsupported argument: {}\n".format(kw))

        if outfile is not None:
            f = open(outfile, 'w')
        else:
            f = sys.stdout

        if (struc.ncomments > 0):
            f.write(struc.comments[0] + "\n")
        else:
            f.write("Atomic structure with {} atom(s)\n".format(struc.natoms))

        # lattice vectors
        f.write("1.0\n")
        for i in range(3):
            f.write("{:20.13f} {:20.13f} {:20.13f}\n".format(
                *struc.avec[frame][i]))

        # atomic species
        if struc.names_known:
            f.write((struc.ntypes*"{:2s} " + "\n").format(*struc.typenames))

        # number of atoms per species
        for t in range(struc.ntypes):
            f.write("{} ".format(struc.natoms_of_typeID(t)))
        f.write("\n")

        # if some atomic coordinates are fixed
        if ((len(struc.fixed) > 0)
           and any([True in fix for fix in struc.fixed])):
            f.write("selective dynamics\n")
            seldyn = True
        else:
            seldyn = False

        # fractional (direct) coordinates
        if cartesian:
            f.write("cartesian\n")
            for t in range(struc.ntypes):
                for i in struc.atoms_of_typeID(t):
                    f.write((3*"{:20.13f} ").format(*struc.coords[frame][i]))
                    if seldyn:
                        for fix in struc.fixed[i]:
                            f.write("F " if fix else "T ")
                    if struc.names_known:
                        f.write("{}".format(struc.types[i]))
                    f.write("\n")
        else:
            f.write("direct\n")
            for t in range(struc.ntypes):
                for i in struc.atoms_of_typeID(t):
                    f.write((3*"{:20.13f} ").format(
                        *struc.fraccoo(i, frame=frame, wrap=wrap)))
                    if seldyn:
                        for fix in struc.fixed[i]:
                            f.write("F " if fix else "T ")
                    if struc.names_known:
                        f.write("{}".format(struc.types[i]))
                    f.write("\n")

        if outfile:
            f.close()


class VasprunParser(ParserABC):
    def __init__(self):
        self.name = 'vasprun'
        self.description = "VAPS's vasprun.xml archive"
        self.extensions = ['xml']
        self.default_file_names = ['vasprun.xml']

    def read(self, infile, align_frames=False, finalscf=None, **kwargs):
        """
        Parse files in VASP's `vasprun.xml' format.

        Special Arguments:
           finalscf   use energy from final SCF step instead of final
                      reported energy; this can be used to avoid readin
                      corrected energies, such as arising from van-der-Waals
                      corrections; no atomic forces will be read, as they
                      are not available for the uncorrected energies
        """

        self._check_amend_args(**kwargs)

        if has_lxml:
            parser = etree.XMLParser(recover=True, huge_tree=True)
            tree = etree.parse(infile, parser)
        else:
            tree = ET.parse(infile)
        root = tree.getroot()

        calc = root.findall('./calculation')
        structure = root.findall('./*/structure')
        energy = root.findall('./*/energy')
        forces = root.findall('./*/varray[@name="forces"]')

        types = []
        for rc in root.findall('./atominfo/array[@name="atoms"]/set/rc'):
            types.append(rc.find('c').text)
        types = np.array(types)

        for i in range(len(energy)):
            if finalscf is None:
                E = float(energy[i].findall('i[@name="e_wo_entrp"]')[0].text)
            else:
                # vasprun.xml uses quite inconsistent variable names.
                # At the very end, the energy without entropy is stored
                # in the variable 'e_wo_entrp', but this variable
                # contains the energy *with* entropy after each SCF
                # step.  The corrected energy is instead stored in
                # 'e_0_energy'.  Note that this variable (e_0_energy)
                # contains only the correction term at the very end.
                E = float(calc[i].findall(
                    './scstep/energy/i[@name="e_0_energy"]')[-1].text)
            avec = []
            for v in structure[i].findall('./crystal/varray[@name="basis"]/v'):
                avec.append([float(el) for el in v.text.split()])
            avec = np.array(avec)
            coo = []
            for v in structure[i].findall('./varray[@name="positions"]/v'):
                coo.append([float(el) for el in v.text.split()])
            coo = np.array(coo)
            if finalscf is None:
                frc = []
                for v in forces[i].findall('./v'):
                    frc.append([float(el) for el in v.text.split()])
                frc = np.array(frc)
            else:
                frc = None

            if i == 0:
                struc = AtomicStructure(coo, types, avec=avec,
                                        fractional=True, energy=E,
                                        forces=frc)
            else:
                struc.add_frame(coo, avec=avec, energy=E,
                                forces=frc, fractional=True)
                if align_frames:
                    struc.align_frames(i, 0)

        self._amend(struc, **kwargs)
        return struc


class XDatCarParser(ParserABC):
    def __init__(self):
        self.name = 'xdatcar'
        self.description = "VASP's XDATCAR format"
        self.extensions = []
        self.default_file_names = ['XDATCAR']

    def read(self, infile, **kwargs):
        """
        Parse atomic structure file in the VASP XDATCAR format.  The parser
        assumes that the atomic positions are in DIRECT coordinates.

        Arguments:
          infile   name of the input file

        Rerturns:
          instance of the AtomicStructure class

        """

        self._check_amend_args(**kwargs)

        if hasattr(infile, 'readline'):
            f = infile
            close_file = False
        else:
            f = open(infile, 'r')
            close_file = True

        comment = f.readline()
        a = float(f.readline().split()[0])
        avec = np.zeros((3, 3))
        for i in range(3):
            avec[i] = [float(el) for el in f.readline().split()]
        avec *= a
        line = f.readline()

        # try to read type names (modern VASP format)
        try:
            natoms = [int(el) for el in line.split()]
            ntypes = len(natoms)
            typenames = []
        except ValueError:
            typenames = line.split()
            line = f.readline()
            natoms = [int(el) for el in line.split()]
            ntypes = len(natoms)

        line = f.readline().strip()
        iframe = 0
        while line:
            types = []
            coo = []
            for t in range(ntypes):
                for i in range(natoms[t]):
                    types.append(t)
                    line = f.readline().split()
                    coo.append([float(el) for el in line[0:3]])
            if iframe == 0:
                struc = AtomicStructure(coo, types, typenames=typenames,
                                        avec=avec, fractional=True)
                struc.add_comment(comment.strip())
            else:
                struc.add_frame(coo, fractional=True)
            iframe += 1
            line = f.readline().strip()

        if close_file:
            f.close()

        self._amend(struc, **kwargs)
        return struc
