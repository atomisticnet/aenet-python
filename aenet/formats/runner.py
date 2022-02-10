#!/usr/bin/env python

"""
Read atomic structure information in the RuNNer format.

"""

import re
import sys
import numpy as np

from .. import units
from ..geometry import AtomicStructure
from .parser_abc import ParserABC

__author__ = "Nongnuch Artrith"
__date__ = "2013-08-25"


class RuNNerParser(ParserABC):
    def __init__(self):
        self.name = 'runner'
        self.description = 'RuNNer input.data format'
        self.extensions = []
        self.default_file_names = ['input.data']

    def read(self, infile, index=0, **kwargs):
        """
        Parse file with atomic coordinates in the RuNNer `input.data'
        format.

        Arguments:
          infile   name of the input file
          index    if not None, just return the selected structure

        Returns:
          an instance of the AtomicStructure class
        """

        self._check_amend_args(**kwargs)

        re_lattice = re.compile('^ *lattice')
        re_atom = re.compile('^ *atom')
        re_energy = re.compile('^ *energy')
        re_begin = re.compile('^ *begin')
        re_end = re.compile('^ *end')

        if hasattr(infile, 'readline'):
            f = infile
            close_file = False
        else:
            f = open(infile, 'r')
            close_file = True

        structures = []

        for line in f:
            if re_begin.match(line):
                avec = []
                coords = []
                forces = []
                types = []
                energy = None
            elif re_end.match(line):
                if (len(avec) > 0):
                    avec = np.array(avec)*units.Bohr2Ang
                else:
                    avec = None
                coords = np.array(coords)*units.Bohr2Ang
                forces = np.array(forces)*units.Ha2eV/units.Bohr2Ang
                types = np.array(types)
                struc = AtomicStructure(coords, types, avec=avec,
                                        forces=forces, energy=energy)
                structures.append(struc)
                if (index is not None) and (len(structures) > index):
                    break
            elif re_lattice.match(line):
                vec = [float(el) for el in line.split()[1:4]]
                avec.append(vec)
            elif re_atom.match(line):
                coo = [float(el) for el in line.split()[1:4]]
                frc = [float(el) for el in line.split()[7:10]]
                name = line.split()[4]
                coords.append(coo)
                forces.append(frc)
                types.append(name)
            elif re_energy.match(line):
                energy = float(line.split()[1])*units.Ha2eV

        if close_file:
            f.close()

        for struc in structures:
            self._amend(struc, **kwargs)
        if index is not None:
            return structures[index]
        else:
            return structures

    def write(self, struc, outfile=None, frame=-1, **kwargs):
        """
        Write structure information in the RuNNer input.data format.

        """

        for kw in kwargs:
            sys.stderr.write("Warning: unsupported argument: {}\n".format(kw))

        if outfile is not None:
            f = open(outfile, 'w')
        else:
            f = sys.stdout

        f.write("begin\n")
        f.write("comment\n")

        if struc.pbc:
            for v in struc.avec[frame]:
                f.write("lattice {0:14.6f} {1:14.6f} {2:14.6f}\n".format(
                    *(v/units.Bohr2Ang)))

        for i in range(struc.natoms):
            x = struc.coords[frame][i][0]/units.Bohr2Ang
            y = struc.coords[frame][i][1]/units.Bohr2Ang
            z = struc.coords[frame][i][2]/units.Bohr2Ang
            f.write("atom {0:17.9f} {1:17.9f} {2:17.9f} {3:3s}".format(
                    x, y, z, struc.types[i]))
            # set the atomic charge and energy to zero for every atom
            f.write("    0.000000000     0.000000000")
            if ((struc.forces[frame] is not None)
                    and (len(struc.forces[frame]) > 0)):
                f.write((3*" {:17.9f}").format(
                    struc.forces[frame][i, 0]/units.Ha2eV*units.Bohr2Ang,
                    struc.forces[frame][i, 1]/units.Ha2eV*units.Bohr2Ang,
                    struc.forces[frame][i, 2]/units.Ha2eV*units.Bohr2Ang))
            else:
                f.write((3*" {:17.9f}").format(0, 0, 0))
            f.write("\n")

        if struc.energy[frame]:
            f.write("energy {:17.8f}\n".format(
                struc.energy[frame]/units.Ha2eV))
        else:
            f.write("energy        0.00000000\n")

        f.write("charge        0.00000000\n")
        f.write("end\n")

        if outfile:
            f.close()
