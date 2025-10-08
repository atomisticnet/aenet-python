#!/usr/bin/env python

"""
Read cp.x output.

"""

import re
import numpy as np

from ..geometry import AtomicStructure
from ..exceptions import ArgumentError
from .. import units
from .parser_abc import ParserABC

__author__ = "Alexander Urban"
__date__ = "2013-10-17"


class CPParser(ParserABC):
    def __init__(self):
        self.name = "cp"
        self.description = "cp.x trajectory (QE)"
        self.extensions = ['pos']
        self.default_file_names = ['cp.pos']

    def read(self, infile='cp.pos', datafile=None, **kwargs):
        """
        Read trajectory in the `cp.pos' format of Quantum Espresso's `cp.x'.

        Arguments:
          infile     name of the input file (cp.pos format)
          datafile   name of the cp.x output file (for type names)

        Returns:
          an instance of the AtomicStructure class
        """

        self._check_amend_args(**kwargs)

        if datafile is None:
            raise ArgumentError(
                'cp.x output file required (specify with --datafile)')

        alat = re.compile('^ *alat *= ')
        positions = re.compile('^ *ATOMIC_POSITIONS')

        avec = []
        types = []

        # (1) read types and lattice vectors from output file

        if hasattr(infile, 'readline'):
            f_out = datafile
            close_file = False
        else:
            f_out = open(datafile, 'r')
            close_file = True

        line = f_out.readline()
        while line:
            if alat.match(line):
                avec.append([float(el) for el in f_out.readline().split()[2:]])
                avec.append([float(el) for el in f_out.readline().split()[2:]])
                avec.append([float(el) for el in f_out.readline().split()[2:]])
            elif positions.match(line):
                line = f_out.readline()
                while (len(line.strip()) > 0):
                    types.append(line.split()[0])
                    line = f_out.readline()
                break
            line = f_out.readline()

        avec = np.array(avec)*units.Bohr2Ang
        types = np.array(types)
        natoms = len(types)

        if close_file:
            f_out.close()

        # (2) read trajectory from positions file

        if hasattr(infile, 'readline'):
            f_pos = infile
            close_file = False
        else:
            f_pos = open(infile, 'r')
            close_file = True

        istep = 0
        iline = 0

        coords = []

        for line in f_pos:
            iline += 1
            coords.append([float(el) for el in line.split()])
            if ((iline % natoms) == 0):
                if (istep == 0):
                    struc = AtomicStructure(
                        np.array(coords)*units.Bohr2Ang, types, avec=avec)
                else:
                    struc.add_frame(np.array(coords)*units.Bohr2Ang)
                coords = []
                istep += 1

        if close_file:
            f_pos.close()

        self._amend(struc, **kwargs)
        return struc
