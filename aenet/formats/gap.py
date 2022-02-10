#!/usr/bin/env python

"""
Read and write files in the structure format used by the Gaussian
Approximation Potential (GAP) code.

"""

import re
import numpy as np

from ..geometry import AtomicStructure
from .parser_abc import ParserABC

__author__ = "Nongnuch Artrith, Alexander Urban"
__date__ = "2015-06-21"


class GAPParser(ParserABC):
    def __init__(self):
        self.name = 'gap'
        self.description = 'GAP reference structure format'
        self.extensions = ['gap']
        self.default_file_names = []

    def read(self, infile, frame=None, **kwargs):
        """
        Parse atomic structure file in the GAP format.

        Arguments:
          infile       name of the input file
          frame (int)  read selected frame only

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

        def read_frame(f):
            line = f.readline()
            if len(line) == 0:
                return None
            else:
                natoms = int(line.split()[0])
                line = f.readline()
                match = re.search(r"Lattice *= *\"([^\"]*)\"", line)
                avec = np.reshape(
                    [float(el) for el in match.groups()[0].split()], (3, 3))
                match = re.search(r"energy *= *([0-9.-]*) ", line)
                energy = float(match.groups()[0])
                coords = []
                forces = []
                types = []
                for i in range(natoms):
                    line = f.readline().split()
                    types.append(line[0])
                    coords.append([float(el) for el in line[1:4]])
                    forces.append([float(el) for el in line[4:7]])
                coords = np.array(coords)
                forces = np.array(forces)
                return (natoms, avec, types, coords, forces, energy)

        istep = 0
        struc = None
        current_frame = read_frame(f)
        while current_frame is not None:
            if (frame is None) or (frame == istep):
                if struc:
                    if not struc.natoms == current_frame[0]:
                        raise NotImplementedError(
                            "Frames with varying numbers of atoms are "
                            "currently not supported.")
                    if not all(struc.types == current_frame[2]):
                        raise NotImplementedError(
                            "Frames with varying atomic species are "
                            "currently not supported.")
                    struc.add_frame(current_frame[3],
                                    avec=current_frame[1],
                                    forces=current_frame[4],
                                    energy=current_frame[5])
                else:
                    struc = AtomicStructure(current_frame[3],
                                            current_frame[2],
                                            avec=current_frame[1],
                                            forces=current_frame[4],
                                            energy=current_frame[5])
            current_frame = read_frame(f)

        if close_file:
            f.close()

        self._amend(struc, **kwargs)
        return struc
