#!/usr/bin/env python

"""
External geometry input format (fort.34) used by the CRYSTAL code.

"""

import numpy as np
import sys

from ..geometry import AtomicStructure
from ..exceptions import ArgumentError
from .parser_abc import ParserABC
from ..staticdata import atomic_number, atomic_species
from ..util import symmetry_equivalent_atoms

__author__ = "Alexander Urban"
__date__ = "2015-11-16"


class CRYSTALParser(ParserABC):
    def __init__(self):
        self.name = 'crystal'
        self.description = "CRYSTAL's fort.34 format"
        self.extensions = ['crystal', 'gui']
        self.default_file_names = ['fort.34']

    def read(self, infile, **kwargs):
        """
        Parse atomic structure file in CRYSTAL's external (fort.34) format.

        Arguments:
          infile   name of the input file

        Returns:
          instance of the AtomicStructure class
        """

        self._check_amend_args(**kwargs)

        if hasattr(infile, 'readline'):
            f = infile
            close_file = False
        else:
            f = open(infile, 'r')
            close_file = True

        (dimensionality, centring, crystal_type
         ) = [int(el) for el in f.readline().split()[0:3]]

        # read lattice vectors
        avec = []
        for i in range(3):
            avec.append([float(el) for el in f.readline().split()[0:3]])
        avec = np.array(avec)
        bvec = np.linalg.inv(avec)

        # number of symmetry operations
        nsymop = int(f.readline().strip())

        # read symmetry operations and convert to reciprocal space
        symops = []
        for i in range(nsymop):
            S = []
            for i in range(3):
                S.append([float(el) for el in f.readline().split()[0:3]])
            # S = np.array(S)
            S = np.dot(np.dot(avec, S), bvec)
            T = np.array([float(el) for el in f.readline().split()[0:3]])
            T = np.dot(T, bvec)
            symops.append((S.copy(), T.copy()))

        # read asymmetric unit (irreducible atomic coordinates)
        # and convert to fractional coordinates:
        nirr = int(f.readline().split()[0])
        irrcoords = []
        irrtypes = []
        for i in range(nirr):
            line = f.readline().strip()
            Z = int(line.split()[0]) - 1
            irrtypes.append(str(atomic_species[Z]['symbol']))
            R = np.array([float(el) for el in line.split()[1:4]])
            r = np.dot(R, bvec)
            irrcoords.append(r.copy())

        if close_file:
            f.close()

        # r' = inv(A).S.A.r

        # generate symmetrically equivalent atomic coordinates
        (coords, types) = symmetry_equivalent_atoms(irrcoords, irrtypes,
                                                    symops)

        struc = AtomicStructure(coords, types=types, avec=avec,
                                fractional=True)
        struc.add_comment("Converted from CRYSTAL's fort.34 format")

        self._amend(struc, **kwargs)
        return struc

    def write(self, struc, outfile=None, frame=-1, **kwargs):
        """
        Write structure information CRYSTAL's fort.34 format.
        """

        for kw in kwargs:
            raise ArgumentError(
                "Warning: unsupported argument: {}\n".format(kw))

        if outfile is not None:
            f = open(outfile, 'w')
        else:
            f = sys.stdout

        if not struc.pbc:
            raise NotImplementedError(
                "Currently, only the CRYSTAL format for periodic "
                "structures is available.")

        # dimensionality, centring, and crystal type
        f.write((3*" {:3d}").format(3, 1, 1) + "\n")

        # lattice vectors
        for v in struc.avec[frame]:
            f.write((3*" {:19.12e}").format(*v) + "\n")

        # symmetry operations (only identity)
        f.write(" 1\n")
        f.write((3*" {:19.12e}").format(1, 0, 0) + "\n")
        f.write((3*" {:19.12e}").format(0, 1, 0) + "\n")
        f.write((3*" {:19.12e}").format(0, 0, 1) + "\n")
        f.write((3*" {:19.12e}").format(0, 0, 0) + "\n")

        # coordinates
        f.write(" {}\n".format(struc.natoms))
        for i in range(struc.natoms):
            x = struc.coords[frame][i][0]
            y = struc.coords[frame][i][1]
            z = struc.coords[frame][i][2]
            f.write((" {:4d}" + 3*"     {:15.12f}").format(
                    atomic_number[struc.types[i]], x, y, z) + "\n")

        if outfile:
            f.close()
