#!/usr/bin/env python

"""
Read and write atomic coordinates in the PDB format.

"""

import sys
import numpy as np

from .. import util
from .parser_abc import ParserABC

__author__ = "Alexander Urban"
__date__ = "2014-10-13"


class PDBParser(ParserABC):
    def __init__(self):
        self.name = 'pdb'
        self.description = 'Protein Data Base format'
        self.extensions = ['pdb']
        self.default_file_names = []

    def write(self, struc, outfile=None, frame=None, atom_attrib=None,
              **kwargs):
        """
        Write trajectory in the PDB format (appended PDB files)

        Arguments:
          outfile    name of the output file; None: write to stdout
          atom_attrib a list of attributes (floats) for each atom

        """

        for kw in kwargs:
            sys.stderr.write("Warning: unsupported argument: {}\n".format(kw))

        if not struc.pbc:
            a = b = c = 1.0
            alpha = beta = gamma = 90.0

        if atom_attrib is None:
            atom_attrib = [0.0 for i in range(struc.natoms)]

        header = ("CRYST1 {:8.3f} {:8.3f} {:8.3f} " +
                  "{:6.2f} {:6.2f} {:6.2f} P 1\nMODEL     1\n")

        atom = "ATOM {0:6d}   {1:2s} MOL     1     {2:7.3f} {3:7.3f} {4:7.3f} "
        atom += "{5:5.2f}  0.00          {1:2s}\n"

        if outfile is not None:
            f = open(outfile, 'w')
        else:
            f = sys.stdout

        for i, coords in enumerate(struc.coords):
            if struc.pbc:
                # reciprocal lattice vectors
                bvec = np.linalg.inv(struc.avec[i])
                (avec_rot, a, b, c, alpha, beta, gamma
                 ) = util.standard_cell(struc.avec[i], angles=True)
                coo = np.dot(coords, bvec)
                coo = np.dot(coo, avec_rot)
            else:
                coo = coords

            f.write(header.format(a, b, c, alpha, beta, gamma))
            for i in range(len(coo)):
                x = coo[i][0]
                y = coo[i][1]
                z = coo[i][2]
                f.write(
                    atom.format(i, struc.types[i], x, y, z, atom_attrib[i]))
            f.write("ENDMOL\n")

        if outfile:
            f.close()
