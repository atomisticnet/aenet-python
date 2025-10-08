#!/usr/bin/env python

"""
Read and write XYZ coordinates files compatible with the Prismatic
STEM simulation software.

See also: http://prism-em.com

"""

import sys
import numpy as np

from ..geometry import AtomicStructure
from .parser_abc import ParserABC
from ..staticdata import atomic_species, atomic_number
from .. import util

__author__ = "Alexander Urban"
__date__ = "2018-04-29"


class PrismaticParser(ParserABC):
    def __init__(self):
        self.name = 'prism'
        self.description = 'Prismatic XYZ format'
        self.extensions = ['prism']
        self.default_file_names = []

    def read(self, infile, **kwargs):
        """
        Parse atomic structure file in Prismatic's XYZ format.

        Arguments:
          infile   name of the input file

        Rerturns:
          instance of the AtomicStructure class
        """

        self._check_amend_args(**kwargs)

        if hasattr(infile, "readline"):
            f = infile
            close_file = False
        else:
            f = open(infile, 'r')
            close_file = True

        comment = f.readline()
        # prismatic only handles orthorhombic cells
        a, b, c = [float(dim) for dim in f.readline().split()[:3]]
        alpha = beta = gamma = 90.0
        avec = util.cellmatrix_from_params(a, b, c, alpha, beta, gamma)

        coords = []  # coordinates
        types = []  # atom types
        occups = []  # occupancies
        rms_vib = []  # RMS thermal vibration (usually 0.05-0.1 Ang)
        for line in f:
            fields = line.split()
            if len(fields) < 6:
                break
            types.append(atomic_species[int(fields[0])-1]['symbol'])
            coords.append([float(coo) for coo in fields[1:4]])
            occups.append(float(fields[4]))
            rms_vib.append(float(fields[5]))
        struc = AtomicStructure(coords[:], types[:], avec=avec)
        if (len(comment.strip()) > 0):
            struc.add_comment(comment.strip())

        if close_file:
            f.close()

        self._amend(struc, **kwargs)
        return struc

    def write(self, struc, outfile=None, frame=None, **kwargs):
        """
        Write atomic structure to file in Prismatic's XYZ format.

        Arguments:
          struc       instance of the AtomicStructure class
          outfile     name of the output file; if None, the contents
                      will be written to stdout
          frame       number of frame to write out; if None, all frames
                      will be written to a trajectory file
        """

        for kw in kwargs:
            sys.stderr.write("Warning: unsupported argument: {}\n".format(kw))

        if hasattr(outfile, 'write'):
            f = outfile
            closefile = False
        elif (outfile):
            f = open(outfile, 'w')
            closefile = True
        else:
            f = sys.stdout
            closefile = False

        if frame is None:
            frame = -1

        if (struc.ncomments > 0):
            f.write(struc.comments[min(frame, struc.ncomments)] + "\n")
        else:
            f.write("Prismatic XYZ format for STEM simulations\n")

        if (struc.pbc):
            (avec, a, b, c, alpha, beta, gamma
             ) = util.standard_cell(struc.avec[frame], angles=True)
            # hardcoded threshold for angles:
            anglediff = [abs(ang-90.0) > 0.001 for ang in (alpha, beta, gamma)]
            if any(anglediff):
                print("Error: Prismatic only accepts orthorhombic structures")
                print("       Angles: {} {} {}".format(alpha, beta, gamma))
                sys.exit()
        else:
            a = (np.max(struc.coords[frame][:, 0])
                 - np.min(struc.coords[frame][:, 0]))
            b = (np.max(struc.coords[frame][:, 1])
                 - np.min(struc.coords[frame][:, 1]))
            c = (np.max(struc.coords[frame][:, 2])
                 - np.min(struc.coords[frame][:, 2]))
        f.write("    {:15.8f}  {:15.8f}  {:15.8f}\n".format(a, b, c))

        for i in range(struc.natoms):
            f.write("{:3d} ".format(atomic_number[struc.types[i]]))
            f.write("{:15.8f}  {:15.8f}  {:15.8f}".format(
                *struc.coords[frame][i]))
            # set occupancy always to 1.0
            f.write("  1.0")
            # set RMS thermal vibrations always to 0.08
            f.write("  0.08")
            f.write("\n")

        # file end marker
        f.write("-1\n")

        if closefile:
            f.close()
