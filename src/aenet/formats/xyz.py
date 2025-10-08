#!/usr/bin/env python

"""
Read and write XYZ coordinates files.

"""

import sys

from ..geometry import AtomicStructure
from .. import util
from .parser_abc import ParserABC

__author__ = "Alexander Urban"
__date__ = "2013-07-12"


class XYZParser(ParserABC):
    def __init__(self):
        self.name = 'xyz'
        self.description = 'XYZ Cartesian coordinates'
        self.extensions = ['xyz']
        self.default_file_names = []

    def read(self, infile, **kwargs):
        """
        Parse atomic structure file in XYZ format.

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

        step = 0
        line = f.readline()
        while (line):
            step += 1
            natoms = int(line.strip())
            comment = f.readline()
            coords = []
            forces = []
            types = []
            for i in range(natoms):
                line = f.readline().split()
                types.append(line[0].strip())
                coords.append([float(el) for el in line[1:4]])
            if (len(line) >= 7):
                forces.append([float(el) for el in line[4:7]])
            if (step == 1):
                struc = AtomicStructure(coords[:], types[:], forces=forces[:])
            else:
                struc.add_frame(coords[:], forces=forces[:])
            if (len(comment.strip()) > 0):
                struc.add_comment(comment.strip())
            line = f.readline()

        if close_file:
            f.close()

        self._amend(struc, **kwargs)
        return struc

    def write(self, struc, outfile=None, frame=None, **kwargs):
        """
        Write atomic structure to file in XYZ format.

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
            for i in range(struc.nframes):
                self.write(struc, f, frame=i)
        else:
            f.write("{:d}\n".format(struc.natoms))
            if (struc.pbc):
                (avec, a, b, c, alpha, beta, gamma
                 ) = util.standard_cell(struc.avec[frame], angles=True)
                f.write("a = {}, b = {}, c = {}, ".format(a, b, c) +
                        "alpha = {}, beta = {}, gamma = {}\n".format(
                            alpha, beta, gamma))
            else:
                if (struc.ncomments > 0):
                    f.write(struc.comments[min(frame, struc.ncomments)] + "\n")
                else:
                    f.write("XYZ Cartesian atomic coordinates\n")
            for i in range(struc.natoms):
                f.write("{:2s}  ".format(struc.types[i]))
                f.write("{:15.8f}  {:15.8f}  {:15.8f}".format(
                    *struc.coords[frame][i]))
                if (struc.forces is not None and len(struc.forces) > 0):
                    if (struc.forces[frame] is not None
                            and len(struc.forces[frame]) > 0):
                        f.write("{:15.8f}  {:15.8f}  {:15.8f}\n".format(
                            *struc.forces[frame][i]))
                    else:
                        f.write("\n")
                else:
                    f.write("\n")

        if closefile:
            f.close()
