#!/usr/bin/env python

"""
Read and write files in the XCrysDen structure format (XSF).

"""

import sys
import numpy as np

from ..geometry import AtomicStructure
from ..staticdata import atomic_number
from .parser_abc import ParserABC

__author__ = "Nongnuch Artrith, Alexander Urban"
__date__ = "2013-05-29"


class XSFParser(ParserABC):
    def __init__(self):
        self.name = 'xsf'
        self.description = 'XCrysDen Structure Format'
        self.extensions = ['xsf']
        self.default_file_names = []

    def write(self, struc, outfile=None, frame=-1,
              numeric_species=False, **kwargs):
        """
        Write structure information in an extended XSF format.
        """

        for kw in kwargs:
            sys.stderr.write("Warning: unsupported argument: {}\n".format(kw))

        frame = min(frame,  struc.nframes-1)
        frame = max(frame, -struc.nframes)

        if outfile:
            f = open(outfile, 'w')
        else:
            f = sys.stdout

        for c in struc.comments:
            f.write("# " + c + "\n")
        if (len(struc.comments) > 0):
            f.write("\n")

        if struc.energy[frame] is not None:
            f.write("# total energy = {:.8f} eV\n".format(struc.energy[frame]))
            f.write("\n")

        if struc.pbc:
            f.write("CRYSTAL\n")
            f.write("PRIMVEC\n")
            for v in struc.avec[frame]:
                f.write("    {:14.8f} {:14.8f} {:14.8f}\n".format(*v))
            f.write("PRIMCOORD\n")
            f.write("{} 1\n".format(struc.natoms))
        else:
            f.write("ATOMS\n")

        for i in range(struc.natoms):
            if numeric_species:
                try:
                    itype = atomic_number[struc.types[i].strip()]
                except KeyError:
                    sys.stderr.write("Warning: unknown species: {}".format(
                                     struc.types[i]))
                    itype = 1
                f.write("{:3d} ".format(itype))
            else:
                f.write("{:2s} ".format(struc.types[i]))
            f.write((3*" {:14.8f}").format(struc.coords[frame][i, 0],
                                           struc.coords[frame][i, 1],
                                           struc.coords[frame][i, 2]))
            if ((struc.forces[frame] is not None)
                    and (len(struc.forces[frame]) > 0)):
                f.write((3*" {:14.8f}").format(struc.forces[frame][i, 0],
                                               struc.forces[frame][i, 1],
                                               struc.forces[frame][i, 2]))
            f.write("\n")

        if outfile:
            f.close()

    def read(self, infile, **kwargs):
        """
        Parse atomic structure file in the XSF format.

        Arguments:
          infile   name of the input file

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

        avec = None
        energy = None

        istep = 0
        addframe = False

        line = f.readline()
        while line:
            if 'ANIMSTEPS' in line:
                # it's a trajectory
                pass
            elif 'PRIMVEC' in line:
                avec = []
                avec.append([float(el) for el in f.readline().split()])
                avec.append([float(el) for el in f.readline().split()])
                avec.append([float(el) for el in f.readline().split()])
                avec = np.array(avec)
            elif 'PRIMCOORD' in line:
                istep += 1
                f.readline()
                (coords, forces, types, line) = self._read_coords(f)
                addframe = True
            elif 'ATOMS' in line:
                istep += 1
                (coords, forces, types, line) = self._read_coords(f)
                addframe = True
            elif 'total energy' in line:
                energy = float(line.split('=')[1].split()[0])
            if addframe and (istep == 1):
                if avec is None:
                    struc = AtomicStructure(
                        np.array(coords), types, avec=None,
                        forces=forces, energy=energy)
                else:
                    struc = AtomicStructure(
                        np.array(coords), types, avec=avec.copy(),
                        forces=forces, energy=energy)
                addframe = False
            elif addframe:
                struc.add_frame(np.array(coords), avec=avec.copy(),
                                forces=forces)
                addframe = False
            else:
                line = f.readline()

        if close_file:
            f.close()

        self._amend(struc, **kwargs)
        return struc

    def _read_coords(self, f):
        """
        Read coordinates block from an XSF file.

        Arguments:
          f    file object

        Returns:
          tuple (coords, forces, types) with
          coords    atomic coordinates
          forces    atomic forces (if present in file)
          types     atomic species (symbol or number)
        """

        coords = []
        forces = []
        types = []

        line = f.readline()
        while line and (len(line.split()) >= 4):
            line = line.split()
            types.append(line[0])
            coords.append([float(el) for el in line[1:4]])
            if (len(line) > 4):
                forces.append([float(el) for el in line[4:7]])
            line = f.readline()

        if (len(forces) == 0):
            forces = None
        else:
            forces = np.array(forces)

        return (coords, forces, types, line)
