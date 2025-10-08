#!/usr/bin/env python

"""
Read and write LAMMPS data and dump files.

"""

import sys
import re
import numpy as np

from .. import util
from ..geometry import AtomicStructure
from ..staticdata import atomic_species, atomic_number
from .parser_abc import ParserABC

__author__ = "Nongnuch Artrith"
__date__ = "2013-03-05"


class LammpsDataParser(ParserABC):
    def __init__(self):
        self.name = 'lammpsdata'
        self.description = 'subset of LAMMPS data format'
        self.extensions = ['data']
        self.default_file_names = []

    def write(self, struc, outfile=None, frame=-1, **kwargs):

        for kw in kwargs:
            sys.stderr.write("Warning: unsupported argument: {}\n".format(kw))

        if struc.pbc:

            avec = util.standard_cell(struc.avec[frame])

            """
            At this point the rotated lattice vector matrix is:

                             [ a1   b1   c1 ]
              A = (a b c) =  [ 0    b2   c2 ]
                             [ 0    0    c3 ]
            """

            # LAMMPS cell parameters
            xhi = avec[0][0]
            xlo = 0.0
            yhi = avec[1][1]
            ylo = 0.0
            zhi = avec[2][2]
            zlo = 0.0
            xy = avec[1][0]
            xz = avec[2][0]
            yz = avec[2][1]

            # check, if the cell is triclinic
            if np.any(np.array([abs(xy), abs(xz), abs(yz)]) > 0.001):
                tilded = True
            else:
                tilded = False

        else:  # non-periodic --> get smallest confining cell

            xlo = np.min(struc.coords[frame][:, 0])
            xhi = np.max(struc.coords[frame][:, 0])
            ylo = np.min(struc.coords[frame][:, 1])
            yhi = np.max(struc.coords[frame][:, 1])
            zlo = np.min(struc.coords[frame][:, 2])
            zhi = np.max(struc.coords[frame][:, 2])
            tilded = False

        types_lmp = {}
        itype = 0
        for t in struc.types:
            if t not in types_lmp:
                itype += 1
                types_lmp[t] = itype

        if outfile:
            f = open(outfile, "w")
        else:
            f = sys.stdout

        f.write("# LAMMPS data format\n\n")
        f.write("{0} atoms\n".format(len(struc.coords[frame])))
        f.write("{0} atom types\n\n".format(struc.ntypes))
        f.write("{0:10.6f} {1:10.6f} xlo xhi\n".format(xlo, xhi))
        f.write("{0:10.6f} {1:10.6f} ylo yhi\n".format(ylo, yhi))
        f.write("{0:10.6f} {1:10.6f} zlo zhi\n".format(zlo, zhi))
        if tilded:
            f.write("{0:10.6f} {1:10.6f} {2:10.6f} xy xz yz\n".format(
                xy, xz, yz))
        f.write("\nMasses\n\n")
        for t in types_lmp:
            f.write("{0:5d} {1:14.8f}\n".format(
                types_lmp[t], atomic_species[atomic_number[t]-1]['mass']))
        f.write("\nAtoms\n\n")
        for i in range(struc.natoms):
            f.write("{0:5d} {1:3d} ".format(i+1, struc.typeID[i]+1))
            f.write("{0:14.8f} {1:14.8f} {2:14.8f}\n".format(
                *struc.coords[frame][i]))
        f.write("\n")

        if outfile:
            f.close()


class LammpsDumpParser(ParserABC):
    def __init__(self):
        self.name = 'lammps'
        self.description = 'LAMMPS atomic structure dump'
        self.extensions = ['dump', 'lammps', 'lmp']
        self.default_file_names = []

    def write(self, struc, outfile=None, frame=-1, **kwargs):

        for kw in kwargs:
            sys.stderr.write("Warning: unsupported argument: {}\n".format(kw))

        if struc.pbc:

            avec = util.standard_cell(struc.avec[frame])

            """
            At this point the rotated lattice vector matrix is:

                             [ a1   b1   c1 ]
              A = (a b c) =  [ 0    b2   c2 ]
                             [ 0    0    c3 ]
            """

            # LAMMPS cell parameters
            xhi = avec[0][0]
            xlo = 0.0
            yhi = avec[1][1]
            ylo = 0.0
            zhi = avec[2][2]
            zlo = 0.0
            xy = avec[1][0]
            xz = avec[2][0]
            yz = avec[2][1]

            # check, if the cell is triclinic
            if np.any(np.array([abs(xy), abs(xz), abs(yz)]) > 0.001):
                tilded = True
            else:
                tilded = False

            coords_lmp = np.zeros(np.shape(struc.coords[frame]))
            for i in range(struc.natoms):
                coords_lmp[i] = struc.fraccoo(i, frame=frame)

        else:

            xlo = np.min(struc.coords[frame][:, 0])
            xhi = np.max(struc.coords[frame][:, 0])
            ylo = np.min(struc.coords[frame][:, 1])
            yhi = np.max(struc.coords[frame][:, 1])
            zlo = np.min(struc.coords[frame][:, 2])
            zhi = np.max(struc.coords[frame][:, 2])
            Q = np.array([xlo, ylo, zlo])
            scale = 1.0/np.array([xhi-xlo, yhi-ylo, zhi-zlo])
            tilded = False
            # scaled coordinates
            coords_lmp = np.zeros(np.shape(struc.coords[frame]))
            for i in range(struc.natoms):
                coo = (struc.coords[frame][i] - Q) * scale
                coords_lmp[i] = coo

        if outfile:
            f = open(outfile, "w")
        else:
            f = sys.stdout

        f.write("ITEM: TIMESTEP\n0\n")
        f.write("ITEM: NUMBER OF ATOMS\n{0}\n".format(struc.natoms))
        if tilded:
            f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
            xlo_bound = xlo + min(0.0, xy, xz, xy+xz)
            xhi_bound = xhi + max(0.0, xy, xz, xy+xz)
            ylo_bound = ylo + min(0.0, yz)
            yhi_bound = yhi + max(0.0, yz)
            zlo_bound = zlo
            zhi_bound = zhi
            f.write("{0} {1} {2}\n".format(xlo_bound, xhi_bound, xy))
            f.write("{0} {1} {2}\n".format(ylo_bound, yhi_bound, xz))
            f.write("{0} {1} {2}\n".format(zlo_bound, zhi_bound, yz))
        else:
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write("{0} {1}\n".format(xlo, xhi))
            f.write("{0} {1}\n".format(ylo, yhi))
            f.write("{0} {1}\n".format(zlo, zhi))
        f.write("ITEM: ATOMS id type xs ys zs\n")
        for i in range(struc.natoms):
            f.write("{0} {1} {2} {3} {4}\n".format(
                i+1, struc.typeID[i]+1, coords_lmp[i][0],
                coords_lmp[i][1], coords_lmp[i][2]))

        if outfile:
            f.close()

    def read(self, infile, typenames=[], datafile=None, **kwargs):
        """
        Read structural information from a LAMMPS dump (trajectory) file.

        Arguments:
          infile      name of the input file in the LAMMPS dump format
          typenames   list of typenames corresponding to the type numbers

        Returns:
          instance of the AtomicStructure class
        """

        self._check_amend_args(**kwargs)

        re_timestep = re.compile("ITEM: *TIMESTEP")
        re_natoms = re.compile("ITEM: *NUMBER *OF *ATOMS")
        re_bounds = re.compile("ITEM: *BOX *BOUNDS")
        re_atoms = re.compile("ITEM: *ATOMS")

        if datafile:
            re_Step = re.compile("^--* Step  *([0-9]*) -.*$")
            re_TotEng = re.compile("^TotEng  *=  *([0-9.-]*) .*$")
            fdata = open(datafile, 'r')
            timestep_data = -1

        # parse LAMMPS dump file
        step = 0
        energy = None

        if hasattr(infile, 'readline'):
            f = infile
            close_file = False
        else:
            f = open(infile, 'r')
            close_file = True

        line = f.readline()
        while (line):
            if re_timestep.match(line):
                step += 1
                timestep = int(f.readline())
                if datafile:
                    line2 = True
                    while (timestep_data < timestep) and line2:
                        line2 = fdata.readline()
                        if re_Step.match(line2):
                            timestep_data = int(re_Step.match(line2).group(1))
                    if (timestep_data == timestep):
                        line2 = fdata.readline()
                        energy = float(re_TotEng.match(line2).group(1))
                    else:
                        energy = None
                natoms = 0
                bounds = ["", "", ""]
                avec = np.zeros([3, 3])
            elif re_natoms.match(line):
                natoms = int(f.readline())
                coords = np.empty([natoms, 3])
                itype = np.empty(natoms, dtype=int)
            elif re_bounds.match(line):
                line = line.split()
                bounds = line[-3:]
                if (len(line) > 6):
                    # tilded box:
                    line = f.readline().split()
                    (xlo_bound, xhi_bound, xy) = [float(el) for el in line]
                    line = f.readline().split()
                    (ylo_bound, yhi_bound, xz) = [float(el) for el in line]
                    line = f.readline().split()
                    (zlo_bound, zhi_bound, yz) = [float(el) for el in line]
                else:
                    # orthogonal box:
                    line = f.readline().split()
                    (xlo_bound, xhi_bound) = [float(el) for el in line]
                    line = f.readline().split()
                    (ylo_bound, yhi_bound) = [float(el) for el in line]
                    line = f.readline().split()
                    (zlo_bound, zhi_bound) = [float(el) for el in line]
                    xy = xz = yz = 0.0
                xlo = xlo_bound - np.min([0.0, xy, xz, xy+xz])
                xhi = xhi_bound - np.max([0.0, xy, xz, xy+xz])
                ylo = ylo_bound - np.min([0.0, yz])
                yhi = yhi_bound - np.max([0.0, yz])
                zlo = zlo_bound
                zhi = zhi_bound
                avec = 0.0
                avec = np.array([[xhi - xlo, 0.0,       0.0],
                                 [xy,        yhi - ylo, 0.0],
                                 [xz,        yz,        zhi - zlo]])
            elif re_atoms.match(line):
                for i in range(natoms):
                    atomline = f.readline().split()
                    iatom = int(atomline[0])
                    itype[iatom-1] = int(atomline[1]) - 1
                    coords[iatom-1] = [float(el) for el in atomline[2:]]
                if (bounds != ["pp", "pp", "pp"]):
                    avec = []
                if (step == 1):
                    struc = AtomicStructure(
                        coords.copy(), itype, typenames=typenames,
                        avec=avec, fractional=True, energy=energy)
                else:
                    struc.add_frame(coords.copy(), avec=avec, fractional=True,
                                    energy=energy)
            line = f.readline()

        if close_file:
            f.close()

        if datafile:
            fdata.close()

        self._amend(struc, **kwargs)
        return struc
