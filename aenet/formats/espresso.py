#!/usr/bin/env python

"""
I/O routines for the Quantum Espresso input file format.

"""

import sys
import re
import json
import numpy as np

from .. import units
from ..geometry import AtomicStructure
from ..exceptions import ArgumentError
from .parser_abc import ParserABC

__author__ = "Nongnuch Artrith"
__date__ = "2013-03-05"


def namelist2dict(string):
    s = string.strip()
    s = re.sub(r"!.*\n", r"\n", s)
    s = re.sub(r"!.*$", r"", s)
    s = re.sub(r"\n", r", ", s)
    s = re.sub(r" *,[, ]*", r", ", s)
    s = re.sub(r" *([^ ]*) *= *([^ ,]*)", r' "\g<1>": \g<2>', s)
    s = re.sub(r"&[a-zA-Z0-9,]* *", r"{\n", s)
    s = re.sub(r"[, ]*/", r"\n}", s)
    s = re.sub(r"'", r'"', s)
    s = re.sub(r", ", r",\n", s)
    s = re.sub(r".true.", "true", s, re.I)
    s = re.sub(r".false.", "false", s, re.I)
    return json.loads(s)


class EspressoParser(ParserABC):
    def __init__(self):
        self.name = 'espresso'
        self.description = 'Quantum Espresso input format'
        self.extensions = ['espresso']
        self.default_file_names = []

    def read(self, infile, **kwargs):
        """
        Parse file with atomic coordinates in the Quantum ESPRESSO format.

        Arguments:
          infile   name of the input file

        Returns:
          an instance of the AtomicStructure class
        """

        self._check_amend_args(**kwargs)

        re_system = re.compile("^ *&system", re.IGNORECASE)
        re_positions = re.compile("^ *atomic_positions", re.IGNORECASE)
        re_cell = re.compile("^ *cell_parameters", re.IGNORECASE)

        def parse_system(line, fp):
            re_end = re.compile(r"/ *$")
            namelist = line
            while True:
                line = fp.readline()
                namelist += line
                if re_end.search(line):
                    break
            params = namelist2dict(namelist)
            if "nat" in params:
                natoms = params["nat"]
            else:
                natoms = 0
            if "ntyp" in params:
                ntypes = params["ntyp"]
            else:
                ntypes = 0
            if "ibrav" in params:
                ibrav = params["ibrav"]
            else:
                ibrav = 0
            try:
                a = params["celldm(1)"]
            except KeyError:
                try:
                    a = params["a"]*units.Ang2Bohr
                except KeyError:
                    a = 1.0
            if ibrav == 0:
                avec = None
            elif ibrav == 1:  # cubic P
                avec = np.identity(3)*a
            elif ibrav == 2:  # cubic F (FCC)
                avec = np.array([[-1.0, -.0, 1.0],
                                 [0.0, 1.0, 1.0],
                                 [-1.0, 1.0, 0.0]])*0.5*a
            elif ibrav == 3:  # cubic I (BCC)
                avec = np.array([[1.0, 1.0, 1.0],
                                 [-1.0, 1.0, 1.0],
                                 [-1.0, -1.0, 1.0]])*0.5*a
            elif ibrav == 4:  # Hexagonal and Trigonal P; celldm(3)=c/a
                c_over_a = params["celldm(3)"]
                avec = np.array([[1.0, 0.0, 0.0],
                                 [-0.5, np.sqrt(3.0)/2.0, 0.0],
                                 [0.0, 0.0, c_over_a]])*a
            elif ibrav == 5:  # Trigonal R, 3fold axis c; celldm(4)=cos(alpha)
                ca = params["celldm(4)"]
                tx = np.sqrt((1.0 - ca)/2.0)
                ty = np.sqrt((1.0 - ca)/6.0)
                tz = np.sqrt((1.0 + 2.0*ca)/3.0)
                avec = np.array([[tx, -ty, tz],
                                 [0.0, 2.0*ty, tz],
                                 [-tx, -ty, tz]])*a
            elif ibrav == -5:  # Trigonal R, <111>; celldm(4)=cos(alpha)
                ap = a/np.sqrt(3.0)
                ca = params["celldm(4)"]
                tx = np.sqrt((1.0 - ca)/2.0)
                ty = np.sqrt((1.0 - ca)/6.0)
                tz = np.sqrt((1.0 + 2.0*ca)/3.0)
                u = tz - 2.0*np.sqrt(2.0)*ty
                v = tz + np.sqrt(2.0)*ty
                avec = np.array([[u, v, v],
                                 [v, u, v],
                                 [v, v, u]])*ap
            elif ibrav == 6:  # Tetragonal P (st); celldm(3)=c/a
                c_over_a = params["celldm(3)"]
                avec = np.array([[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, c_over_a]])*a
            elif ibrav == 7:  # Tetragonal I (bct); celldm(3)=c/a
                c_over_a = params["celldm(3)"]
                avec = np.array([[1.0, -1.0, c_over_a],
                                 [1.0, 1.0, c_over_a],
                                 [-1.0, -1.0, c_over_a]])*0.5*a
            elif ibrav == 8:  # Orthorhombic P; celldm(2)=b/a; celldm(3)=c/a
                b_over_a = params["celldm(2)"]
                c_over_a = params["celldm(3)"]
                avec = np.array([[1.0, 0.0, 0.0],
                                 [0.0, b_over_a, 0.0],
                                 [0.0, 0.0, c_over_a]])*a
            elif ibrav == 9:  # Orthorhombic base-centered(bco)
                b = params["celldm(2)"]*a
                c = params["celldm(3)"]*a
                avec = np.array([[a/2.0, b/2.0, 0.0],
                                 [-a/2.0, b/2.0, 0.0],
                                 [0.0, 0.0, c]])
            # There is an error in the QE docs for ibrav == -9
            # elif ibrav == -9:  # Orthorhombic base-centered(bco)
            #     b = params["celldm(2)"]*a
            #     c = params["celldm(3)"]*a
            #     avec = np.array([[a/2.0, -b/2.0, 0.0],
            #                      [a/2.0, b/2.0, 0.0],
            #                      [0.0, 0.0, c]])
            elif ibrav == 10:  # Orthorhombic face-centered
                b = params["celldm(2)"]*a
                c = params["celldm(3)"]*a
                avec = np.array([[a/2.0, 0.0, c/2.0],
                                 [a/2.0, b/2.0, 0.0],
                                 [0.0, b/2.0, c/2.0]])
            elif ibrav == 11:  # Orthorhombic body-centered
                b = params["celldm(2)"]*a
                c = params["celldm(3)"]*a
                avec = np.array([[a/2.0, b/2.0, c/2.0],
                                 [-a/2.0, b/2.0, c/2.0],
                                 [-1/2.0, -b/2.0, c/2.0]])
            elif ibrav == 12:  # Monoclinic P, unique axis c
                b = params["celldm(2)"]*a
                c = params["celldm(3)"]*a
                cosab = params["celldm(4)"]
                sinab = np.sqrt(1.0 - cosab**2)
                avec = np.array([[a, 0.0, 0.0],
                                 [b*cosab, b*sinab, 0.0],
                                 [0.0, 0.0, c]])
            elif ibrav == -12:  # Monoclinic P, unique axis b
                b = params["celldm(2)"]*a
                c = params["celldm(3)"]*a
                cosac = params["celldm(5)"]
                sinac = np.sqrt(1.0 - cosac**2)
                avec = np.array([[a, 0.0, 0.0],
                                 [0.0, b, 0.0],
                                 [c*sinac, 0.0, c*cosac]])
            elif ibrav == 13:  # Monoclinic base-centered
                b = params["celldm(2)"]*a
                c = params["celldm(3)"]*a
                cosab = params["celldm(4)"]
                sinab = np.sqrt(1.0 - cosab**2)
                avec = np.array([[a/2.0, 0.0, -c/2.0],
                                 [b*cosab, b*sinab, 0.0],
                                 [a/2.0, 0.0, c/2.0]])
            elif ibrav == 14:  # Triclinic
                b = params["celldm(2)"]*a
                c = params["celldm(3)"]*a
                cosab = params["celldm(4)"]
                cosac = params["celldm(5)"]
                cosbc = params["celldm(6)"]
                sinab = np.sqrt(1.0 - cosab**2)
                a31 = c*cosac
                a32 = c*(cosbc - cosac*cosab)/sinab
                a33 = c*np.sqrt(1.0 + 2.0*cosbc*cosac*cosab
                                - cosbc**2 - cosac**2 - cosab**2)/sinab
                avec = np.array([[a, 0.0, 0.0],
                                 [b*cosab, b*sinab, 0.0],
                                 [a31, a32, a33]])
            else:
                raise ArgumentError(
                    "ibrav = {} not implemented.".format(ibrav))
            if avec is not None:
                avec *= units.Bohr2Ang
            a *= units.Bohr2Ang
            return (avec, a, ibrav, ntypes, natoms)

        def parse_coordinates(line, fp, natoms, alat):
            if None in [natoms, alat]:
                raise ArgumentError(
                    "Incomplete espresso input file.  Conversion aborted.")
            if re.search("crystal", line, re.IGNORECASE):
                fractional = True
                scale = 1.0
            elif re.search("bohr", line, re.IGNORECASE):
                fractional = False
                scale = units.Bohr2Ang
            elif re.search("angstrom", line, re.IGNORECASE):
                fractional = False
                scale = 1.0
            else:
                fractional = False
                scale = alat
            coords = []
            types = []
            for i in range(natoms):
                line = fp.readline()
                types.append(line.split()[0])
                coords.append([float(el) for el in line.split()[1:4]])
            coords = np.array(coords)*scale
            types = np.array(types)
            return (coords, types, fractional)

        def parse_cell(line, fp, alat, ibrav):
            if ibrav != 0:
                raise ArgumentError(
                    "ibrav = {}, but cell parameters given".format(ibrav))
            if re.search("bohr", line, re.IGNORECASE):
                scale = units.Bohr2Ang
            elif re.search("angstrom", line, re.IGNORECASE):
                scale = 1.0
            else:
                scale = alat
            avec = []
            for i in range(3):
                line = fp.readline()
                avec.append([float(el) for el in line.split()[0:4]])
            avec = np.array(avec)*scale
            return avec

        avec = []
        coords = []
        types = []
        natoms = None
        ntypes = None
        alat = None

        if hasattr(infile, 'readline'):
            fp = infile
            close_file = False
        else:
            fp = open(infile, 'r')
            close_file = False

        while True:
            line = fp.readline()
            if not line:
                break
            if re_system.search(line):
                (avec, alat, ibrav, ntypes, natoms) = parse_system(line, fp)
            elif re_positions.search(line):
                (coords, types, fractional
                 ) = parse_coordinates(line, fp, natoms, alat)
            elif re_cell.search(line):
                avec = parse_cell(line, fp, alat, ibrav)

        if close_file:
            fp.close()

        if len(avec) > 0:
            avec = np.array(avec)

        struc = AtomicStructure(coords, types, avec=avec,
                                fractional=fractional)
        self._amend(struc, **kwargs)
        return struc

    def write(self, struc, outfile=None, frame=-1, **kwargs):
        """
        Write structure information in the PWSCF/Espresso input format.
        """

        for kw in kwargs:
            sys.stderr.write("Warning: unsupported argument: {}\n".format(kw))

        if outfile:
            f = open(outfile, 'w')
        else:
            f = sys.stdout

        if not struc.pbc:
            raise ArgumentError("The espresso format only supports "
                                "periodic structures")

        f.write("ATOMIC_POSITIONS (crystal)\n")
        for i in range(struc.natoms):
            coo = struc.fraccoo(i, frame=frame)
            f.write("{0:2s}  {1:20.12f}  {2:20.12f}  {3:20.12f}\n".format(
                    struc.types[i], coo[0], coo[1], coo[2]))
        f.write("\n")
        f.write("CELL_PARAMETERS\n")
        for v in struc.avec[frame]:
            v_Bohr = v*units.Ang2Bohr
            f.write("{0:20.12f} {1:20.12f} {2:20.12f}\n".format(*v_Bohr))

        if outfile:
            f.close()
