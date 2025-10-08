#!/usr/bin/env python3

from aenet.commandline.tools import AenetToolABC

import sys
import numpy as np

from ..formats import formats
from .. import util
from ..io import structure
from ..geometry import AtomicStructure

__author__ = "Alexander Urban, Nongnuch Artrith"
__date__ = "2013-06-06"
__email__ = "aenet@atomistic.net"


class Sconv(AenetToolABC):
    """
    Conversion between atomic structure formats.

    """

    def _set_arguments(self):
        self.parser.add_argument(
            "input_file",
            help="Path to the input file; use '-' for standard in.",
            default="-",
            nargs="?")

        self.parser.add_argument(
            "output_file",
            help="Path to the output file; omit for standard out.",
            default=None,
            nargs="?")

        self.parser.add_argument(
            "--input-format", "-i",
            help="Input atomic structure format.",
            default=None,
            metavar='FORMAT',
            choices=[f for f in formats if formats[f].readable])

        self.parser.add_argument(
            "--output-format", "-o",
            help="Output atomic structure format.",
            default=None,
            metavar='FORMAT',
            choices=[f for f in formats if formats[f].writable])

        self.parser.add_argument(
            "--formats",
            help="Print list of supported formats and exit.",
            action="store_true")

        self.parser.add_argument(
            "--info",
            help="Print information about input structure and exit.",
            action="store_true")

        self.parser.add_argument(
            "--frame",
            help="Select a single frame from a trajectory.",
            type=int,
            default=None,
            metavar='FRAME_NUM')

        self.parser.add_argument(
            "--typenames",
            help="List of atom type names or name of a file containing "
                 "this information (exact meaning depends on input format).",
            type=str,
            default=[],
            nargs="+",
            metavar='TYPE')

        self.parser.add_argument(
            "--datafile", "-d",
            help="File with additional data; usually output file "
                 "(exact meaning depends on input format).",
            default=None,
            metavar='PATH')

        self.parser.add_argument(
            "--align",
            help="Align frames in periodic MD trajectories.",
            action="store_true",
            default=False)

        self.parser.add_argument(
            "--split",
            help="Split trajectory/relaxtion steps into separate files.",
            action="store_true")

        self.parser.add_argument(
            "--rotate",
            help="Rotate lattice basis to standard orientation.",
            action="store_true")

        self.parser.add_argument(
            "--sort",
            help="Sort atomic coordinates.",
            action="store_true")

        self.parser.add_argument(
            "--rotate-angle",
            help="Rotate structure a given angle around one of its axes. "
                 "AXIS is 1, 2, or 3; the ANGLE is in degrees.  "
                 "Instead of an angle, a VECTOR of three components can "
                 "be specified to which the chosen axis will be aligned",
            type=str,
            default=None,
            nargs="+",
            metavar=('AXIS', 'ANGLE_OR_VECTOR'))

        self.parser.add_argument(
            "--wrap",
            help="Wrap coordinates of periodic structure to unit cell.",
            action="store_true")

        self.parser.add_argument(
            "--scale",
            help="Scale entire structure by a factor (lattice constant "
                 "for PBC). If three numbers are specified, the three "
                 "lattice vectors are scaled independently.",
            type=float,
            default=None,
            metavar='FACTOR',
            nargs="+")

        self.parser.add_argument(
            "--shift",
            help="Shift entire structure by a Cartesian vector. "
                 "Alternatively, shift to (0, 0, 0) using 'origin' or "
                 "to the cell center with 'box'.  'origin' and 'box' "
                 "take an atom index (1...N) as optional second "
                 "argument (default is the geometric center).",
            default=None,
            nargs="+")

        self.parser.add_argument(
            "--cut",
            help="Cut a spherical region from an atomic structure "
                 "centered around ATOM (1...N) with a radius of CUTOFF "
                 "in Angstroms.",
            default=None,
            nargs=2,
            metavar=('atom', 'cutoff'))

        self.parser.add_argument(
            "--supercell",
            help="Multiply input cell to create a supercell "
                 "(only periodic structures).",
            type=int,
            default=None,
            nargs=3,
            metavar=('n1', 'n2', 'n3'))

        self.parser.add_argument(
            "-a",
            help="New first lattice vector in multiples of original "
                 "lattice vectors.",
            type=float,
            default=[1.0, 0.0, 0.0],
            nargs=3,
            metavar=('a1', 'a2', 'a3'))

        self.parser.add_argument(
            "-b",
            help="New second lattice vector in multiples of original "
                 "lattice vectors.",
            type=float,
            default=[0.0, 1.0, 0.0],
            nargs=3,
            metavar=('b1', 'b2', 'b3'))

        self.parser.add_argument(
            "-c",
            help="New third lattice vector in multiples of original "
                 "lattice vectors.",
            type=float,
            default=[0.0, 0.0, 1.0],
            nargs=3,
            metavar=('c1', 'c2', 'c3'))

        self.parser.add_argument(
            "--vac",
            help="Add vacuum (Angstrom) by elongating the third (c) "
                 "lattice vector.",
            type=float,
            default=None)

        self.parser.add_argument(
            "--fix-atoms",
            help="Fix selected atoms.  Use list of integers or ranges such "
                 "as '1:50', or specify Cartesian ranges in one of the "
                 "lattice directions.  Example: 'z -3.0 1.0' will fix "
                 "all atoms with third lattice coordinate between -3 and 1.",
            type=str,
            default=None,
            nargs="+")

        self.parser.add_argument(
            "--remove-atoms",
            help="Remove selected atoms.  Use list of integers or ranges such "
                 "as '1 2 3:10' (to remove atoms 1 through 10).  The atom "
                 "index starts with 1.",
            type=str,
            default=None,
            nargs="+")

        self.parser.add_argument(
            "--input-options", "--io",
            help="Further input options (supported formats in parenthesis): ",
            default=[],
            nargs="+",
            metavar='OPTION')

        self.parser.add_argument(
            "--output-options", "--oo",
            help="Further input options (supported formats in parenthesis): "
                 "numeric_species (xsf)",
            default=[],
            nargs="+",
            metavar='OPTION')

    def _man(self):
        return """
        Convert between atomic structure formats, process output files
        from electronic structure calculations, and perform operations
        on atomic structures.

        Not all of the backends for the various file formats support
        each of the command line switches.  A warning will be printed,
        if an unsupported option has been selected.

        """

    def run(self, args):
        self.sconvert(infile=args.input_file,
                      outfile=args.output_file,
                      infrmt=args.input_format,
                      outfrmt=args.output_format,
                      print_formats=args.formats,
                      print_info=args.info,
                      frame=args.frame,
                      typenames=args.typenames,
                      datafile=args.datafile,
                      align_frames=args.align,
                      split=args.split,
                      rotate=args.rotate,
                      sort=args.sort,
                      rotate_angle=args.rotate_angle,
                      wrap=args.wrap,
                      scale=args.scale,
                      shift=args.shift,
                      cut=args.cut,
                      supercell=args.supercell,
                      transform=np.array([args.a, args.b, args.c]),
                      vac=args.vac,
                      fix_atoms=args.fix_atoms,
                      remove_atoms=args.remove_atoms,
                      input_options=args.input_options,
                      output_options=args.output_options)

    def sanity_check_arguments(self, infile, infrmt, outfile, outfrmt,
                               print_info, frame, split):

        if (infile == "-" and not infrmt):
            sys.stderr.write("Error: reading from stdin requires an "
                             "input format.\n")
            sys.exit()

        if not (outfile or outfrmt or print_info):
            sys.stderr.write("Error: neither output file nor format given.\n")
            sys.exit()

        if (frame and split):
            sys.stderr.write(
                "Error: incompatible options: --frame, --split.\n")
            sys.exit()

        if (split and not outfile):
            sys.stderr.write("Error: options --split requires an output "
                             "file name.\n")
            sys.exit()

    def read_input_structure(self, infile, infrmt, input_options,
                             typenames, datafile):
        """ parse input options and read input structure """
        kwargs = {}
        for o in input_options:
            kw = o.split("=")
            kw.append(True)
            kwargs[kw[0]] = kw[1]
        if (len(typenames) > 0):
            kwargs['typenames'] = typenames
        if datafile:
            kwargs['datafile'] = datafile
        struc = structure.read_safely(infile, frmt=infrmt, **kwargs)
        return struc

    def print_structure_info(self, struc, infile):
        if (infile == "-"):
            print("\n Atomic structure read from standard in.")
        else:
            print("\n Atomic structure read from file {}.".format(infile))
        print(struc)

    def fix_atomic_coordinates(self, struc, fix_atoms):
        if fix_atoms[0] == "x":
            x0 = float(fix_atoms[1])
            x1 = float(fix_atoms[2])
            fix_atoms = [i+1 for i in range(struc.natoms)
                         if x0 <= struc.coords[-1][i][0] <= x1]
        elif fix_atoms[0] == "y":
            y0 = float(fix_atoms[1])
            y1 = float(fix_atoms[2])
            fix_atoms = [i+1 for i in range(struc.natoms)
                         if y0 <= struc.coords[-1][i][1] <= y1]
        elif fix_atoms[0] == "z":
            z0 = float(fix_atoms[1])
            z1 = float(fix_atoms[2])
            fix_atoms = [i+1 for i in range(struc.natoms)
                         if z0 <= struc.coords[-1][i][2] <= z1]
        else:
            fix_atoms = util.csv2list(fix_atoms)
        struc.set_fixed_atoms(fix_atoms, by_index=True)
        return struc

    def transform_cell(self, struc, transform, **kwargs):
        if not struc.pbc:
            sys.stderr.write(
                "Error: no cell to transform in isolated structure.\n")
            sys.exit()
        if 'frame' not in kwargs:
            if (struc.nframes > 1):
                sys.stderr.write(
                    "Warning: only the last frame will be transformed.\n")
            f = -1
        else:
            f = kwargs['frame']
        (avec, coords, types
         ) = util.transform_cell(
             struc.avec[f], struc.cart2frac(struc.coords[f], frame=f),
             struc.types, transform)
        struc = AtomicStructure(coords, types, avec=avec, fractional=True)
        return struc

    def make_supercell(self, struc, supercell, **kwargs):
        if 'frame' not in kwargs:
            struc = struc.supercell(supercell)
        else:
            struc = struc.supercell(supercell, frame=kwargs['frame'])
        return struc

    def standard_cell(self, struc, **kwargs):
        if 'frame' in kwargs:
            frame = kwargs['frame']
            frac_coords = struc.cart2frac(struc.coords[frame], frame=frame)
            struc.avec[frame] = util.standard_cell(struc.avec[frame])
            struc.bvec[frame] = np.linalg.inv(struc.avec[frame])
            struc.coords[frame] = struc.frac2cart(frac_coords, frame=frame)
        else:
            for f in range(struc.nframes):
                frac_coords = struc.cart2frac(struc.coords[f], frame=f)
                struc.avec[f] = util.standard_cell(struc.avec[f])
                struc.bvec[f] = np.linalg.inv(struc.avec[f])
                struc.coords[f] = struc.frac2cart(frac_coords, frame=f)
        return struc

    def rotate_structure(self, struc, rotate_angle, **kwargs):
        iaxis = int(rotate_angle[0]) - 1
        if struc.pbc:
            if 'frame' in kwargs:
                axis = struc.avec[kwargs['frame']][iaxis]
            else:
                axis = struc.avec[-1][iaxis]
        else:
            axis = np.identity(3)[iaxis]
        if len(rotate_angle) == 2:
            angle = float(rotate_angle[1])
            struc2 = struc.rotate(axis=axis, angle=angle, degrees=True)
        elif len(rotate_angle) >= 4:
            v2 = axis
            v1 = [float(a) for a in rotate_angle[1:4]]
            struc2 = struc.rotate(vectors=(v1, v2))
        return struc2

    def translate_structure(self, struc, shift, **kwargs):
        shift_frame = kwargs['frame'] if 'frame' in kwargs else -1
        if len(shift) == 1:
            atom = None
            shift = shift[0]
        elif len(shift) == 2:
            atom = int(shift[1]) - 1
            shift = shift[0]
        elif len(shift) == 3:
            atom = None
            shift = [float(s) for s in shift]
        else:
            sys.stderr.write("Error: invalid translation specified: ")
            sys.stderr.write(str(shift) + "\n")
            sys.exit()
        struc2 = struc.translate(shift, atom=atom, frame=shift_frame)
        return struc2

    def scale_structure(self, struc, scale, **kwargs):
        if len(scale) == 1:
            S = np.array([[scale[0], 0.0, 0.0],
                          [0.0, scale[0], 0.0],
                          [0.0, 0.0, scale[0]]])
        elif len(scale) == 3:
            S = np.array([[scale[0], 0.0, 0.0],
                          [0.0, scale[1], 0.0],
                          [0.0, 0.0, scale[2]]])
        else:
            raise ValueError("'--scale' takes exactly 1 or 3 values, "
                             "{} are given".format(len(scale)))

        if 'frame' in kwargs:
            frames = [kwargs['frame']]
        else:
            frames = range(struc.nframes)

        for frame in frames:
            if struc.pbc:
                frac_coords = struc.cart2frac(struc.coords[frame], frame=frame)
                struc.avec[frame] = np.dot(S, struc.avec[frame])
                struc.bvec[frame] = np.linalg.inv(struc.avec[frame])
                struc.coords[frame] = struc.frac2cart(frac_coords, frame=frame)
            else:
                struc.coords[frame] = np.dot(struc.coords[frame], S)
        return struc

    def write_output_structure(self, struc, outfile, outfrmt, split,
                               frame, **kwargs):
        if split:
            outfile = outfile.split('.')
            ext = outfile[-1]
            fname = outfile[0]
            for o in outfile[1:-1]:
                fname += "." + o
            s = len("{}".format(struc.nframes))
            s = "-%0{}d".format(s)
            for i in range(struc.nframes):
                outfile = fname + (s % (i+1)) + "." + ext
                kwargs['frame'] = i
                structure.write_safely(
                    struc, filename=outfile, frmt=outfrmt, **kwargs)
        else:
            if frame is not None:
                kwargs['frame'] = frame
            structure.write_safely(
                struc, filename=outfile, frmt=outfrmt, **kwargs)

    def sconvert(self, infile="-", outfile=None, infrmt=None, outfrmt=None,
                 print_formats=False, print_info=False, frame=None,
                 typenames=[], datafile=None, align_frames=False,
                 split=False, rotate=False, sort=False, rotate_angle=None,
                 wrap=False, scale=None, shift=None, cut=None, supercell=None,
                 transform=np.identity(3), vac=None, fix_atoms=None,
                 remove_atoms=None, input_options=[], output_options=[]):

        if print_formats:
            structure.print_supported_formats()
            sys.exit()

        self.sanity_check_arguments(infile, infrmt, outfile, outfrmt,
                                    print_info, frame, split)

        if (infile == "-"):
            infile = sys.stdin

        struc = self. read_input_structure(
            infile, infrmt, input_options, typenames, datafile)

        if align_frames:
            struc.align_all_frames()

        # just print information about input structure and exit
        if print_info:
            self.print_structure_info(struc, infile)
            sys.exit()

        # transformation of the input structure
        kwargs = {}
        if frame is not None:
            frame = max(min(frame, struc.nframes-1), -struc.nframes)
            kwargs['frame'] = frame
        if fix_atoms is not None:
            struc = self.fix_atomic_coordinates(struc, fix_atoms)
        if remove_atoms is not None:
            del_idx = np.array(util.csv2list(remove_atoms), dtype=int) - 1
            struc = struc.remove_atoms(del_idx)
        if not np.array_equal(transform, np.identity(3)):
            struc = self.transform_cell(struc, transform, **kwargs)
        if supercell is not None:
            struc = self.make_supercell(struc, supercell, **kwargs)
        if rotate and struc.pbc:
            struc = self.standard_cell(struc, **kwargs)
        if rotate_angle is not None:
            struc = self.rotate_structure(struc, rotate_angle, **kwargs)
        if shift is not None:
            struc = self.translate_structure(struc, shift, **kwargs)
        if scale is not None:
            struc = self.scale_structure(struc, scale, **kwargs)
        if cut is not None:
            struc = struc.get_neighbors(i=int(cut[0])-1, cutoff=float(cut[1]))
        if vac is not None:
            struc = struc.add_vacuum(vac)
        if wrap:
            struc.wrap_to_cell()
        if sort:
            struc.sort()

        # write output structure in requested format
        kwargs = {}
        for o in output_options:
            kw = o.split("=")
            kw.append(True)
            kwargs[kw[0]] = kw[1]
        self.write_output_structure(
            struc, outfile, outfrmt, split, frame, **kwargs)


if __name__ == "__main__":
    tool = Sconv()
    args = tool.parser.parse_args()
    tool.run(args)
