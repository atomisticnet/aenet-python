#!/usr/bin/env python

"""
A type to represent the most important atomic structure data.

"""

import numpy as np
import sys

from .exceptions import ArgumentError, IncompatibleStructureError
from . import util

__author__ = "Alexander Urban, Nongnuch Artrith"
__date__ = "2013-06-05"


class AtomicStructure(object):
    """
    A container class for atomic structure information.

    Class Attributes:

      fixed      list of Boolean values, denoting fixed Cartesian components
                 of atomic coordinates
      comments   list of comments known for this structure

      coords[i]  list of atomic Cartesian coordinates of all atoms in the
                 structure and for the i-th frame
      energy[i]  total energy (if known) of the configuration in the i-th
                 frame
      forces[i]  list of Cartesian atomic forces for the configuration in
                 the i-th frame

      pbc        True, iff the structure is periodic
      avec[i]    matrix of lattice vectors for the i-th frame
      bvec[i]    inverse of the matrix of lattice vectors for the i-th frame

      types        list of atomic species of each atom in the structure
      typeID       type ID (numeric identifier) of atomic species of each
                   atom
      names_known  True, if chemical symbols of the atomic specie species
                   are known (and not only IDs)
      typenames[i] chemical symbol for atom type ID i
    """

    def __init__(self, coords, types, typenames=[], avec=[],
                 fractional=False, energy=None, forces=None, fixed=None):
        """
        Atomic structure information.

        Arguments:
          coords[i][j]   j-th component of the atomic coordinates of
                         atom i; Per default Cartesian coordinates are
                         expected.  This behavior can be changed with
                         the 'fractional' argument.
          types[i]       atomic species symbol or ID of atom i
          typenames[i]   symbol corresponding to atomic species ID i,
                         if 'types' contains just IDs
          avec[i][j]     j-th component of the i-th lattice vector
          fractional     if True the input coordinates are parsed as
                         fractional lattice coordinates, i.e., lattice
                         vectors must be defined
          fixed          if fixed[i][j] is True, then the j-th coordinate
                         of the i-th atom is fixed in relaxations or dynamics
        """

        self.fixed = []
        self.comments = []
        self.coords = []
        self.energy = []
        self.forces = []
        self.types = []
        self.typeID = []
        self.names_known = False
        self.typenames = []

        if avec is None or (len(avec) == 0):
            self.avec = None
            self.bvec = None
            self.pbc = False
        else:
            self.avec = [np.array(avec)]
            self.bvec = [np.linalg.inv(avec)]
            self.pbc = True

        if fractional and not self.pbc:
            raise ArgumentError("Fractional coordinates require "
                                "lattice vectors.")

        self.types = np.array(types)
        if (typenames is not None) and (len(typenames) > 0):
            # type names specified separately
            # --> assume types are IDs
            self.typenames = typenames
            self.names_known = True
            self.typeID = np.array(self.types)
            self.types = np.array([self.typenames[i] for i in self.typeID])
        else:
            if (isinstance(self.types[0], str)):
                # types are known and now stored in self.types
                # generate unique type IDs 1, ..., N
                self.typenames = []
                for t in self.types:
                    if not (t in self.typenames):
                        self.typenames.append(t)
                self.typeID = np.array(
                    [self.typenames.index(t) for t in self.types])
                self.names_known = True
            else:
                self.typenames = None
                self.names_known = False
                self.typeID = self.types
                self.types = np.empty(self.typeID.shape, dtype=str)
                for i in range(len(self.types)):
                    self.types[i] = str(self.typeID[i])

        if fractional:
            self.coords = [self.frac2cart(coords)]
        else:
            self.coords = [np.array(coords)]

        self.energy = [energy]

        if forces is not None:
            self.forces = [np.array(forces)]
        else:
            self.forces = [[]]

        if fixed is None or len(fixed) != self.natoms:
            # no fixed coordinates:
            self.fixed = np.zeros(self.coords[0].shape, dtype=bool)
        else:
            self.fixed = np.array(fixed, dtype=bool)

    @classmethod
    def from_pymatgen_structure(cls, structure, **kwargs):
        coords = structure.frac_coords[:]
        types = [str(s.symbol) for s in structure.species]
        avec = structure.lattice.matrix
        return cls(coords, types, avec=avec, fractional=True, **kwargs)

    @classmethod
    def from_ase_atoms(cls, atoms, **kwargs):
        structure = None
        if hasattr(atoms, "positions"):
            trajec = [atoms]
        else:
            trajec = atoms
        pbc = all(trajec[-1].get_pbc())
        if not pbc:
            avec = None
        for frame in trajec:
            if pbc:
                avec = atoms.cell.array
            coords = frame.positions
            types = frame.get_chemical_symbols()
            try:
                energy = frame.get_potential_energy()
            except RuntimeError:
                energy = None
            try:
                forces = frame.get_forces()
            except (RuntimeError, ValueError):
                forces = None
            if structure is None:
                structure = cls(coords, types, avec=avec,
                                fractional=False, energy=energy,
                                forces=forces, **kwargs)
            else:
                structure.add_frame(coords, avec=avec, fractional=False,
                                    energy=energy, forces=forces)
        return structure

    def __str__(self):
        """
        Structure information based on the final frame of a structure
        collection (if there are more than one).

        """
        ostr = "\n"
        ostr += " Composition        : "
        ostr += " ".join(np.sort(["{}{}".format(k.strip(), self.composition[k])
                                  for k in self.composition])) + "\n"
        ostr += " Number of atoms    : {}\n".format(self.natoms)
        ostr += " Number of species  : {}\n".format(self.ntypes)
        if (self.nframes > 1):
            ostr += " Number of frames   : {}\n".format(self.nframes)
            ostr += "\n Final configuration\n"
        if self.energy[-1] is not None:
            ostr += " Total energy       : {} eV\n".format(self.energy[-1])
        ostr += " Geometric center   : "
        ostr += "{:.8f} {:.8f} {:.8f} (Ang)\n".format(*self.geometric_center())
        if not self.pbc and self.natoms <= 500:
            ostr += " Diameter           : "
            ostr += "{:.3f} (Ang)\n".format(self.diameter())
        if self.pbc:
            ostr += " Unit cell volume   : {:.3f} Ang^3\n".format(
                self.cellvolume())
            (avec, a, b, c, alpha, beta, gamma, ab, ac, bc
             ) = self.standard_cell(angles=True, areas=True)
            ostr += (" Cell parameters    : " +
                     "{:.8f} {:.8f} {:.8f} ".format(a, b, c) +
                     "{:.8f} {:.8f} {:.8f}\n").format(alpha, beta, gamma)
            ostr += (" Areas (ab, bc, ac) : " +
                     "{:.8f} {:.8f} {:.8f} Ang^2\n".format(ab, ac, bc))
            ostr += "\n Lattice vectors\n\n"
            for i in range(3):
                ostr += " a{} = {:15.8f}  {:15.8f}  {:15.8f}\n".format(
                    i+1, *self.avec[-1][i])
        if self.natoms <= 50:
            ostr += "\n Cartesian coordinates\n\n"
            for i in range(len(self.coords[-1])):
                if self.names_known:
                    ostr += " {:2s} ".format(self.types[i])
                else:
                    ostr += " {:2d} ".format(self.types[i])
                ostr += "  {:15.8f}  {:15.8f}  {:15.8f}\n".format(
                    *self.coords[-1][i])
        return ostr

    def __eq__(self, other):
        equal = False
        if isinstance(other, self.__class__):
            equal = True
            for i in range(self.nframes):
                equal = equal and np.all(self.coords[i] == other.coords[i])
            equal = equal and (self.pbc == other.pbc)
            if self.pbc:
                for i in range(self.nframes):
                    equal = equal and np.all(self.avec[i] == other.avec[i])
            equal = equal and (self.composition == other.composition)
        return equal

    def __neq__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        if other is None:
            return self
        if not isinstance(other, self.__class__):
            raise TypeError("Cannot add instances of {} and {}".format(
                self.__class__, other.__class__))
        if any(self.types != other.types) or (self.pbc != other.pbc):
            raise IncompatibleStructureError(
                "Cannot add incompatible structures.")
        sumstruc = self.copy()
        for i in range(other.nframes):
            if other.pbc:
                avec = other.avec[i]
            else:
                avec = None
            sumstruc.add_frame(other.coords[i], avec=avec,
                               energy=other.energy[i],
                               forces=other.forces[i])
        return sumstruc

    def __radd__(self, other):
        return self.__add__(other)

    def __and__(self, other):
        if other is None:
            return self
        if not isinstance(other, self.__class__):
            raise TypeError("Cannot combine instances of {} and {}".format(
                self.__class__, other.__class__))
        if ((self.pbc != other.pbc)
           or (self.nframes != other.nframes)):
            raise IncompatibleStructureError(
                "Cannot combine incompatible structures.")
        combined = None
        types = list(self.types) + list(other.types)
        fixed = list(self.fixed) + list(other.fixed)
        for i in range(other.nframes):
            coords = list(self.coords[i]) + list(other.coords[i])
            if self.pbc:
                avec = self.avec[i]
            if combined is None:
                combined = AtomicStructure(coords, types, avec=avec,
                                           fixed=fixed)
            else:
                combined.add_frame(coords, avec=self.avec[i])
        return combined

    def copy(self):
        """
        Return a copy of this instance.

        """
        if self.pbc:
            struc = AtomicStructure(self.coords[0], self.types,
                                    avec=self.avec[0],
                                    energy=self.energy[0],
                                    forces=self.forces[0],
                                    fixed=self.fixed)
            for f in range(1, self.nframes):
                struc.add_frame(self.coords[f], avec=self.avec[f],
                                energy=self.energy[f],
                                forces=self.forces[f])
        else:
            struc = AtomicStructure(self.coords[0], self.types,
                                    energy=self.energy[0],
                                    forces=self.forces[0],
                                    fixed=self.fixed)
            for f in range(1, self.nframes):
                struc.add_frame(self.coords[f],
                                energy=self.energy[f],
                                forces=self.forces[f])
        return struc

    @property
    def natoms(self):
        """Total number of atoms in the structure"""
        return len(self.coords[-1])

    @property
    def ncomments(self):
        """Total number comments available for this structure"""
        return len(self.comments)

    @property
    def nframes(self):
        """Total number of frames (configurations)"""
        return len(self.coords)

    @property
    def ntypes(self):
        """Number of different atomic species"""
        return np.max(self.typeID) + 1

    @property
    def composition(self):
        """Composition as dictionary"""
        if self.names_known:
            return {self.typenames[i]: self.natoms_of_typeID(i)
                    for i in range(self.ntypes)}
        else:
            return {i: self.natoms_of_typeID(i)
                    for i in range(self.ntypes)}

    def cellvolume(self, frame=-1):
        """Unit cell volume"""
        return np.linalg.det(self.avec[frame])

    def atoms_of_typeID(self, t):
        """ return atom indices of atoms of type ID t"""
        return np.arange(self.natoms, dtype=int)[self.typeID == t]

    def natoms_of_typeID(self, t):
        """Number of atoms of given species t"""
        return 0 if t not in self.typeID else len(self.types[self.typeID == t])

    def fraccoo(self, i, frame=-1, wrap=True):
        """ fractional coordinates of atom i

        Arguments:
          i        atomic index
          frame    select coordinates and lattice vectors from this frame
          wrap     if True, coordinates will be wrapped to [0,1)

        Returns:
          ndarray of length 3
        """
        coo = self.cart2frac(self.coords[frame][i], frame=frame)
        if wrap:
            coo = util.wrap_pbc(coo)
        return coo

    def geometric_center(self, frame=-1):
        """
        geometric center of coordinates

        Arguments:
          frame    frame number to calculate geometric center of

        Returns:
          ndarray of length 3
        """

        center = np.sum(self.coords[frame], 0)/self.natoms
        return center

    def diameter(self, frame=-1):
        """
        longest distance between any two atoms in the structure

        Note: The diameter is only defined for isolated structures.

        Arguments:
          frame    frame number to calculate geometric center of

        Returns:
          diameter in Angstroms

        """

        if self.pbc:
            sys.stderr.write("Warning: Diameter not defined for PBC.\n")
            return None

        dmax = 0.0
        for i in range(self.natoms):
            for j in range(i+1, self.natoms):
                vec = self.coords[frame][i] - self.coords[frame][j]
                dist = np.linalg.norm(vec)
                dmax = max(dmax, dist)
        return dmax

    def standard_cell(self, frame=-1, angles=False, areas=False):
        """
        Return lattice vectors in standard orientation, and optionally
        also the cell parameters.

        See util.standard_cell() for a full documentation.
        """

        return util.standard_cell(self.avec[frame], angles, areas)

    def sort(self):
        """
        Sort atomic coordinates.
        """
        types = []
        for it in range(self.ntypes):
            t = self.typenames[it]
            types += [t for i in range(self.natoms_of_typeID(it))]
        coords = []
        for f in range(self.nframes):
            frame_coords = []
            for it in range(self.ntypes):
                type_coords = self.coords[f][self.atoms_of_typeID(it)]
                idx = np.argsort(-type_coords[:, 2])
                type_coords = type_coords[idx]
                frame_coords.extend(type_coords)
            coords.append(np.array(frame_coords))
        self.types = np.array(types)
        self.coords = coords

    def rotate(self, vectors=None, axis=None, angle=None, degrees=False):
        """
        Rotate coordinates. The axis and angle of the rotation can either be
        directly specified, or in terms of two reference vectors.  In
        the second case, the axis is chosen perpendicular to the two
        given vectors, and the angle of rotation is the angle between
        the two vectors.  Center of the rotation is the origin.

        Arguments:
          vectors  list of two reference vectors v1 and v2
          axis     vector in the direction of the rotation axis
          angle    angle of the rotation; per default in radians
          degrees  If True, the rotation angle is specified in degrees;
                   default is radians

        Returns:
          A new AtomicStructure object with rotated cell/coordinates.

        See util.rotate() for complete documentation.

        """

        for i in range(self.nframes):
            if self.pbc:
                avec_rot = util.rotate(
                    self.avec[i], vectors=vectors, axis=axis,
                    angle=angle, degrees=degrees)
                coords_rot = self.cart2frac(self.coords[i], frame=i)
                fractional = True
            else:
                avec_rot = None
                coords_rot = util.rotate(
                    self.coords[i], vectors=vectors, axis=axis,
                    angle=angle, degrees=degrees)
                fractional = False
            if i == 0:
                structure = AtomicStructure(
                    coords_rot, self.types, avec=avec_rot,
                    fractional=fractional, energy=self.energy[i],
                    forces=self.forces[i])
            else:
                structure.add_frame(
                    coords_rot, avec=avec_rot, fractional=fractional,
                    energy=self.energy[i], forces=self.forces[i])

        return structure

    def translate(self, shift, atom=None, frame=-1):
        """
        Translate the entire structure by a given shift.

        Arguments:
          shift (vector or string)
              shift == 'origin'   shift geometric center to (0, 0, 0)
                    == 'box'      shift geometric center to (0.5, 0.5, 0.5)
                    == (a, b, c)  shift all coordinates by vector (a, b, c)
          atom (int)   index of a reference atom relative to which the
                       translation is specified; only if the shift vector
                       is not explicitly given
          frame (int)  number of the reference frame

        Returns:
          translated structure (AtomicStructure instance)

        """
        if atom is None:
            reference = self.geometric_center(frame=frame)
        else:
            reference = self.coords[frame][atom]
        if str(shift).strip().lower() == 'origin':
            T = -reference
        elif str(shift).strip().lower() == 'box':
            if not self.pbc:
                raise ArgumentError("Translation to box center is only "
                                    "possible for periodic structures.")
            T = (-reference
                 + self.frac2cart([0.5, 0.5, 0.5], frame=frame))
        else:
            T = np.array(shift)
        struc = self.copy()
        for f in range(struc.nframes):
            for coo in struc.coords[f]:
                coo += T
        return struc

    def add_frame(self, coords, avec=None, energy=None, forces=None,
                  fractional=False):
        if avec is not None and len(avec) == 0:
            avec = None
        if forces is not None and len(forces) == 0:
            forces = None
        if (fractional or avec is not None) and not self.pbc:
            raise ArgumentError("Can not add fractional coordinates to "
                                "an isolated structure.")

        if fractional and (avec is not None):
            self.coords.append(self.frac2cart(coords, avec))
        elif fractional:
            self.coords.append(self.frac2cart(coords))
        else:
            self.coords.append(np.array(coords))

        if avec is not None:
            self.avec.append(np.array(avec))
            self.bvec.append(np.linalg.inv(avec))
        else:
            if self.pbc:
                self.avec.append(self.avec[-1])
                self.bvec.append(self.bvec[-1])
        self.energy.append(energy)
        if forces is not None:
            self.forces.append(np.array(forces))
        else:
            self.forces.append(None)

    def add_comment(self, comment):
        self.comments.append(comment)

    def set_fixed_atoms(self, fixed, by_index=False):
        """
        Define which atoms should be fixed.
        Arguments:
          fixed: list or array with a boolean entry for each atom;
             if by_index==True, list of atomic indices starting with 1
          by_index: if true, treat 'fixed' as list of indices
        """
        if by_index:
            fixed = [(i+1 in fixed) for i in range(self.natoms)]
            fixed = [[f, f, f] for f in fixed]
        if len(fixed) == len(self.fixed):
            self.fixed = np.array(fixed, dtype='bool')
        else:
            raise ArgumentError("Number of atom fixes not equal to "
                                "number of atoms.")

    def align(self, s1, frame=-1):
        """
        Align structure to reference structure s1 by adding/subtracting
        lattice vectors to the coordinates (PBC only).

        Arguments:
          s1     reference structure (instance of AtomicStructure)
          frame  selected frame of trajectory (default: final step)

        """

        if (s1.natoms != self.natoms) or (s1.ntypes != self.ntypes):
            raise ArgumentError("Incompatible reference structure "
                                "for alignment.")

        if not (s1.pbc and self.pbc):
            return

        for i in range(self.natoms):
            vec = s1.coords[frame][i] - self.coords[frame][i]
            vec = np.round(self.cart2frac(vec, frame=frame))
            vec = self.frac2cart(vec, frame=frame)
            self.coords[frame][i] += vec

    def align_isolated_structures(self, s1, scale=False, frame=-1):
        """
        Align isolated structures with reference structure using the Kabsch
        algorithm.

        See also: https://en.wikipedia.org/wiki/Kabsch_algorithm
                  W. Kabsch, Acta Cryst. (1976) A32, 922-923;
                             Acta Cryst. (1978) A34, 827-828.

        Arguments:
          s1     reference structure (instance of AtomicStructure)
          scale  if True, the structure will be scaled to match the
                 dimensions of the reference structure
          frame  selected frame of trajectory (default: -1 = final step)

        Returns:
          rmsd

        """

        if (s1.natoms != self.natoms) or (s1.ntypes != self.ntypes):
            raise ArgumentError("Incompatible reference structure "
                                "for alignment.")

        if self.pbc or s1.pbc:
            return

        # translate geometric centers to origin
        center1 = self.geometric_center(frame=frame)
        center2 = s1.geometric_center(frame=frame)
        coo1 = np.asarray(self.coords[frame]) - center1
        coo2 = np.asarray(s1.coords[frame]) - center2
        N = len(coo1)

        if scale:
            scaling_factor = 0.0
            # compare distances from origin for each atom
            N_avg = 0
            for i in range(N):
                l1 = np.linalg.norm(coo1[i])
                l2 = np.linalg.norm(coo2[i])
                if (l1 > 0.01) and (l2 > 0.01):
                    scaling_factor += l2/l1
                    N_avg += 1
            scaling_factor /= N_avg
            coo1 *= scaling_factor
        else:
            scaling_factor = 1.0

        # covariance matrix
        A = coo1.T.dot(coo2)
        # singular value decomposition
        V, S, W_T = np.linalg.svd(A)
        # determine correct sign
        d = np.linalg.det((V.dot(W_T)).T) < 0.0
        if d:
            S[-1] = -S[-1]
            V[:, -1] = -V[:, -1]
        # compute optimal rotation matrix
        U = V.dot(W_T)

        # rotate coordinates
        coo1 = coo1.dot(U)

        # compute rmsd
        rmsd = 0.0
        for i in range(N):
            vec = coo1[i] - coo2[i]
            rmsd += np.sum(vec*vec)
        rmsd = np.sqrt(rmsd/N)

        # replace coordinates with aligned ones
        for i in range(self.nframes):
            coo_rot = (np.asarray(self.coords[i]) - center1)*scaling_factor
            coo_rot = coo_rot.dot(U)
            self.coords[i] = coo_rot + center2

        return rmsd

    def align_frames(self, frame=-1, ref_frame=0):
        """
        Align selected frame with a reference frame by
        adding/subtracting lattice vectors to the coordinates (PBC
        only).

        Arguments:
          frame      selected frame of trajectory (default: final step)
          ref_frame  reference frame to align with (default: first step)
        """

        if (not self.pbc) or (frame == ref_frame):
            return

        if (abs(frame) > self.nframes) or (abs(ref_frame) > self.nframes):
            raise ArgumentError("Specified frame number is out of range.")

        for i in range(self.natoms):
            vec = self.coords[ref_frame][i] - self.coords[frame][i]
            vec = np.round(self.cart2frac(vec, frame=frame))
            vec = self.frac2cart(vec, frame=frame)
            self.coords[frame][i] += vec

    def align_all_frames(self):
        """
        Convenience method to align all frames at once using the
        'align_frames' method.

        Arguments:
          ref_frame  reference frame to align with (default: first step)

        """

        if not self.pbc:
            return

        for i in range(1, self.nframes):
            self.align_frames(frame=i, ref_frame=i-1)

    def interpolate(self, s2, n, s3=None):
        """
        Interpolate between this structure and a second structure S2,
        possibly via a third structure S3 in N steps.

        See the documentation for geometry.interpolate() for more info.
        """

        return interpolate(self, s2, n, s3)

    def supercell(self, mult=[1, 1, 1], frame=None):
        """
        Construct a supercell of a periodic structure.

        Arguments:
          mult     list of 3 integers = multiples of original cell in the
                   direction of the three lattice vectors
          frame    frame to be used
        Returns:
          struc    instance of AtomicStructure
        """

        if not self.pbc:
            sys.stderr.write("Warning: Cannot construct supercell for"
                             " non-periodic structure.\n")
            return

        def _supercell(s, f, m):
            # new lattice vectors:
            avec_new = np.zeros([3, 3])
            avec_new[0] = m[0]*s.avec[f][0]
            avec_new[1] = m[1]*s.avec[f][1]
            avec_new[2] = m[2]*s.avec[f][2]
            # new atomic coordinates:
            natoms_new = m[0]*m[1]*m[2]*s.natoms
            coo_new = np.zeros([natoms_new, 3])
            types_new = []
            fixed_new = []
            iat2 = 0
            for iat in range(s.natoms):
                for i0 in range(0, m[0]):
                    for i1 in range(0, m[1]):
                        for i2 in range(0, m[2]):
                            coo = s.fraccoo(iat, frame=f)
                            coo_new[iat2, 0] = (coo[0] + float(i0))/float(m[0])
                            coo_new[iat2, 1] = (coo[1] + float(i1))/float(m[1])
                            coo_new[iat2, 2] = (coo[2] + float(i2))/float(m[2])
                            types_new.append(s.types[iat])
                            fixed_new.append(s.fixed[iat])
                            iat2 += 1
            return avec_new, coo_new, types_new, fixed_new

        if frame is not None:
            frame_list = [frame]
        else:
            frame_list = range(self.nframes)

        struc = None
        for f in frame_list:
            avec, coo, types, fixed = _supercell(self, f, mult)
            if struc is None:
                struc = AtomicStructure(coo, types, avec=avec,
                                        fractional=True, fixed=fixed)
            else:
                struc.add_frame(coo, avec=avec, fractional=True)

        return struc

    def wrap_to_cell(self):
        """
        Wrap coordinates of a periodic structure back to the simulation
        cell, i.e., fractional coordinates will fall into the range [0, 1[
        """

        if not self.pbc:
            return

        for iframe in range(self.nframes):
            for iatom in range(self.natoms):
                coo = self.fraccoo(iatom, frame=iframe)
                self.coords[iframe][iatom] = self.frac2cart(coo, frame=iframe)

    def distance(self, i, j, frame=-1, minimal=True):
        """
        Return the distance between atoms i and j.

        Arguments:
          i, j      atomic indices
          frame     MD frame index
          minimal   if True, search for minimal distance between periodic
                    images (PBC only)

        """

        if (not self.pbc) or (not minimal):
            return np.linalg.norm(self.coords[frame][j]-self.coords[frame][i])
        else:
            coo_i = self.fracoo(i, frame=frame, wrap=False)
            coo_j = self.fracoo(i, frame=frame, wrap=False)
            dist = coo_j - coo_i
            T = np.sign(dist)*np.floor(np.abs(dist))
            dist -= T
            return np.linalg.norm(self.frac2cart(dist, frame=frame))

    def add_vacuum(self, amount, direction=3, frame=-1,
                   perpendicular=True, duplicate_cell_boundary=True):
        """
        Add vacuum to construct a slab model or box an isolated structure.

        Arguments:
          amount (float): amount of vacuum in Angstrom
          direction (int): lattice direction (1, 2, or 3); only for
            periodic structure
          frame (int): frame to be used
          perpendicular (bool): measure vacuum in direction perpendicular
            to the other two lattice vectors (that is usually what you
            want for slabs); only for periodic structures
          duplicate_cell_boundary (bool): if true, duplicate atoms at the
            cell boundary in the direction of the vacuum, so that the
            topmost and bottom layers of the slab are identical.  Only
            for periodic structures.

        Returns:
          struc    an instance of AtomicStructure
        """

        if self.pbc:
            avec_new = self.avec[frame].copy()
            if perpendicular:
                # surface plane and surface normal
                plane = np.array(
                    [avec_new[i] for i in range(3) if i != (direction-1)])
                normal = np.cross(plane[0], plane[1])
                normal /= np.linalg.norm(normal)
                d = np.dot(avec_new[direction-1], normal)
            else:
                d = np.linalg.norm(avec_new[direction-1])
            stretch = (d + amount)/d
            avec_new[direction-1] *= stretch
            coords = self.coords[frame].copy()
            fixed = list(self.fixed.copy())
            types = list(self.types.copy())
            if duplicate_cell_boundary:
                frac_coords = np.array([self.fraccoo(i, frame=frame, wrap=True)
                                        for i in range(self.natoms)])
                # make sure that the Cartesian coordinates correspond to
                # the wrapped fractional coordinates
                coords = list(self.frac2cart(frac_coords, frame=frame))
                coo_vac_dir = frac_coords[:, direction-1]
                # consider all atoms within 0.2 Ang of the cell boundaries
                # as boundary atoms TODO: make an option
                tol = 0.2/np.linalg.norm(self.avec[frame][direction-1])
                boundary = (np.abs(coo_vac_dir) < tol
                            ) + (np.abs(coo_vac_dir-1.0) < tol)
                T = np.array([0.0, 0.0, 0.0])
                T[direction-1] = 1.0
                idx = [i for i in range(self.natoms) if boundary[i]]
                for i in idx:
                    coo = frac_coords[i]
                    if abs(coo[direction-1]) < tol:
                        coords.append(self.frac2cart(coo+T, frame=frame))
                    else:
                        coords.append(self.frac2cart(coo-T, frame=frame))
                    fixed.append(fixed[i])
                    types.append(types[i])
            struc = AtomicStructure(coords, types, avec=avec_new, fixed=fixed)
        else:
            a = np.max(self.coords[frame][0]) - np.min(self.coords[frame][0])
            b = np.max(self.coords[frame][0]) - np.min(self.coords[frame][0])
            c = np.max(self.coords[frame][0]) - np.min(self.coords[frame][0])
            a += amount
            b += amount
            c += amount
            avec_new = np.array([[a, 0.0, 0.0],
                                 [0.0, b, 0.0],
                                 [0.0, 0.0, c]])
            com = self.geometric_center()
            shift = 0.5*np.array([a, b, c]) - com
            s1 = self.translate(shift)
            struc = AtomicStructure(s1.coords[frame], s1.types,
                                    avec=avec_new, fixed=self.fixed)

        return struc

    def remove_atoms(self, idx):
        """
        Remove atoms from structure.

        Args:
          idx (list): list of atomic indices starting with 0
          frame (int): selected trajectory frame

        Returns:
          new AtomicStructure object

        """
        new_struc = self.copy()
        del_idx = np.array(idx, dtype=int)
        for i in range(new_struc.nframes):
            new_struc.coords[i] = np.delete(
                new_struc.coords[i], del_idx, axis=0)
        new_struc.types = np.delete(new_struc.types, del_idx)
        new_struc.typeID = np.delete(new_struc.typeID, del_idx)
        new_struc.fixed = np.delete(new_struc.fixed, del_idx, axis=0)
        return new_struc

    def get_neighbors(self, i, cutoff, return_self=True, frame=-1):
        """
        Get all neighbors of atom 'i' within cutoff 'cutoff'.

        Args:
          i (int): index of the central atom starting with zero
          cutoff (float): cutoff radius in Angstroms
          return_self (bool): whether to return the central atom
          frame (int): frame to be used (from trajectory)

        Returns:
          an atomic structure object

        """
        # Lazy import to avoid circular dependency
        from .torch_featurize.neighborlist import TorchNeighborList

        # Create neighbor list (accepts numpy arrays)
        nbl = TorchNeighborList(cutoff=cutoff, device='cpu')

        # Get neighbors with coordinates computed automatically
        positions = self.coords[frame]
        cell = self.avec[frame] if self.pbc else None

        result = nbl.get_neighbors_of_atom(
            i, positions, cell=cell, return_coordinates=True
        )

        # Extract data (convert back to numpy)
        neighbor_idx = result['indices'].cpu().numpy()

        # Handle empty neighbor list
        if len(neighbor_idx) == 0:
            if return_self:
                # Only return the central atom
                coords = np.array([self.coords[frame][i]])
                types = np.array([self.types[i]])
                return AtomicStructure(coords=coords, types=types)
            else:
                # No neighbors found and not returning self
                return None
        else:
            neighbor_coords = result['coordinates'].cpu().numpy()

            # Build result structure
            if return_self:
                coords = np.vstack([self.coords[frame][i], neighbor_coords])
                types = np.hstack([self.types[i], self.types[neighbor_idx]])
            else:
                coords = neighbor_coords
                types = self.types[neighbor_idx]

            return AtomicStructure(coords=coords, types=types)

    def frac2cart(self, fraccoo, avec=None, frame=-1):
        if (avec is None or len(avec) == 0):
            if not self.pbc:
                raise ArgumentError("No lattice vectors given, no "
                                    "periodic structure.")
            return np.dot(fraccoo, self.avec[frame])
        else:
            return np.dot(fraccoo, avec)

    def cart2frac(self, cartcoo, avec=None, frame=-1):
        if (avec is None or len(avec) == 0):
            if not self.pbc:
                raise ArgumentError("No lattice vectors given, no "
                                    "periodic structure.")
            return np.dot(cartcoo, self.bvec[frame])
        else:
            return np.dot(cartcoo, np.linalg.inv(avec))


def interpolate(s1, s2, n=1, s3=None):
    """
    Interpolate between the two structures s1 and s2.  Return n
    interpolated structures.  If only s1 and s2 are specified, the
    structures will be linearly interpolated.  If also s3 is given,
    all interpolated structures will be on a parabola through s3.

    Arguments:
      s1, s2    end points for interpolated path (AtomicStructure)
      s3        optional intermediate point (AtomicStructure)
      n         number of interpolated structures to be returned

    Returns:
      Instance of AtmicStructure with n+2 frames, where the first and
      the last frame correspond to s1 and s2.
    """

    if ((s2.natoms != s1.natoms) or (s2.ntypes != s1.ntypes) or
            (s1.pbc != s2.pbc)):
        raise ArgumentError(
            "Incompatible start and end structures for interpolation.")
    if (s3) and ((s3.natoms != s1.natoms) or (s3.ntypes != s1.ntypes) or
                 (s3.pbc != s1.pbc)):
        raise ArgumentError(
            "Incompatible intermediate structure for interpolation.")

    if s1.pbc:
        s = AtomicStructure(s1.coords[-1], s1.types, avec=s1.avec[-1],
                            fixed=s1.fixed)
    else:
        s = AtomicStructure(s1.coords[-1], s1.types, fixed=s1.fixed)
    if not s3:
        # linear interpolation between s1 and s2
        for x in np.linspace(0.0, 1.0, n+2):
            if (x > 0.0):
                coords = (1.0-x)*s1.coords[-1] + x*s2.coords[-1]
                if s.pbc:
                    avec = (1.0-x)*s1.avec[-1] + x*s2.avec[-1]
                    s.add_frame(coords, avec=avec)
                else:
                    s.add_frame(coords)
    else:
        c0 = s1.coords[-1]
        c1 = -3.0*s1.coords[-1] - s2.coords[-1] + 4.0*s3.coords[-1]
        c2 = 2.0*s1.coords[-1] + 2.0*s2.coords[-1] - 4.0*s3.coords[-1]
        if s.pbc:
            a0 = s1.avec[-1]
            a1 = -3.0*s1.avec[-1] - s2.avec[-1] + 4.0*s3.avec[-1]
            a2 = 2.0*s1.avec[-1] + 2.0*s2.avec[-1] - 4.0*s3.avec[-1]
        # second order interpolation, assuming s3 is exactly half-way
        # from s1 to s2
        for x in np.linspace(0.0, 1.0, n+2):
            if (x > 0.0):
                coords = c0 + c1*x + c2*x*x
                if s.pbc:
                    avec = a0 + a1*x + a2*x*x
                    s.add_frame(coords, avec=avec)
                else:
                    s.add_frame(coords)

    return s
