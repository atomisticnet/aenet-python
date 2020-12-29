"""
Classes for interacting with aenet training set files.

Currently, only training set fuiles in ASCII format are supported.

"""

from typing import List
import numpy as np
import scipy.stats

from .serialize import Serializable

__author__ = "The aenet developers"
__email__ = "aenet@atomistic.net"
__date__ = "2020-11-21"
__version__ = "0.1"


class AtomicStructure(Serializable):
    """
    Class to hold all information of an atomic structure.

    Attributes:

    path (str): Path to the original structure file
    energy (float): Total energy of the structure
    atom_types (List[str]): List of atom types (chemical symbols)
    atoms (List[dict]): Atomic information as dictionary with the keys
         {"type": atom_type,
          "fingerprint": fingerprint,
          "coords": coords,
          "forces": forces}

    Properties:

    max_descriptor_length (int): Dimension of longest fingerprint among
      all atoms of the atomic structure

    """

    def __init__(self, path: str, energy: float, atom_types: List[str],
                 atoms: List[dict]):
        self.path = path
        self.energy = energy
        self.atom_types = atom_types
        self.atoms = atoms

    def __str__(self):
        out = "AtomicStructure Info:\n"
        out += "  Path           : {}\n".format(self.path)
        out += "  Energy         : {:.6e}\n".format(self.energy)
        out += "  Atom types     : "
        out += " ".join(sorted(self.atom_types)) + "\n"
        out += "  Number of atoms: {}".format(len(self.atoms))
        return out

    @property
    def num_atoms(self):
        return len(self.atoms)

    @property
    def max_descriptor_length(self):
        """
        Dimension of longest fingerprint among all atoms of the atomic
        structure

        """
        max_length = 0
        for a in self.atoms:
            max_length = max(max_length, len(a['fingerprint']))
        return max_length

    def moment_fingerprint(self, all_atom_types: List[str], moment=2):
        """
        Calculate a fingerprint for a collection of atoms using a moment
        expansion for each atom type.

        Note: this implementation assumes that the atomic descriptors for
        each species have the same length.

        """
        dimension = self.max_descriptor_length
        structure_fingerprint = []
        for i, s in enumerate(all_atom_types):
            atomic_fingerprints = [
                a['fingerprint'] for a in self.atoms if a['type'] == i]
            if len(atomic_fingerprints) == 0:
                structure_fingerprint.extend(
                    [0.0 for j in range(moment*dimension)])
            else:
                mean = np.mean(atomic_fingerprints, axis=0)
                structure_fingerprint.extend(mean)
                for m in range(2, moment+1):
                    structure_fingerprint.extend(
                        scipy.stats.moment(atomic_fingerprints,
                                           moment=moment, axis=0))
        return structure_fingerprint


class TrnSet(object):
    """
    Class for parsing aenet training set files.

    *Attention*: atom type indices here internally start with zero
                 (whereas they start with 1 in Fortran)

    """

    def __init__(self, name: str, normalized: bool, scale: float, shift:
                 float, atom_types: List[str], atomic_energy:
                 List[float], num_atoms_tot: int, num_structures: int,
                 E_min: float, E_max: float, E_av: float,
                 ascii_file=None):
        self.name = name
        self.normalized = normalized
        self.scale = scale
        self.shift = shift
        self.atom_types = atom_types
        self.atomic_energy = atomic_energy
        self.num_atoms_tot = num_atoms_tot
        self.num_structures = num_structures
        self.E_min, self.E_max, self.E_av = (E_min, E_max, E_av)
        self.ascii_file = ascii_file
        self.opened = False

    def __del__(self):
        if self.opened:
            self.close()

    def __str__(self):
        out = "\nTraining set info:\n"
        out += "  Name           : {}\n".format(self.name)
        if self.normalized:
            out += "  Scale, shift   : {}, {}\n".format(self.scale, self.shift)
        out += "  Atom types     : " + " ".join(self.atom_types) + "\n"
        out += "  Atomic energies: " + " ".join(
            ["{:.3f}".format(E) for E in self.atomic_energy]) + "\n"
        out += "  #atom, #struc. : {} {}\n".format(
            self.num_atoms_tot, self.num_structures)
        out += "  E_min, max, av : {:.3f} {:.3f} {:.3f}\n".format(
            self.E_min, self.E_max, self.E_av)
        if self.ascii_file:
            out += "  ASCII file     : {}\n".format(self.ascii_file)
        return out

    @classmethod
    def from_ascii_file(cls, ascii_file):
        """
        Load training set from aenet ASCII file.

        """
        with open(ascii_file) as fp:
            name = fp.readline().strip()
            normalized = fp.readline().strip()
            if normalized == "T":
                normalized = True
            else:
                normalized = False
            scale = float(fp.readline().strip())
            shift = float(fp.readline().strip())
            num_types = int(fp.readline().strip())
            atom_types = []
            for i in range(num_types):
                atom_types.append(fp.readline().strip())
            atomic_energy = []
            while len(atomic_energy) < num_types:
                atomic_energy.extend(
                    [float(E) for E in fp.readline().strip().split()])
            num_atoms_tot = int(fp.readline().strip())
            num_structures = int(fp.readline().strip())
            E_min, E_max, E_av = [
                float(E) for E in fp.readline().strip().split()]
        return cls(name, normalized, scale, shift, atom_types,
                   atomic_energy, num_atoms_tot, num_structures, E_min,
                   E_max, E_av, ascii_file=ascii_file)

    @property
    def num_types(self):
        return len(self.atom_types)

    def open(self):
        """
        Open training set file for reading.

        """
        if self.ascii_file is None:
            raise ValueError("Cannot open training set file. No file give.")

        if self.opened:
            self.rewind()
        else:
            self._fp = open(self.ascii_file)
            self.opened = True
            self._istruc = 0
            self.skip_header()

    def close(self):
        if self.opened:
            self._fp.close()
            self.opened = False

    def rewind(self):
        self.close()
        self.open()

    def skip_header(self):
        """
        Skip over training set file header until first atomic structure.

        """
        if not self.opened:
            return
        self._fp.readline()
        self._fp.readline()
        self._fp.readline()
        self._fp.readline()
        self._fp.readline()
        for i in range(self.num_types):
            self._fp.readline()
        atomic_energy = []
        while len(atomic_energy) < self.num_types:
            atomic_energy.extend(self._fp.readline().strip().split())
        self._fp.readline()
        self._fp.readline()
        self._fp.readline()

    def read_next_structure(self, read_coords=False, read_forces=False):
        """
        Read next atomic structure from file.

        """
        if not self.opened:
            self.open()

        if self._istruc >= self.num_structures:
            self.close()
            return None

        path = self._fp.readline().strip()
        num_atoms, num_types = [
            int(N) for N in self._fp.readline().strip().split()]
        energy = float(self._fp.readline().strip())
        atoms = []
        coords = forces = None
        for i in range(num_atoms):
            # lowest atom type index is zero (unlike in Fortran)
            atom_type = int(self._fp.readline().strip()) - 1
            # skip coordinates, forces, and descriptor dimension
            if read_coords:
                coords = [float(val) for val in self._fp.readline().split()]
            else:
                self._fp.readline()
            if read_forces:
                forces = [float(val) for val in self._fp.readline().split()]
            else:
                self._fp.readline()
            self._fp.readline()
            # read descriptor
            fingerprint = [
                float(v) for v in self._fp.readline().strip().split()]
            atoms.append({"type": atom_type,
                          "fingerprint": fingerprint,
                          "coords": coords,
                          "forces": forces})
        atom_types = sorted(set([a["type"] for a in atoms]))
        atom_types = [self.atom_types[i] for i in atom_types]
        structure = AtomicStructure(path, energy, atom_types, atoms)
        return structure
