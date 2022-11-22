"""
Classes for interacting with aenet training set files.

Currently, only training set fuiles in ASCII format are supported.

"""

from typing import List, Literal
import os
import subprocess

import numpy as np
import tables as tb
import scipy.stats

from . import config
from .serialize import Serializable
from .io.structure import read_safely
from .geometry import AtomicStructure

__author__ = "The aenet developers"
__email__ = "aenet@atomistic.net"
__date__ = "2020-11-21"
__version__ = "0.1"


class FeaturizedAtomicStructure(Serializable):
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

        # the path string can contain additional information that is 
        # stripped here if needed
        if (not os.path.exists(path) 
            and os.path.exists(path.split(".xsf")[0] + ".xsf")):
            self.path = path.split(".xsf")[0] + ".xsf"

        avec = None
        if os.path.exists(self.path):
            inp_structure = read_safely(self.path, frmt='xsf')
            if inp_structure.pbc:
                avec = inp_structure.avec[-1]

        types = [at['type'] for at in self.atoms]
        coords = [at['coords'] for at in self.atoms]
        forces = [at['forces'] for at in self.atoms]
        self.structure = AtomicStructure(coords, types, avec=avec, 
                                         energy=self.energy,
                                         forces=forces)

    def __str__(self):
        out = "FeaturizedAtomicStructure Info:\n"
        out += "  Path           : {}\n".format(self.path)
        out += "  Atom types     : "
        out += " ".join(sorted(self.atom_types)) + "\n"
        out += str(self.structure)
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

    @property
    def composition(self):
        comp = {}
        types = [a['type'] for a in self.atoms]
        for i, t in enumerate(self.atom_types):
            n = types.count(i)
            if n > 0:
                comp[t] = n
        return comp

    def moment_fingerprint(self, sel_atom_types: List[str] = None, moment=2):
        """
        Calculate a fingerprint for a collection of atoms using a moment
        expansion for each atom type.

        Note: this implementation assumes that the atomic descriptors for
        each species have the same length.

        """
        if sel_atom_types is None:
            sel_atom_types = self.atom_types
        dimension = self.max_descriptor_length
        structure_fingerprint = []
        for i, s in enumerate(self.atom_types):
            if s not in sel_atom_types:
                continue
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

    @staticmethod
    def _compute_moments(lst, moment: int=1, axis=0):
        """
        Helper function to compute moments of a list up to a degree.
        Default is the mean.
        To reduce down a 2D array to a single moment, run the function twice.
        """
        moments_list = np.array([np.mean(lst, axis=0)])
        moments_list = np.append(moments_list, scipy.stats.moment(lst, moment=range(2, moment+1), axis=0)).flatten().tolist()
        return moments_list

    def _global_moment_fingerprint(self, moment: int=1, average: str='none', exclude_zero: bool=False):
        """
        Calculate the global fingerprint from local atomic fingerprints using a moment expansion.

        Note: this implementation assumes that the atomic descriptors for each species have the same length.

        :param str average: Averaging parameter that only accepts some possible values. Possible values:
            - 'none': The global fingerprint involves the moments of all atomic fingerprints.
                       F_global = moments(F_A U F_B U ...),
            - 'inner': The global fingerprint involves the moments of averaged atomic fingerprints for each species.
                       F_global = moments(mean(F_A) U mean(F_B) U ...),
            - 'outer': The global fingerprint involves the mean of all moments of atomic fingerprints. 
                        F_global = mean(moments(F_A) U moments(F_B) U ...),

        where 
            F_global is the global fingerprint,
            F_s is the union of atomic fingerprints for species s (F_s = F_s(1) U F_s(2) U ...), 
            F_s(i) is atomic fingerprint for species s at site i.

        :param bool exclude_zero: Whether to exclude or include species that are not part of the structure.
        """
        if average not in ('none', 'inner', 'outer'):
            raise ValueError("Not supported averaging method. Choose a valid averaging method.")
        if not isinstance(moment, int) or moment < 1:
            raise ValueError("Not supported moment. Moment should be a positive integer (i.e. 1, 2, 3, ...).")
        
        dimension = self.max_descriptor_length
        atoms_info = self.atom_types

        structure_fingerprint = []
        for i, s in enumerate(self.atom_types):
            atomic_fingerprint = [a['fingerprint'] for a in atoms_info if a['type'] == s]

            if not atomic_fingerprint:
                if exclude_zero:
                    continue
                else:
                    atomic_fingerprint = [[0.0 for _ in range(dimension)]]

            if average=='none':
                structure_fingerprint.extend(atomic_fingerprint)
            elif average=='inner':
                mean = _compute_moments(atomic_fingerprint, moment=1, axis=0)
                structure_fingerprint.append(mean)
            elif average=='outer':
                moments = _compute_moments(atomic_fingerprint, moment=moment, axis=0)
                structure_fingerprint.append(moments)

        if average in ('none', 'inner'):
            global_fingerprint = _compute_moments(structure_fingerprint, moment=moment, axis=0)
        elif average=='outer':
            global_fingerprint = _compute_moments(structure_fingerprint, moment=1, axis=0)

        return np.array(global_fingerprint)


class TrnSet(object):
    """
    Class for parsing aenet training set files.

    *Attention*: atom type indices here internally start with zero
                 (whereas they start with 1 in Fortran)

    """

    def __init__(self, name: str, normalized: bool, scale: float, shift:
                 float, atom_types: List[str], 
                 atomic_energy: List[float], 
                 num_atoms_tot: int, num_structures: int,
                 E_min: float, E_max: float, E_av: float,
                 filename: os.PathLike = None, 
                 fileformat: Literal["ascii", "hdf5"] = 'ascii', 
                 origin: os.PathLike = None, **kwargs):
        for arg in kwargs:
            TypeError("Unexpected keyword argument '{}'.".format(arg))
        self.name = name
        self.normalized = normalized
        self.scale = scale
        self.shift = shift
        self.atom_types = atom_types
        self.atomic_energy = atomic_energy
        self.num_atoms_tot = num_atoms_tot
        self.num_structures = num_structures
        self.E_min, self.E_max, self.E_av = (E_min, E_max, E_av)
        self.origin = origin
        self.opened = False
        if filename is not None:
            self.filename = filename
            self.format = fileformat
            self.open()          
            if self.origin is None:
                dirname = os.path.dirname(filename)
                self.origin = dirname if len(dirname) > 0 else None

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
        if self.filename is not None:
            out += "  File (format)  : {} ({})\n".format(
                self.filename, self.format)
        return out

    def __iter__(self):
        return self.iter_structures(read_coords=True, read_forces=True)

    @classmethod
    def from_file(cls, filename: os.PathLike, 
                  file_format: Literal[
                    'guess', 'ascii', 'hdf5', 'binary'] = 'guess', 
                  **kwargs):
        if not os.path.exists(filename):
            raise FileNotFoundError("File not found: {}".format(filename))
        if file_format == 'guess':
            file_format = None
            try:
                f = tb.open_file(filename)
                f.close()
                file_format = 'hdf5'
            except tb.HDF5ExtError:
                try:
                    # we accept all UTF-8 characters, not only ASCII
                    with open(filename, 'r', encoding='utf-8') as f:
                        for line in f:
                            pass
                    file_format = 'ascii'
                except UnicodeDecodeError:
                    file_format = 'binary'
        if file_format == 'hdf5':
            return cls.from_hdf5_file(filename, **kwargs)
        elif file_format == 'ascii':
            return cls.from_ascii_file(filename, **kwargs)
        elif file_format == 'binary':
            return cls.from_fortran_binary_file(filename, **kwargs)
        else:
            raise ValueError("Unexpected file format '{}'".format(file_format))

    @classmethod
    def from_ascii_file(cls, ascii_file: os.PathLike, **kwargs):
        """
        Load training set from aenet ASCII file.

        Args:
          ascii_file: path to an aenet training set file in ASCII format

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
                   E_max, E_av, filename=ascii_file, fileformat='ascii',
                   **kwargs)

    @classmethod
    def from_fortran_binary_file(cls, 
                                 binary_file: os.PathLike, 
                                 ascii_file: os.PathLike = None,
                                 **kwargs):
        """
        First convert training set file in Fortran binary format to ASCII
        format, then open it.  This requires the tool 'trnset2ASCII.x'.
        """
        aenet_paths = config.read('aenet')
        if not os.path.exists(aenet_paths['trnset2ascii_x_path']):
            raise FileNotFoundError(
                "Cannot find `trnset2ASCII.x`. Configure with `aenet config`.")
        if not os.path.exists(binary_file):
            raise FileNotFoundError("File not found: '{}'".format(binary_file))
        if ascii_file is None:
            ascii_file = binary_file + ".ascii"
        output = subprocess.run(
            [aenet_paths['trnset2ascii_x_path'], '--raw', 
            binary_file, ascii_file], stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE)
        if len(output.stderr.strip()) > 0:
            raise IOError("Conversion of binary to text file failed.")
        return cls.from_ascii_file(ascii_file, **kwargs)

    @classmethod
    def from_hdf5_file(cls, hdf5_file: os.PathLike, **kwargs):
        h5file = tb.open_file(hdf5_file, mode='r')
        metadata = h5file.root.metadata[0]
        name = metadata['name'].decode('utf-8')
        normalized = metadata['normalized']
        scale = metadata['scale']
        shift = metadata['shift']
        atom_types = [t.decode('utf-8') for t in metadata['atom_types']]
        atomic_energy = metadata['atomic_energy']
        num_atoms_tot = metadata['num_atoms_tot']
        num_structures = metadata['num_structures']
        E_min = metadata['E_min']
        E_max = metadata['E_max']
        E_av = metadata['E_av']
        h5file.close()
        return cls(name, normalized, scale, shift, atom_types,
                   atomic_energy, num_atoms_tot, num_structures, E_min,
                   E_max, E_av, filename=hdf5_file, fileformat='hdf5',
                   **kwargs)

    @property
    def num_types(self):
        return len(self.atom_types)

    def to_hdf5(self, filename: os.PathLike, complevel: int = 1):
        """
        Save data set to HDF5 file.

        """
        h5file = tb.open_file(filename, mode='w', name='Aenet reference data')
        structures = h5file.create_group(
            h5file.root, "structures", "Atomic structures")

        metadata = h5file.create_table(
            h5file.root, "metadata", {
                'name': tb.StringCol(itemsize=1024),
                'normalized': tb.BoolCol(),
                'scale': tb.Float64Col(),
                'shift': tb.Float64Col(),
                'atom_types': tb.StringCol(itemsize=64, shape=(self.num_types,)),
                'atomic_energy': tb.Float64Col(shape=(self.num_types,)),
                'num_atoms_tot': tb.UInt64Col(),
                'num_structures': tb.UInt64Col(),
                'E_min': tb.Float64Col(),
                'E_max': tb.Float64Col(),
                'E_av': tb.Float64Col()},
            "General information about the data set")

        info_table_dict = {
            "path": tb.StringCol(itemsize=1024),
            "first_atom":  tb.UInt64Col(),
            "num_atoms": tb.UInt32Col(),
            "energy": tb.Float64Col()
        }
        atom_table_dict = {
            "structure": tb.UInt64Col(),
            "type":  tb.StringCol(itemsize=64),
            "coords": tb.Float64Col(shape=(3,)),
            "forces": tb.Float64Col(shape=(3,))
        }
        info = h5file.create_table(
            structures, "info", info_table_dict, 
            "Atomic structure information", 
            tb.Filters(complevel, shuffle=False))
        atoms = h5file.create_table(
            structures, "atoms", atom_table_dict, 
            "Atomic data", 
            tb.Filters(complevel, shuffle=False))
        features = h5file.create_vlarray(
            structures, "features", tb.Float64Atom(),
            "Atomic environment features", 
            tb.Filters(complevel, shuffle=False))

        metadata.row['name'] = self.name
        metadata.row['normalized'] = self.normalized
        metadata.row['scale'] = self.scale
        metadata.row['shift'] = self.shift
        metadata.row['atom_types'] = self.atom_types
        metadata.row['atomic_energy'] = self.atomic_energy
        metadata.row['num_atoms_tot'] = self.num_atoms_tot
        metadata.row['num_structures'] = self.num_structures
        metadata.row['E_min'] = self.E_min
        metadata.row['E_max'] = self.E_max
        metadata.row['E_av'] = self.E_av
        metadata.row.append()

        self.rewind()
        iatom = 0
        for i in range(self.num_structures):
            s = self.read_next_structure(read_coords=True, read_forces=True)
            info.row['path'] = s.path
            info.row['first_atom'] = iatom
            info.row['num_atoms'] = s.num_atoms
            info.row['energy'] = s.energy
            info.row.append()
            for j in range(s.num_atoms):
                atoms.row['structure'] = i
                atoms.row['type'] = s.atoms[j]['type']
                atoms.row['coords'] = s.atoms[j]['coords']
                atoms.row['forces'] = s.atoms[j]['forces']
                atoms.row.append()
                features.append(s.atoms[j]['fingerprint'])
            iatom += s.num_atoms
        h5file.close()

    def open(self):
        """
        Open training set file for reading.

        """
        if self.filename is None:
            raise ValueError("Cannot open training set file. No file give.")

        if self.opened:
            self.rewind()
        elif self.format == 'ascii':
            self._fp = open(self.filename)
            self.opened = True
            self._istruc = 0
            self._ascii_skip_header()
        elif self.format == 'hdf5':
            self._fp = tb.open_file(self.filename)
            self.opened = True
            self._istruc = 0

    def close(self):
        if self.opened:
            self._fp.close()
            self.opened = False

    def rewind(self):
        self.close()
        self.open()

    def _ascii_skip_header(self):
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

    def iter_structures(self, read_coords=False, read_forces=False):
        self.rewind()
        for i in range(self.num_structures):
            yield self.read_next_structure(read_coords, read_forces)

    def read_structure(self, idx: int, read_coords=False, read_forces=False):
        if self.format == 'ascii':
            if self._istruc > idx:
                self.rewind()
            while self._istruc < idx:
                _ = self._read_next_structure_ascii(False, False)
            return self._read_next_structure_ascii(read_coords, read_forces)
        elif self.format == 'hdf5':
            return self._read_structure_hdf5(idx, read_coords, read_forces)
        else:
            raise ValueError("Unknown format: {}".format(self.format))

    def read_next_structure(self, read_coords=False, read_forces=False):
        if self.format == 'ascii':
            return self._read_next_structure_ascii(read_coords, read_forces)
        elif self.format == 'hdf5':
            return self._read_next_structure_hdf5(read_coords, read_forces)
        else:
            raise ValueError("Unknown format: {}".format(self.format))

    def _read_structure_hdf5(self, idx, read_coords, read_forces):
        row = self._fp.root.structures.info[idx]
        path = row['path'].decode('utf-8')
        if self.origin is not None:
            path = os.path.abspath(os.path.join(self.origin, path))
        energy = row['energy']
        first_atom = row['first_atom']
        num_atoms = row['num_atoms']
        atoms = []
        for i in range(first_atom, first_atom + num_atoms):
            row = self._fp.root.structures.atoms[i]
            fingerprint = self._fp.root.structures.features[i]
            atoms.append({"type": row['type'].decode('utf-8'),
                         "fingerprint": fingerprint,
                         "coords": row['coords'] if read_coords else None,
                         "forces": row['forces'] if read_forces else None})
        return FeaturizedAtomicStructure(
            path, energy, self.atom_types, atoms)

    def _read_next_structure_hdf5(self, read_coords, read_forces):
        s = self._read_structure_hdf5(self._istruc, read_coords, read_forces)
        self._istruc += 1
        return s

    def _read_next_structure_ascii(self, read_coords, read_forces):
        """
        Read next atomic structure from file.

        """
        if not self.opened:
            self.open()

        if self._istruc >= self.num_structures:
            self.close()
            return None

        path = self._fp.readline().strip()
        if self.origin is not None:
            path = os.path.abspath(os.path.join(self.origin, path))
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
            atoms.append({"type": self.atom_types[atom_type],
                          "fingerprint": fingerprint,
                          "coords": coords,
                          "forces": forces})
        self._istruc += 1
        return FeaturizedAtomicStructure(
            path, energy, self.atom_types, atoms)
