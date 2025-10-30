"""
Classes for interacting with aenet training set files.

Currently, only training set fuiles in ASCII format are supported.

"""

from typing import List
import os
import subprocess
import warnings

import numpy as np
import tables as tb

from . import config
from .serialize import Serializable
from .io.structure import read_safely
from .geometry import AtomicStructure
from .util import compute_moments

__author__ = "The aenet developers"
__email__ = "aenet@atomistic.net"
__date__ = "2020-11-21"
__version__ = "0.1"


class FeaturizedAtomicStructure(Serializable):
    """
    Class to hold all information of an atomic structure.

    Attributes
    ----------
    path : str
        Path to the original structure file.
    energy : float
        Total energy of the structure.
    atom_types : list[str]
        List of atom types (chemical symbols).
    atoms : list[dict]
        Atomic information per atom with keys:
        {'type': atom_type, 'fingerprint': fingerprint,
         'coords': coords, 'forces': forces}.
    neighbor_info : dict or None
        Optional neighbor information for force training. If present, contains:
        - 'neighbor_counts': (n_atoms,) array of neighbor counts
        - 'neighbor_lists': list of (nnb,) arrays with neighbor indices
        - 'neighbor_vectors': list of (nnb, 3) arrays with displacement vectors
    cell : numpy.ndarray or None
        Unit cell lattice vectors as (3, 3) array where rows are
        lattice vectors.
    pbc : bool
        True for 3D-periodic structures; False for isolated structures.

    Notes
    -----
    Properties like has_neighbor_info, has_cell, is_periodic, num_atoms,
    max_descriptor_length, composition, atom_weights, and atom_features
    are documented on their respective properties.
    """

    def __init__(self, path: str, energy: float, atom_types: List[str],
                 atoms: List[dict], neighbor_info: dict = None,
                 cell: np.ndarray = None, pbc: bool = None):
        self.path = path
        self.energy = energy
        self.atom_types = atom_types
        self.atoms = atoms
        self.neighbor_info = neighbor_info
        self.cell = cell
        self.pbc = pbc if pbc is not None else False

        # the path string can contain additional information that is
        # stripped here if needed
        if (not os.path.exists(path)
                and os.path.exists(path.split(".xsf")[0] + ".xsf")):
            self.path = path.split(".xsf")[0] + ".xsf"

    def __str__(self):
        out = "FeaturizedAtomicStructure Info:\n"
        out += "  Path           : {}\n".format(self.path)
        out += "  Atom types     : "
        out += " ".join(sorted(self.atom_types)) + "\n"
        out += "  Neighbor info  : {}\n".format(
            "Available" if self.has_neighbor_info else "Not available"
        )
        out += str(self.structure)
        return out

    @property
    def avec(self):
        """
        Get unit cell lattice vectors.

        Returns cell from HDF5 if available, otherwise reads from XSF file
        as fallback for legacy binary format support.
        """
        # Prioritize cell from HDF5
        if self.cell is not None:
            return self.cell

        # Fallback: read from XSF file (for legacy binary format)
        avec = None
        if os.path.exists(self.path):
            inp_structure = read_safely(self.path, frmt='xsf')
            if inp_structure.pbc:
                avec = inp_structure.avec[-1]
        return avec

    @property
    def types(self):
        return [at['type'] for at in self.atoms]

    @property
    def coords(self):
        return np.array([at['coords'] for at in self.atoms])

    @property
    def forces(self):
        return np.array([at['forces'] for at in self.atoms])

    @property
    def structure(self):
        structure = AtomicStructure(
            self.coords,
            self.types,
            avec=self.avec,
            energy=self.energy,
            forces=self.forces
            )
        return structure

    @property
    def has_neighbor_info(self):
        """
        Returns True if neighbor information is available for force training.
        """
        return self.neighbor_info is not None

    @property
    def has_cell(self):
        """
        Returns True if unit cell information is available.
        """
        return self.cell is not None

    @property
    def is_periodic(self):
        """
        Returns True if structure is periodic (3D-periodic).
        """
        return self.pbc and self.cell is not None

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
        for t in self.atom_types:
            n = types.count(t)
            if n > 0:
                comp[t] = n
        return comp

    @property
    def atom_weights(self):
        s = len(self.atom_types) // 2
        weights = (list(range(-s, 0))
                   + ([0] if len(self.atom_types) % 2 != 0 else [])
                   + list(range(1, s + 1)))
        return {a: w for a, w in zip(self.atom_types, weights)}

    @property
    def atom_features(self):
        return np.array([self.atoms[i]['fingerprint']
                         for i in range(self.num_atoms)])

    def atom_features_for_type(self, atom_type: str):
        """
        Return only the features for atoms of a selected type.

        Args:
            atom_type (str): Chemical symbol.
        """
        idx = (np.array([self.atoms[i]['type']
                         for i in range(self.num_atoms)]) == atom_type)
        return self.atom_features[idx]

    def global_moment_fingerprint(self, outer_moment: int = 1,
                                  inner_moment: int = 1,
                                  weighted: bool = False,
                                  weights: dict = None,
                                  append_weighted: bool = False,
                                  stack_type_features: bool = False,
                                  exclude_zero_atoms: bool = False,
                                  atom_types: List[str] = None):
        """
        Calculate a global fingerprint from local atomic fingerprints using a
        moment expansion.

        This implementation assumes that atomic descriptors for each species
        have the same length.

        Parameters
        ----------
        outer_moment : int, default=1
            Up to which outer moment to compute. Must be >= 1.
            Not used when ``stack_type_features`` is True.
        inner_moment : int, default=1
            Up to which inner moment to compute. Must be >= 0
            (0 = no moment, 1 = mean).
        weighted : bool, default=False
            Whether to apply species weights to the type fingerprints.
        weights : dict[str, float] or None, default=None
            Mapping of atom symbol to weight. Defaults to ``self.atom_weights``
            when ``weighted`` is True.
        append_weighted : bool, default=False
            If True and ``weighted`` is True, append the weighted features to
            the unweighted features; otherwise only return weighted features.
        stack_type_features : bool, default=False
            If True, concatenate per-type feature vectors instead of performing
            the outer moment expansion.
        exclude_zero_atoms : bool, default=False
            If True, skip species with zero count in the structure.
        atom_types : list[str] or None, default=None
            Subset of chemical symbols to consider. Defaults to all
            ``self.atom_types``.

        Returns
        -------
        numpy.ndarray
            Global fingerprint vector.

        Notes
        -----
        The global fingerprint can be conceived as::

            F_global = outer_moments(w_A * inner_moments(F_A)
                                     U w_B * inner_moments(F_B) U ...)

        where

            - ``F_global`` is the global fingerprint
            - ``F_s`` is the union of atomic fingerprints for species ``s``
              (``F_s = F_s(1) U F_s(2) U ...``)
            - ``F_s(i)`` is the atomic fingerprint for
              species ``s`` at site ``i``
            - ``w_s`` is the weight for species ``s``

        The dimension is
        ``len(type_fingerprint) * inner_moment * outer_moment``
        (or ``len(type_fingerprint) * outer_moment`` if ``inner_moment`` is 0).
        """
        if not isinstance(outer_moment, int) or outer_moment < 1:
            raise ValueError(
                "Not supported outer moment. Outer moment "
                "should be a positive integer (i.e. 1, 2, 3, ...).")
        if not isinstance(inner_moment, int) or inner_moment < 0:
            raise ValueError(
                "Not supported inner moment. Inner moment should be a "
                "non-negative integer (i.e. 0, 1, 2, 3, ...).")
        if (weighted and weights is not None
            and (len(weights) != len(self.atom_types)
                 or not all(w in weights for w in self.atom_types))):
            raise ValueError("The weights dictionary should contain "
                             "only the included elements.")
        if stack_type_features and weighted:
            warnings.warn("Type weighting is usually not useful when "
                          "the type features are stacked.")

        if weighted and weights is None:
            weights = self.atom_weights

        if atom_types is None:
            atom_types = self.atom_types

        dimension = self.max_descriptor_length
        structure_fingerprint = []
        for s in atom_types:
            atom_features = self.atom_features_for_type(s)

            if len(atom_features) == 0:
                # no atoms of type s found
                if exclude_zero_atoms:
                    continue
                else:
                    atom_features = np.zeros(dimension)

            type_fingerprint = compute_moments(
                atom_features, inner_moment).flatten()

            if weighted:
                type_fingerprint_w = weights[s]*type_fingerprint
                if append_weighted:
                    # append the weighted features to the unweighted
                    type_fingerprint = np.append(
                        type_fingerprint,
                        type_fingerprint_w)
                else:
                    # only keep the weighted features
                    type_fingerprint = type_fingerprint_w

            structure_fingerprint.append(type_fingerprint)

        # combine features from the individual atom types either
        # by concatenation or by performing another (outer) moment
        # expansion
        if stack_type_features:
            structure_fingerprint = np.array(structure_fingerprint).flatten()
        else:
            structure_fingerprint = compute_moments(
                structure_fingerprint, outer_moment).flatten()

        return structure_fingerprint


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
                 fileformat: str = 'ascii',
                 origin: os.PathLike = None, **kwargs):
        for arg in kwargs:
            TypeError("Unexpected keyword argument '{}'.".format(arg))
        if fileformat not in ["ascii", "hdf5"]:
            raise ValueError('Invalid file format {}'.format(fileformat))
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

    def __getitem__(self, key):
        """
        Enable subscript access to structures.

        Supports integer indexing, negative indexing, and slicing.

        Examples:
            >>> struc = trnset[0]  # First structure
            >>> last = trnset[-1]  # Last structure
            >>> subset = trnset[0:10]  # First 10 structures
        """
        if isinstance(key, int):
            # Handle negative indices
            if key < 0:
                key = self.num_structures + key
            if key < 0 or key >= self.num_structures:
                raise IndexError("Structure index out of range")
            return self.read_structure(key, read_coords=True, read_forces=True)
        elif isinstance(key, slice):
            # Handle slicing
            indices = range(*key.indices(self.num_structures))
            return [self[i] for i in indices]
        else:
            raise TypeError("Indices must be integers or slices, not "
                            + "{}".format(type(key).__name__))

    def __enter__(self):
        if not self.opened:
            self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        # returning False propagates exceptions
        return False

    @classmethod
    def from_file(cls, filename: os.PathLike,
                  file_format: str = 'guess',
                  **kwargs):
        if not os.path.exists(filename):
            raise FileNotFoundError("File not found: {}".format(filename))
        if file_format not in ['guess', 'ascii', 'hdf5', 'binary']:
            raise ValueError('Invalid file format {}'.format(file_format))
        elif file_format == 'guess':
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
             binary_file, ascii_file],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
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

    def has_neighbor_info(self) -> bool:
        """
        Check if the training set file contains neighbor information.

        Returns:
            True if neighbor information is available (only for HDF5 format),
            False otherwise.
        """
        if self.format != 'hdf5':
            return False

        if not self.opened:
            self.open()
            was_closed = True
        else:
            was_closed = False

        try:
            has_info = hasattr(self._fp.root.structures, 'neighbor_info')
        finally:
            if was_closed:
                self.close()

        return has_info

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
                'atom_types': tb.StringCol(
                    itemsize=64, shape=(self.num_types,)),
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

        # Read neighbor information if available for this structure
        neighbor_info = None
        if hasattr(self._fp.root.structures, 'neighbor_info'):
            neighbor_group = self._fp.root.structures.neighbor_info
            # Check if this structure has a neighbor info group
            group_name = f'structure_{idx:06d}'
            if group_name in neighbor_group:
                neighbor_info = self._read_neighbor_info_hdf5(idx, num_atoms)

        # Read cell information if available for this structure
        cell = None
        pbc = False
        if hasattr(self._fp.root.structures, 'cells'):
            cells_group = self._fp.root.structures.cells
            group_name = f'structure_{idx:06d}'
            if group_name in cells_group:
                cell, pbc = self._read_cell_info_hdf5(idx)

        return FeaturizedAtomicStructure(
            path, energy, self.atom_types, atoms,
            neighbor_info=neighbor_info, cell=cell, pbc=pbc)

    def _read_neighbor_info_hdf5(self, structure_idx: int,
                                 num_atoms: int) -> dict:
        """
        Read neighbor information for a specific structure from HDF5.

        Uses per-structure group format for O(1) lookup.

        Args:
            structure_idx: Index of the structure
            num_atoms: Number of atoms in structure

        Returns:
            Dictionary with neighbor information:
                - 'neighbor_counts': (n_atoms,) array of neighbor counts
                - 'neighbor_lists': List of (nnb,) arrays with neighbor indices
                - 'neighbor_vectors': List of (nnb, 3) arrays with
                     displacement vectors
        """
        neighbor_group = self._fp.root.structures.neighbor_info

        # Access per-structure group
        group_name = f'structure_{structure_idx:06d}'
        if group_name not in neighbor_group:
            raise ValueError(
                f"No neighbor info found for structure {structure_idx}"
            )

        struc_group = neighbor_group._f_get_child(group_name)

        # Read neighbor counts
        neighbor_counts = struc_group.neighbor_counts[:]

        # Read neighbor lists
        neighbor_lists = [
            struc_group.neighbor_lists[i] for i in range(num_atoms)
        ]

        # Read and reshape neighbor vectors
        neighbor_vectors = []
        for i in range(num_atoms):
            nbvecs_flat = struc_group.neighbor_vectors[i]
            num_neighbors = neighbor_counts[i]
            nbvecs = nbvecs_flat.reshape(num_neighbors, 3)
            neighbor_vectors.append(nbvecs)

        return {
            'neighbor_counts': neighbor_counts,
            'neighbor_lists': neighbor_lists,
            'neighbor_vectors': neighbor_vectors
        }

    def _read_cell_info_hdf5(self, structure_idx: int):
        """
        Read cell information for a specific structure from HDF5.

        Args:
            structure_idx: Index of the structure

        Returns:
            Tuple of (cell, pbc) where:
                - cell: (3, 3) numpy array of lattice vectors
                - pbc: boolean, True for 3D-periodic
        """
        cells_group = self._fp.root.structures.cells

        # Access per-structure group
        group_name = f'structure_{structure_idx:06d}'
        if group_name not in cells_group:
            raise ValueError(
                f"No cell info found for structure {structure_idx}"
            )

        cell_group = cells_group._f_get_child(group_name)

        # Read lattice vectors
        cell = cell_group.lattice_vectors[:]

        # Read pbc (single boolean)
        pbc = bool(cell_group.pbc[()])

        return cell, pbc

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
