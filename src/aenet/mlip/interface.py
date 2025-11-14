"""
High-level Python interface to aenet library for predictions.

This module provides LibAenetInterface, which uses the libaenet CFFI
wrapper to provide fast energy and force predictions for AtomicStructure
objects without subprocess overhead.

"""

import numpy as np
from typing import Dict, Optional, Union

from . import libaenet
from ..geometry import AtomicStructure

__author__ = "The aenet developers"
__email__ = "aenet@atomistic.net"


class LibAenetInterface:
    """
    High-level interface to aenet library for energy and force predictions.

    This interface uses the libaenet CFFI wrapper directly, providing
    fast predictions without subprocess overhead. Suitable for use in MD
    simulations, geometry optimizations, and other iterative calculations.

    Parameters
    ----------
    potential_paths : Dict[str, str]
        Dictionary mapping element symbols to potential file paths
    potential_format : str, optional
        Format of potential files: 'ascii' or None (binary).
        Default: None (binary)

    Examples
    --------
    >>> interface = LibAenetInterface({
    ...     'Ti': 'Ti.nn',
    ...     'O': 'O.nn'
    ... })
    >>> energy, forces = interface.predict(structure, forces=True)
    """

    def __init__(
        self,
        potential_paths: Dict[str, str],
        potential_format: Optional[str] = None
    ):
        self.potential_paths = potential_paths
        self.potential_format = potential_format
        self._atom_types = list(potential_paths.keys())
        self._session = None
        self._free_atom_energy = None  # cached per-type free-atom energies

    def _ensure_session(self):
        """Acquire libaenet session if needed."""
        if self._session is None:
            self._session = libaenet._session_manager.acquire_session(
                self._atom_types,
                self.potential_paths,
                self.potential_format
            )

    def predict(
        self,
        structure: AtomicStructure,
        forces: bool = False
    ) -> Union[float, tuple]:
        """
        Predict energy and optionally forces for a structure.

        Parameters
        ----------
        structure : AtomicStructure
            The atomic structure to evaluate
        forces : bool, optional
            If True, also calculate atomic forces. Default: False

        Returns
        -------
        energy : float
            Total energy in eV
        forces : np.ndarray, optional
            Atomic forces in eV/Angstrom, shape (natoms, 3).
            Only returned if forces=True

        Raises
        ------
        ValueError
            If structure contains unknown atom types
        """

        # Validate atom types before acquiring session to avoid
        # unnecessary init when the structure is incompatible.
        struct_types = [str(t) for t in getattr(structure, "types", [])]
        unknown = sorted(set(struct_types) - set(self._atom_types))
        if len(unknown) > 0:
            raise ValueError(
                f"Structure contains unknown atom types: {unknown}; "
                f"expected subset of {self._atom_types}"
            )

        self._ensure_session()

        # Cache per-type free-atom energies (Fortran provides these)
        if self._free_atom_energy is None:
            self._free_atom_energy = {
                t: float(libaenet.free_atom_energy(t))
                for t in self._atom_types
            }

        # Get cutoff radius
        Rc_min, Rc_max = libaenet.get_cutoff_radius()

        # Build neighbor list helper (NumPy-based NeighborList).
        # We compute neighbors per-atom using the existing neighborlist
        # implementation to obtain neighbor indices and coordinates.
        from ..nblist import NeighborList

        natoms = structure.natoms
        total_energy = 0.0

        if forces:
            total_forces = np.zeros((natoms, 3), dtype=np.float64)
        else:
            total_forces = None

        # Map atom types to type IDs (normalize to plain str
        # to avoid numpy.str_ issues)
        type_ids = [libaenet.get_type_id(str(typ)) for typ in structure.types]

        # Pre-create neighbor list object once for the structure at Rc_max
        if structure.pbc:
            nl = NeighborList.from_AtomicStructure(
                structure, frame=-1, interaction_range=Rc_max
            )
        else:
            nl = NeighborList(
                structure.coords[-1],
                lattice_vectors=None,
                cartesian=True,
                types=structure.types,
                interaction_range=Rc_max,
            )

        # Calculate atomic energies and forces
        for i in range(natoms):
            # Get neighbor indices within Rc_max
            nbl_idx, dist, Tvecs = nl.get_neighbors_and_distances(
                i, r=Rc_max, return_coords=False, return_self=False
            )
            # Keep as numpy array for efficient indexing
            neighbors = np.asarray(nbl_idx, dtype=np.int32)
            n_neighbors = len(neighbors)

            # Prepare neighbor data
            coo_i = structure.coords[-1][i]
            type_i = type_ids[i]

            if n_neighbors == 0:
                neighbor_coords = np.empty((0, 3), dtype=np.float64)
                neighbor_types = np.empty((0,), dtype=np.int32)
                neighbor_indices = np.empty((0,), dtype=np.int32)
            else:
                # Use absolute Cartesian positions (not displacements)
                base_coords = structure.coords[-1]
                if structure.pbc:
                    # Apply periodic image translations to neighbors
                    lattice = np.asarray(structure.avec[-1], dtype=np.float64)
                    Tarr = np.asarray(Tvecs, dtype=np.int32)
                    shifts = np.dot(Tarr, lattice)  # shape (n_neighbors, 3)
                    # Filter out any accidental self in home cell (T == 0)
                    mask = np.ones(len(neighbors), dtype=bool)
                    if len(neighbors) > 0:
                        mask &= ~((neighbors == i) & (
                            np.all(Tarr == 0, axis=1)))
                    neighbors = neighbors[mask]
                    shifts = shifts[mask]
                    n_neighbors = len(neighbors)
                    if n_neighbors == 0:
                        neighbor_coords = np.empty((0, 3), dtype=np.float64)
                        neighbor_types = np.empty((0,), dtype=np.int32)
                        neighbor_indices = np.empty((0,), dtype=np.int32)
                    else:
                        neighbor_coords = (
                            base_coords[neighbors] + shifts).astype(
                            np.float64
                        )
                        neighbor_types = np.array(
                            [type_ids[j] for j in neighbors], dtype=np.int32)
                        neighbor_indices = neighbors + 1  # 1-based for Fortran
                else:
                    # Non-periodic: just absolute base positions,
                    # exclude self if present
                    mask = neighbors != i
                    neighbors = neighbors[mask]
                    n_neighbors = len(neighbors)
                    if n_neighbors == 0:
                        neighbor_coords = np.empty((0, 3), dtype=np.float64)
                        neighbor_types = np.empty((0,), dtype=np.int32)
                        neighbor_indices = np.empty((0,), dtype=np.int32)
                    else:
                        neighbor_coords = base_coords[
                            neighbors].astype(np.float64)
                        neighbor_types = np.array(
                            [type_ids[j] for j in neighbors], dtype=np.int32)
                        neighbor_indices = neighbors + 1  # 1-based for Fortran

            if forces:
                # Calculate energy and forces
                E_i, F = libaenet.atomic_energy_and_forces(
                    coo_i,
                    type_i,
                    i + 1,
                    neighbor_coords,
                    neighbor_types,
                    neighbor_indices,
                    natoms,
                    forces=total_forces
                )
                if np.isnan(E_i):
                    print(f"NaN at atom {i}: coo_i={coo_i}, type={type_i}, "
                          f"n_neighbors={n_neighbors}")
                    if n_neighbors > 0:
                        print(f"  neighbor_coords[0]={neighbor_coords[0]}")
                total_energy += E_i
                total_forces = F  # Update with accumulated forces
            else:
                # Energy only - use simpler interface
                # For now, use the forces interface but ignore forces
                E_i, _ = libaenet.atomic_energy_and_forces(
                    coo_i,
                    type_i,
                    i + 1,
                    neighbor_coords,
                    neighbor_types,
                    neighbor_indices,
                    natoms,
                    forces=None
                )
                if np.isnan(E_i):
                    print(f"NaN at atom {i}: coo_i={coo_i}, type={type_i}, "
                          f"n_neighbors={n_neighbors}")
                    if n_neighbors > 0:
                        print(f"  neighbor_coords[0]={neighbor_coords[0]}")
                total_energy += E_i

        # Library atomic contributions already sum to total energy.
        # Do not add per-atom offsets here to avoid double counting.
        if forces:
            return total_energy, total_forces
        else:
            return total_energy

    def __del__(self):
        """Cleanup: release the session."""
        if self._session is not None:
            self._session.release()
