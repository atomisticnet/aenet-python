"""
ASE Calculator interface for aenet potentials.

This module provides AenetCalculator, which uses ASE's native neighbor
lists and directly interfaces with libaenet for fast calculations.

"""

from typing import Dict, List, Optional
import numpy as np

from . import libaenet

__author__ = "The aenet developers"
__email__ = "aenet@atomistic.net"


# Check if ASE is available
try:
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    Calculator = object  # Dummy base class
    all_changes = []  # Dummy for type hints


class AenetCalculator(Calculator):
    """
    ASE Calculator interface for aenet neural network potentials.

    This calculator uses ASE's native neighbor lists and directly manages
    the libaenet library lifecycle for optimal performance.

    Parameters
    ----------
    potential_paths : Dict[str, str]
        Dictionary mapping element symbols to potential file paths
    potential_format : str, optional
        Format of potential files: 'ascii' or None (binary).
        Default: None (binary)
    **kwargs
        Additional keyword arguments passed to ASE Calculator

    Examples
    --------
    >>> from ase.build import bulk
    >>> from aenet.mlip import AenetASECalculator
    >>>
    >>> # Create calculator
    >>> calc = AenetASECalculator({
    ...     'Ti': 'Ti.nn',
    ...     'O': 'O.nn'
    ... })
    >>>
    >>> # Attach to atoms object
    >>> atoms = bulk('TiO2', 'rutile', a=4.6, c=2.95)
    >>> atoms.calc = calc
    >>>
    >>> # Use standard ASE methods
    >>> energy = atoms.get_potential_energy()
    >>> forces = atoms.get_forces()
    """

    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        potential_paths: Dict[str, str],
        potential_format: Optional[str] = None,
        **kwargs
    ):
        if not ASE_AVAILABLE:
            raise ImportError(
                "ASE is required for AenetCalculator. "
                "Install with: pip install aenet[ase]"
            )

        Calculator.__init__(self, **kwargs)

        self.potential_paths = potential_paths
        self.potential_format = potential_format
        self._atom_types = list(potential_paths.keys())
        self._session = None

    def _ensure_session(self):
        """Acquire libaenet session if needed."""
        if self._session is None:
            self._session = libaenet._session_manager.acquire_session(
                self._atom_types,
                self.potential_paths,
                self.potential_format
            )

    def calculate(
        self,
        atoms: Optional['Atoms'] = None,
        properties: List[str] = ['energy'],
        system_changes: List[str] = all_changes
    ):
        """
        Calculate properties using ASE neighbor lists.

        Parameters
        ----------
        atoms : ase.Atoms, optional
            Atoms object to calculate properties for
        properties : list of str
            List of properties to calculate ('energy', 'forces')
        system_changes : list of str
            List of changes since last calculation
        """
        # Call parent class to handle caching and change detection
        Calculator.calculate(self, atoms, properties, system_changes)

        self._ensure_session()

        # Get structure info
        positions = self.atoms.get_positions()
        symbols = self.atoms.get_chemical_symbols()
        cell = self.atoms.get_cell()
        natoms = len(self.atoms)

        # Validate atom types
        unknown = sorted(set(symbols) - set(self._atom_types))
        if len(unknown) > 0:
            raise ValueError(
                f"Structure contains unknown atom types: {unknown}"
            )

        # Map symbols to type IDs
        type_ids = [libaenet.get_type_id(sym) for sym in symbols]

        # Get cutoff radius from libaenet
        Rc_min, Rc_max = libaenet.get_cutoff_radius()

        # Build ASE neighbor list
        from ase.neighborlist import NeighborList
        cutoffs = [Rc_max / 2.0] * natoms  # ASE uses half-cutoffs
        nl = NeighborList(
            cutoffs, skin=0.0, sorted=False, self_interaction=False,
            bothways=True
        )
        nl.update(self.atoms)

        # Determine if forces are needed
        calc_forces = 'forces' in properties

        # Initialize results
        total_energy = 0.0
        if calc_forces:
            total_forces = np.zeros((natoms, 3), dtype=np.float64)
        else:
            total_forces = None

        # Calculate energy and forces for each atom
        for i in range(natoms):
            # Get neighbors from ASE
            indices, offsets = nl.get_neighbors(i)
            n_neighbors = len(indices)

            # Prepare neighbor data
            coo_i = positions[i]
            type_i = type_ids[i]

            if n_neighbors == 0:
                neighbor_coords = np.empty((0, 3), dtype=np.float64)
                neighbor_types = np.empty((0,), dtype=np.int32)
                neighbor_indices = np.empty((0,), dtype=np.int32)
            else:
                # Compute neighbor positions (with PBC shifts)
                neighbor_coords = positions[indices] + offsets @ cell.array
                neighbor_coords = np.asarray(
                    neighbor_coords, dtype=np.float64
                )
                neighbor_types = np.array(
                    [type_ids[j] for j in indices], dtype=np.int32
                )
                neighbor_indices = np.array(
                    indices, dtype=np.int32
                ) + 1  # 1-based for Fortran

            # Call libaenet
            if calc_forces:
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
                total_energy += E_i
                total_forces = F
            else:
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
                total_energy += E_i

        # Store results
        self.results['energy'] = total_energy
        if calc_forces:
            self.results['forces'] = total_forces

    def __del__(self):
        """Cleanup."""
        if self._session is not None:
            self._session.release()
