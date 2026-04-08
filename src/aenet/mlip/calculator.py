"""
ASE Calculator interface for aenet potentials.

This module provides AenetCalculator, which uses ASE's native neighbor
lists and directly interfaces with libaenet for fast calculations.

"""
from typing import Optional

import numpy as np

from . import libaenet
from ._evaluation import (
    evaluate_prepared_structure,
    prepare_ase_atoms,
    validate_atom_types,
)
from .ensemble import (
    AenetEnsembleResult,
    normalize_ensemble_members,
    validate_aggregation_mode,
)

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
    skin : float, optional
        Skin distance for neighbor list in Angstroms. The neighbor list
        will only be rebuilt when atoms move more than skin/2 from their
        positions at the last rebuild. Larger values reduce rebuild frequency
        but increase memory usage. Recommended: 0.3-1.0 Å for typical MD.
        Default: 0.5
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
    >>> # For MD simulations, optionally adjust skin parameter
    >>> calc = AenetASECalculator({
    ...     'Ti': 'Ti.nn',
    ...     'O': 'O.nn'
    ... }, skin=1.0)  # Larger skin for fewer rebuilds
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
        potential_paths: dict[str, str],
        potential_format: Optional[str] = None,
        skin: float = 0.5,
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
        self.skin = skin  # Skin distance for neighbor list (in Angstroms)
        self._atom_types = list(potential_paths.keys())
        self._session = None

        # Neighbor list caching
        self._neighbor_list = None
        self._nl_cutoffs = None
        self._nl_natoms = None

    def _ensure_session(self):
        """Acquire libaenet session if needed."""
        if self._session is None or not self._session.is_current():
            if self._session is not None:
                self._session.release()
            self._session = libaenet._session_manager.acquire_session(
                self._atom_types,
                self.potential_paths,
                self.potential_format
            )

    def calculate(
        self,
        atoms: Optional['Atoms'] = None,
        properties: list[str] = ['energy'],
        system_changes: list[str] = all_changes
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
        symbols = self.atoms.get_chemical_symbols()
        natoms = len(self.atoms)

        # Validate atom types
        validate_atom_types(symbols, self._atom_types)

        # Get cutoff radius from libaenet
        _, Rc_max = libaenet.get_cutoff_radius()

        # Build or reuse ASE neighbor list with caching
        from ase.neighborlist import NeighborList
        cutoffs = [Rc_max / 2.0] * natoms  # ASE uses half-cutoffs

        # Check if we need to rebuild the neighbor list
        rebuild_needed = (
            self._neighbor_list is None or
            self._nl_natoms != natoms or
            self._nl_cutoffs != cutoffs or
            'numbers' in system_changes or
            'cell' in system_changes or
            'pbc' in system_changes
        )

        if rebuild_needed:
            # Create new neighbor list with skin for efficient updates
            self._neighbor_list = NeighborList(
                cutoffs, skin=self.skin, sorted=False, self_interaction=False,
                bothways=True
            )
            self._nl_cutoffs = cutoffs
            self._nl_natoms = natoms

        # Update neighbor list (efficient if atoms haven't moved much)
        self._neighbor_list.update(self.atoms)
        nl = self._neighbor_list

        # Determine if forces are needed
        calc_forces = 'forces' in properties

        prepared = prepare_ase_atoms(self.atoms, nl)
        prediction = evaluate_prepared_structure(prepared, forces=calc_forces)

        # Store results
        if calc_forces:
            total_energy, total_forces = prediction
        else:
            total_energy = prediction
            total_forces = None

        self.results['energy'] = total_energy
        if calc_forces:
            self.results['forces'] = total_forces

    def __del__(self):
        """Cleanup."""
        session = getattr(self, "_session", None)
        if session is not None:
            session.release()


class AenetEnsembleCalculator(Calculator):
    """
    ASE Calculator interface for committee-based aenet inference.

    Parameters
    ----------
    members : List[Dict[str, str]]
        Per-member mappings from element symbols to potential file paths.
    potential_format : str, optional
        Format of potential files: 'ascii' or None (binary).
    aggregation : {"mean", "reference"}, optional
        Aggregation mode for reported energy and forces.
    reference_member : int, optional
        Member index used when ``aggregation='reference'``.
    skin : float, optional
        ASE neighbor-list skin distance in Angstrom.
    **kwargs
        Additional ASE calculator keyword arguments.
    """

    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        members: list[dict[str, str]],
        potential_format: Optional[str] = None,
        aggregation: str = "mean",
        reference_member: int = 0,
        skin: float = 0.5,
        **kwargs
    ):
        if not ASE_AVAILABLE:
            raise ImportError(
                "ASE is required for AenetEnsembleCalculator. "
                "Install with: pip install aenet[ase]"
            )

        Calculator.__init__(self, **kwargs)

        self.members = normalize_ensemble_members(members)
        self.potential_format = potential_format
        self.aggregation = aggregation
        self.reference_member = reference_member
        validate_aggregation_mode(
            aggregation=self.aggregation,
            reference_member=self.reference_member,
            num_members=len(self.members),
        )

        self.skin = skin
        self._atom_types = list(self.members[0].keys())
        self._session = None
        self._active_member_index = None
        self._cutoff_radius = None

        self._neighbor_list = None
        self._nl_cutoffs = None
        self._nl_natoms = None

    def _activate_member(self, member_index: int):
        """Load the requested ensemble member into libaenet."""
        if (
            self._session is None
            or not self._session.is_current()
            or self._active_member_index != member_index
        ):
            if self._session is not None:
                self._session.release()
            self._session = libaenet._session_manager.acquire_session(
                self._atom_types,
                self.members[member_index],
                self.potential_format,
            )
            self._active_member_index = member_index

    def _get_cutoff_radius(self) -> tuple[float, float]:
        """
        Get the shared cutoff radius for all ensemble members.

        Raises
        ------
        ValueError
            If ensemble members use different cutoff radii.
        """
        if self._cutoff_radius is not None:
            return self._cutoff_radius

        cutoffs = []
        for member_index in range(len(self.members)):
            self._activate_member(member_index)
            cutoffs.append(libaenet.get_cutoff_radius())

        reference_cutoff = cutoffs[0]
        for cutoff in cutoffs[1:]:
            if not np.allclose(cutoff, reference_cutoff, atol=0.0, rtol=0.0):
                raise ValueError(
                    "All ensemble members must have identical cutoff radii."
                )

        self._cutoff_radius = reference_cutoff
        return self._cutoff_radius

    def _predict_member(self, prepared, member_index: int, forces: bool = False):
        """Evaluate a prepared ASE structure with a specific member."""
        self._activate_member(member_index)
        return evaluate_prepared_structure(prepared, forces=forces)

    def calculate(
        self,
        atoms: Optional['Atoms'] = None,
        properties: list[str] = ['energy'],
        system_changes: list[str] = all_changes
    ):
        """
        Calculate ensemble predictions using ASE neighbor lists.

        Parameters
        ----------
        atoms : ase.Atoms, optional
            Atoms object to calculate properties for.
        properties : list of str
            Requested properties, e.g. ``['energy', 'forces']``.
        system_changes : list of str
            ASE change markers since the last calculation.
        """
        Calculator.calculate(self, atoms, properties, system_changes)

        symbols = self.atoms.get_chemical_symbols()
        natoms = len(self.atoms)
        validate_atom_types(symbols, self._atom_types)

        _, Rc_max = self._get_cutoff_radius()

        from ase.neighborlist import NeighborList
        cutoffs = [Rc_max / 2.0] * natoms
        rebuild_needed = (
            self._neighbor_list is None or
            self._nl_natoms != natoms or
            self._nl_cutoffs != cutoffs or
            'numbers' in system_changes or
            'cell' in system_changes or
            'pbc' in system_changes
        )

        if rebuild_needed:
            self._neighbor_list = NeighborList(
                cutoffs, skin=self.skin, sorted=False, self_interaction=False,
                bothways=True
            )
            self._nl_cutoffs = cutoffs
            self._nl_natoms = natoms

        self._neighbor_list.update(self.atoms)
        prepared = prepare_ase_atoms(self.atoms, self._neighbor_list)
        calc_forces = 'forces' in properties

        member_energies = []
        member_forces = [] if calc_forces else None
        for member_index in range(len(self.members)):
            prediction = self._predict_member(
                prepared,
                member_index=member_index,
                forces=calc_forces,
            )
            if calc_forces:
                energy, force_array = prediction
                member_energies.append(energy)
                member_forces.append(force_array)
            else:
                member_energies.append(prediction)

        result = AenetEnsembleResult.from_member_predictions(
            member_energies=member_energies,
            member_forces=member_forces,
            aggregation=self.aggregation,
            reference_member=self.reference_member,
        )

        self.results['energy'] = result.energy
        self.results['energy_mean'] = result.energy_mean
        self.results['energy_std'] = result.energy_std
        if calc_forces:
            self.results['forces'] = result.forces
            self.results['forces_mean'] = result.forces_mean
            self.results['forces_std'] = result.forces_std
            self.results['force_uncertainty'] = result.force_uncertainty

    def __del__(self):
        """Cleanup."""
        session = getattr(self, "_session", None)
        if session is not None:
            session.release()
