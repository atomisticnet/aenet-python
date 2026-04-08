"""
High-level Python interface to aenet library for predictions.

This module provides LibAenetInterface, which uses the libaenet CFFI
wrapper to provide fast energy and force predictions for AtomicStructure
objects without subprocess overhead.

"""
from typing import Optional, Union

import numpy as np

from ..geometry import AtomicStructure
from . import libaenet
from ._evaluation import (
    evaluate_prepared_structure,
    prepare_atomic_structure,
    validate_atom_types,
)
from .ensemble import (
    AenetEnsembleResult,
    normalize_ensemble_members,
    validate_aggregation_mode,
)

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
        potential_paths: dict[str, str],
        potential_format: Optional[str] = None
    ):
        self.potential_paths = potential_paths
        self.potential_format = potential_format
        self._atom_types = list(potential_paths.keys())
        self._session = None
        self._free_atom_energy = None  # cached per-type free-atom energies

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
        validate_atom_types(
            getattr(structure, "types", []),
            self._atom_types,
        )

        self._ensure_session()

        # Get cutoff radius
        _, Rc_max = libaenet.get_cutoff_radius()
        prepared = prepare_atomic_structure(structure, Rc_max)
        return evaluate_prepared_structure(prepared, forces=forces)

    def __del__(self):
        """Cleanup: release the session."""
        session = getattr(self, "_session", None)
        if session is not None:
            session.release()


class AenetEnsembleInterface:
    """
    Ensemble-capable libaenet interface for energy and force predictions.

    Parameters
    ----------
    members : List[Dict[str, str]]
        Per-member mappings from element symbols to potential file paths.
    potential_format : str, optional
        Format of potential files: 'ascii' or None (binary).
    aggregation : {"mean", "reference"}, optional
        Aggregation mode for reported energy and forces. The default
        returns the committee mean.
    reference_member : int, optional
        Member index used when ``aggregation='reference'``.
    """

    def __init__(
        self,
        members: list[dict[str, str]],
        potential_format: Optional[str] = None,
        aggregation: str = "mean",
        reference_member: int = 0,
    ):
        self.members = normalize_ensemble_members(members)
        self.potential_format = potential_format
        self.aggregation = aggregation
        self.reference_member = reference_member
        validate_aggregation_mode(
            aggregation=self.aggregation,
            reference_member=self.reference_member,
            num_members=len(self.members),
        )

        self._atom_types = list(self.members[0].keys())
        self._session = None
        self._active_member_index = None
        self._cutoff_radius = None

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
        """Evaluate a prepared structure with a specific member."""
        self._activate_member(member_index)
        return evaluate_prepared_structure(prepared, forces=forces)

    def predict(
        self,
        structure: AtomicStructure,
        forces: bool = False,
    ) -> AenetEnsembleResult:
        """
        Predict energy and optionally forces for a structure.

        Parameters
        ----------
        structure : AtomicStructure
            The atomic structure to evaluate.
        forces : bool, optional
            If True, also calculate atomic forces.

        Returns
        -------
        AenetEnsembleResult
            Aggregated ensemble prediction and uncertainty estimates.
        """
        validate_atom_types(
            getattr(structure, "types", []),
            self._atom_types,
        )
        _, Rc_max = self._get_cutoff_radius()
        prepared = prepare_atomic_structure(structure, Rc_max)

        member_energies = []
        member_forces = [] if forces else None
        for member_index in range(len(self.members)):
            prediction = self._predict_member(
                prepared,
                member_index=member_index,
                forces=forces,
            )
            if forces:
                energy, force_array = prediction
                member_energies.append(energy)
                member_forces.append(force_array)
            else:
                member_energies.append(prediction)

        return AenetEnsembleResult.from_member_predictions(
            member_energies=member_energies,
            member_forces=member_forces,
            aggregation=self.aggregation,
            reference_member=self.reference_member,
        )

    def __del__(self):
        """Cleanup: release the active ensemble member session."""
        session = getattr(self, "_session", None)
        if session is not None:
            session.release()
