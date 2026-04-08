"""Shared ensemble helpers for aenet inference backends."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np


def normalize_ensemble_members(
    members: Sequence[dict[str, str]]
) -> list[dict[str, str]]:
    """
    Validate and normalize ensemble member potential definitions.

    Parameters
    ----------
    members : Sequence[Dict[str, str]]
        Per-member mappings from atom types to potential paths.

    Returns
    -------
    List[Dict[str, str]]
        Shallow copies of the validated member mappings.

    Raises
    ------
    ValueError
        If no members are provided or the atom-type sets differ.
    """
    if len(members) == 0:
        raise ValueError("At least one ensemble member must be provided.")

    normalized_members = [dict(member) for member in members]
    reference_types = set(normalized_members[0].keys())

    if len(reference_types) == 0:
        raise ValueError(
            "Ensemble members must define at least one atom type."
        )

    for index, member in enumerate(normalized_members[1:], start=1):
        member_types = set(member.keys())
        if member_types != reference_types:
            raise ValueError(
                "All ensemble members must define the same atom types. "
                f"Member 0 has {sorted(reference_types)}, but member "
                f"{index} has {sorted(member_types)}."
            )

    return normalized_members


def validate_aggregation_mode(
    aggregation: str,
    reference_member: int,
    num_members: int,
):
    """
    Validate ensemble aggregation settings.

    Parameters
    ----------
    aggregation : str
        Requested aggregation mode.
    reference_member : int
        Reference member index used when ``aggregation='reference'``.
    num_members : int
        Number of ensemble members.

    Raises
    ------
    ValueError
        If the aggregation settings are invalid.
    """
    if aggregation not in {"mean", "reference"}:
        raise ValueError(
            "aggregation must be 'mean' or 'reference', "
            f"got {aggregation!r}"
        )
    if not 0 <= int(reference_member) < int(num_members):
        raise ValueError(
            f"reference_member must be in [0, {num_members - 1}], "
            f"got {reference_member}"
        )


@dataclass
class AenetEnsembleResult:
    """
    Aggregated result from multiple independently trained potentials.

    Attributes
    ----------
    energy : float
        Reported energy after applying the selected aggregation mode.
    forces : np.ndarray or None
        Reported forces after applying the selected aggregation mode.
    energy_mean : float
        Mean energy across ensemble members.
    energy_std : float
        Standard deviation of energy across ensemble members.
    forces_mean : np.ndarray or None
        Mean per-atom forces across ensemble members.
    forces_std : np.ndarray or None
        Standard deviation of per-atom force components across members.
    force_uncertainty : np.ndarray or None
        Mean per-atom standard deviation over Cartesian force components.
    member_energies : np.ndarray
        Raw per-member energies.
    member_forces : np.ndarray or None
        Raw per-member force arrays.
    aggregation : str
        Aggregation mode used for the reported ``energy`` and ``forces``.
    reference_member : int
        Reference member index used when ``aggregation='reference'``.
    """

    energy: float
    forces: Optional[np.ndarray]
    energy_mean: float
    energy_std: float
    forces_mean: Optional[np.ndarray]
    forces_std: Optional[np.ndarray]
    force_uncertainty: Optional[np.ndarray]
    member_energies: np.ndarray
    member_forces: Optional[np.ndarray]
    aggregation: str
    reference_member: int

    @property
    def num_members(self) -> int:
        """Number of ensemble members."""
        return len(self.member_energies)

    @classmethod
    def from_member_predictions(
        cls,
        member_energies,
        member_forces=None,
        aggregation: str = "mean",
        reference_member: int = 0,
    ) -> "AenetEnsembleResult":
        """
        Build an ensemble result from raw member predictions.

        Parameters
        ----------
        member_energies : array-like
            Per-member total energies.
        member_forces : array-like, optional
            Per-member force arrays with shape ``(n_members, natoms, 3)``.
        aggregation : {"mean", "reference"}, optional
            Aggregation mode used for the reported energy and forces.
        reference_member : int, optional
            Reference member index for ``aggregation='reference'``.

        Returns
        -------
        AenetEnsembleResult
            Aggregated ensemble prediction.
        """
        energies = np.asarray(member_energies, dtype=np.float64)
        if energies.ndim != 1:
            raise ValueError(
                "member_energies must be a 1D array-like sequence."
            )

        validate_aggregation_mode(
            aggregation=aggregation,
            reference_member=reference_member,
            num_members=len(energies),
        )

        forces_array = None
        if member_forces is not None:
            forces_array = np.asarray(member_forces, dtype=np.float64)
            if forces_array.ndim != 3:
                raise ValueError(
                    "member_forces must have shape (n_members, natoms, 3)."
                )
            if forces_array.shape[0] != len(energies):
                raise ValueError(
                    "member_forces and member_energies must have the same "
                    "number of members."
                )

        energy_mean = float(np.mean(energies))
        energy_std = float(np.std(energies))

        if forces_array is None:
            forces_mean = None
            forces_std = None
            force_uncertainty = None
        else:
            forces_mean = np.mean(forces_array, axis=0)
            forces_std = np.std(forces_array, axis=0)
            force_uncertainty = np.mean(forces_std, axis=1)

        if aggregation == "mean":
            reported_energy = energy_mean
            reported_forces = forces_mean
        else:
            reported_energy = float(energies[reference_member])
            reported_forces = (
                None
                if forces_array is None
                else np.array(forces_array[reference_member], copy=True)
            )

        return cls(
            energy=reported_energy,
            forces=reported_forces,
            energy_mean=energy_mean,
            energy_std=energy_std,
            forces_mean=forces_mean,
            forces_std=forces_std,
            force_uncertainty=force_uncertainty,
            member_energies=energies,
            member_forces=forces_array,
            aggregation=aggregation,
            reference_member=reference_member,
        )
