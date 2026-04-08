"""
Internal helpers for prepared aenet structure evaluation.

These utilities let multiple high-level interfaces share the same
neighbor-list preparation and libaenet evaluation logic.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreparedAtomEnvironment:
    """Prepared neighbor information for a single central atom."""

    coord: np.ndarray
    neighbor_coords: np.ndarray
    neighbor_indices: np.ndarray


@dataclass(frozen=True)
class PreparedAenetStructure:
    """Prepared structure data ready for repeated libaenet evaluation."""

    atom_types: tuple[str, ...]
    atom_environments: tuple[PreparedAtomEnvironment, ...]
    natoms: int


def validate_atom_types(
    atom_types: Iterable[str],
    allowed_atom_types: Sequence[str],
    context: str = "Structure",
):
    """
    Validate that all atom types are covered by the loaded potentials.

    Parameters
    ----------
    atom_types : Iterable[str]
        Atom types to validate.
    allowed_atom_types : Sequence[str]
        Atom types provided by the potential set.
    context : str, optional
        Prefix used in the error message.

    Raises
    ------
    ValueError
        If any atom type is not available in the potential set.
    """
    normalized_types = [str(atom_type) for atom_type in atom_types]
    unknown = sorted(set(normalized_types) - set(allowed_atom_types))
    if len(unknown) > 0:
        raise ValueError(
            f"{context} contains unknown atom types: {unknown}; "
            f"expected subset of {list(allowed_atom_types)}"
        )


def prepare_atomic_structure(structure, cutoff_radius: float) -> PreparedAenetStructure:
    """
    Prepare an AtomicStructure for repeated libaenet evaluation.

    Parameters
    ----------
    structure : AtomicStructure
        Structure to prepare.
    cutoff_radius : float
        Neighbor cutoff radius in Angstrom.

    Returns
    -------
    PreparedAenetStructure
        Reusable structure representation for libaenet evaluations.
    """
    from ..nblist import NeighborList

    coords = np.asarray(structure.coords[-1], dtype=np.float64)
    atom_types = tuple(str(atom_type) for atom_type in structure.types)

    if structure.pbc:
        nl = NeighborList.from_AtomicStructure(
            structure, frame=-1, interaction_range=cutoff_radius
        )
    else:
        nl = NeighborList(
            coords,
            lattice_vectors=None,
            cartesian=True,
            types=structure.types,
            interaction_range=cutoff_radius,
        )

    atom_envs: list[PreparedAtomEnvironment] = []

    for i in range(len(atom_types)):
        neighbors_raw, _, translations = nl.get_neighbors_and_distances(
            i,
            r=cutoff_radius,
            return_coords=False,
            return_self=False,
        )
        neighbors = np.asarray(neighbors_raw, dtype=np.int32)

        if len(neighbors) == 0:
            atom_envs.append(
                PreparedAtomEnvironment(
                    coord=coords[i],
                    neighbor_coords=np.empty((0, 3), dtype=np.float64),
                    neighbor_indices=np.empty((0,), dtype=np.int32),
                )
            )
            continue

        if structure.pbc:
            lattice = np.asarray(structure.avec[-1], dtype=np.float64)
            translation_array = np.asarray(translations, dtype=np.int32)
            shifts = np.dot(translation_array, lattice)
            mask = ~(
                (neighbors == i) & np.all(translation_array == 0, axis=1)
            )
            neighbors = neighbors[mask]
            shifts = shifts[mask]
            neighbor_coords = (coords[neighbors] + shifts).astype(np.float64)
        else:
            mask = neighbors != i
            neighbors = neighbors[mask]
            neighbor_coords = coords[neighbors].astype(np.float64)

        atom_envs.append(
            PreparedAtomEnvironment(
                coord=coords[i],
                neighbor_coords=neighbor_coords,
                neighbor_indices=neighbors.astype(np.int32, copy=False),
            )
        )

    return PreparedAenetStructure(
        atom_types=atom_types,
        atom_environments=tuple(atom_envs),
        natoms=len(atom_types),
    )


def prepare_ase_atoms(atoms, neighbor_list) -> PreparedAenetStructure:
    """
    Prepare an ASE Atoms object for repeated libaenet evaluation.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to prepare.
    neighbor_list : ase.neighborlist.NeighborList
        Updated ASE neighbor list matching the target cutoff.

    Returns
    -------
    PreparedAenetStructure
        Reusable structure representation for libaenet evaluations.
    """
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    cell = atoms.get_cell().array
    atom_types = tuple(str(symbol) for symbol in atoms.get_chemical_symbols())
    atom_envs: list[PreparedAtomEnvironment] = []

    for i in range(len(atom_types)):
        indices, offsets = neighbor_list.get_neighbors(i)
        neighbors = np.asarray(indices, dtype=np.int32)

        if len(neighbors) == 0:
            atom_envs.append(
                PreparedAtomEnvironment(
                    coord=positions[i],
                    neighbor_coords=np.empty((0, 3), dtype=np.float64),
                    neighbor_indices=np.empty((0,), dtype=np.int32),
                )
            )
            continue

        neighbor_coords = positions[neighbors] + offsets @ cell
        atom_envs.append(
            PreparedAtomEnvironment(
                coord=positions[i],
                neighbor_coords=np.asarray(neighbor_coords, dtype=np.float64),
                neighbor_indices=neighbors.astype(np.int32, copy=False),
            )
        )

    return PreparedAenetStructure(
        atom_types=atom_types,
        atom_environments=tuple(atom_envs),
        natoms=len(atom_types),
    )


def evaluate_prepared_structure(
    prepared: PreparedAenetStructure,
    forces: bool = False,
):
    """
    Evaluate a prepared structure with the currently loaded libaenet state.

    Parameters
    ----------
    prepared : PreparedAenetStructure
        Prepared structure representation.
    forces : bool, optional
        If True, also evaluate atomic forces.

    Returns
    -------
    float or tuple
        Total energy in eV, and optionally an ``(natoms, 3)`` force array.
    """
    from . import libaenet

    type_ids = np.array(
        [libaenet.get_type_id(atom_type) for atom_type in prepared.atom_types],
        dtype=np.int32,
    )

    total_energy = 0.0
    total_forces = (
        np.zeros((prepared.natoms, 3), dtype=np.float64)
        if forces
        else None
    )

    for i, atom_env in enumerate(prepared.atom_environments):
        neighbor_indices = atom_env.neighbor_indices
        if len(neighbor_indices) == 0:
            neighbor_types = np.empty((0,), dtype=np.int32)
        else:
            neighbor_types = type_ids[neighbor_indices]

        energy_i, force_array = libaenet.atomic_energy_and_forces(
            atom_env.coord,
            int(type_ids[i]),
            i + 1,
            atom_env.neighbor_coords,
            neighbor_types,
            neighbor_indices + 1,
            prepared.natoms,
            forces=total_forces,
        )
        total_energy += energy_i
        if forces:
            total_forces = force_array

    if forces:
        return total_energy, total_forces
    return total_energy
