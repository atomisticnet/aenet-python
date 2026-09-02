"""Regression tests for periodic CSR graph construction."""

from contextlib import nullcontext

import pytest
import torch

from aenet.torch_featurize import ChebyshevDescriptor
from aenet.torch_featurize.graph import (
    _compute_r_ij,
    build_csr_from_neighborlist,
    build_triplets_from_csr,
)
from aenet.torch_nblist import TorchNeighborList

DTYPE = torch.float64
RTOL = 1.0e-10
ATOL = 1.0e-10


def _periodic_case(cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return equivalent wrapped and independently translated positions."""
    fractional = torch.tensor(
        [
            [0.98, 0.25, 0.30],
            [0.02, 0.28, 0.30],
            [0.96, 0.38, 0.35],
        ],
        dtype=DTYPE,
    )
    translations = torch.tensor(
        [[0, 0, 0], [2, -1, 0], [-1, 0, 1]],
        dtype=DTYPE,
    )
    return fractional @ cell, (fractional + translations) @ cell


def _build_graph(
    positions: torch.Tensor,
    cell: torch.Tensor,
    backend: str = "ghost",
):
    nbl = TorchNeighborList(
        cutoff=2.0,
        device="cpu",
        dtype=DTYPE,
        pbc_backend=backend,
    )
    pbc = torch.ones(3, dtype=torch.bool)
    graph = build_csr_from_neighborlist(
        positions=positions,
        cell=cell,
        pbc=pbc,
        nbl=nbl,
        min_cutoff=0.1,
        max_cutoff=2.0,
        dtype=DTYPE,
    )
    return nbl, graph


@pytest.mark.cpu
@pytest.mark.parametrize(
    "cell",
    [
        torch.diag(torch.tensor([10.0, 9.0, 8.0], dtype=DTYPE)),
        torch.tensor(
            [[10.0, 0.0, 0.0], [1.5, 9.0, 0.0], [0.7, 1.1, 8.0]],
            dtype=DTYPE,
        ),
    ],
    ids=["orthorhombic", "triclinic"],
)
@pytest.mark.parametrize("backend", ["ghost", "legacy"])
def test_periodic_graph_is_invariant_to_independent_lattice_translations(
    cell,
    backend,
):
    """Backend offsets and CSR vectors use one wrapped-image convention."""
    wrapped, unwrapped = _periodic_case(cell)
    warning_context = (
        pytest.warns(DeprecationWarning)
        if backend == "legacy"
        else nullcontext()
    )
    with warning_context:
        nbl_wrapped, graph_wrapped = _build_graph(wrapped, cell, backend)
    warning_context = (
        pytest.warns(DeprecationWarning)
        if backend == "legacy"
        else nullcontext()
    )
    with warning_context:
        nbl_unwrapped, graph_unwrapped = _build_graph(unwrapped, cell, backend)

    pbc = torch.ones(3, dtype=torch.bool)
    neighbors_wrapped = nbl_wrapped.get_neighbors(
        wrapped, cell, pbc, fractional=False
    )
    neighbors_unwrapped = nbl_unwrapped.get_neighbors(
        unwrapped, cell, pbc, fractional=False
    )

    assert torch.equal(
        neighbors_wrapped["edge_index"], neighbors_unwrapped["edge_index"]
    )
    assert torch.equal(
        neighbors_wrapped["offsets"], neighbors_unwrapped["offsets"]
    )
    torch.testing.assert_close(
        neighbors_wrapped["distances"],
        neighbors_unwrapped["distances"],
        rtol=RTOL,
        atol=ATOL,
    )
    assert torch.equal(
        graph_wrapped["center_ptr"], graph_unwrapped["center_ptr"]
    )
    assert torch.equal(graph_wrapped["nbr_idx"], graph_unwrapped["nbr_idx"])
    torch.testing.assert_close(
        graph_wrapped["d_ij"], graph_unwrapped["d_ij"], rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(
        graph_wrapped["r_ij"], graph_unwrapped["r_ij"], rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(graph_unwrapped["r_ij"], dim=1),
        graph_unwrapped["d_ij"],
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.cpu
def test_compute_r_ij_preserves_nonperiodic_behavior():
    """The periodic fix must not alter ordinary Cartesian displacements."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 0.5]], dtype=DTYPE
    )
    result = _compute_r_ij(
        positions,
        torch.tensor([0]),
        torch.tensor([1]),
        cell=None,
        offsets=None,
        dtype=DTYPE,
    )
    assert torch.equal(result, positions[1:].clone())


@pytest.mark.cpu
def test_periodic_graph_preserves_position_and_cell_autograd():
    """Wrapped reconstruction remains differentiable through inputs."""
    cell = torch.tensor(
        [[10.0, 0.0, 0.0], [1.5, 9.0, 0.0], [0.7, 1.1, 8.0]],
        dtype=DTYPE,
        requires_grad=True,
    )
    wrapped, unwrapped = _periodic_case(cell)
    positions = unwrapped.detach().requires_grad_(True)
    _, graph = _build_graph(positions, cell)

    objective = graph["r_ij"].square().sum()
    position_gradient, cell_gradient = torch.autograd.grad(
        objective, (positions, cell)
    )

    assert torch.isfinite(position_gradient).all()
    assert torch.isfinite(cell_gradient).all()
    assert torch.count_nonzero(position_gradient) > 0
    assert torch.count_nonzero(cell_gradient) > 0


@pytest.mark.cpu
@pytest.mark.parametrize(
    "cell",
    [
        torch.diag(torch.tensor([10.0, 9.0, 8.0], dtype=DTYPE)),
        torch.tensor(
            [[10.0, 0.0, 0.0], [1.5, 9.0, 0.0], [0.7, 1.1, 8.0]],
            dtype=DTYPE,
        ),
    ],
    ids=["orthorhombic", "triclinic"],
)
def test_graph_descriptors_energies_and_forces_are_image_invariant(cell):
    """Angular graph features and their derivatives ignore atom images."""
    descriptor = ChebyshevDescriptor(
        species=["H"],
        rad_order=2,
        rad_cutoff=2.0,
        ang_order=2,
        ang_cutoff=2.0,
        min_cutoff=0.1,
        device="cpu",
        dtype=DTYPE,
    )
    species = ["H", "H", "H"]
    species_indices = torch.zeros(3, dtype=torch.long)
    pbc = torch.ones(3, dtype=torch.bool)
    weights = torch.linspace(
        -0.4, 0.6, descriptor.get_n_features(), dtype=DTYPE
    )

    results = []
    for positions_value in _periodic_case(cell):
        positions = positions_value.detach().requires_grad_(True)
        _, graph = _build_graph(positions, cell)
        triplets = build_triplets_from_csr(
            graph,
            ang_cutoff=descriptor.ang_cutoff,
            min_cutoff=descriptor.min_cutoff,
        )
        graph_features = descriptor.forward_with_graph(
            positions, species_indices, graph, triplets
        )
        standard_features = descriptor.forward_from_positions(
            positions, species, cell, pbc
        )
        energy = (graph_features * weights).sum()
        forces = -torch.autograd.grad(energy, positions)[0]
        results.append((graph_features, standard_features, energy, forces))

    wrapped_result, unwrapped_result = results
    torch.testing.assert_close(
        wrapped_result[0], wrapped_result[1], rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(
        unwrapped_result[0], unwrapped_result[1], rtol=RTOL, atol=ATOL
    )
    for wrapped_value, unwrapped_value in zip(
        wrapped_result, unwrapped_result, strict=True
    ):
        torch.testing.assert_close(
            wrapped_value, unwrapped_value, rtol=RTOL, atol=ATOL
        )
