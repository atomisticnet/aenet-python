"""
Unit tests for TorchNeighborList.
"""

import numpy as np
import pytest
import torch

# Skip all tests if torch_cluster is not available
torch_cluster = pytest.importorskip("torch_cluster")

from aenet.torch_featurize.neighborlist import TorchNeighborList  # noqa: E402


class TestNeighborListIsolated:
    """Test neighbor list for isolated systems (molecules)."""

    def test_water_molecule(self):
        """Test on water molecule."""
        # Water molecule positions from test data
        positions = torch.tensor(
            [
                [0.00000, 0.00000, 0.11779],  # O
                [0.00000, 0.75545, -0.47116],  # H
                [0.00000, -0.75545, -0.47116],  # H
            ],
            dtype=torch.float64,
        )

        nbl = TorchNeighborList(cutoff=4.0, device="cpu")
        result = nbl.get_neighbors(positions)

        # Check output structure
        assert "edge_index" in result
        assert "distances" in result
        assert "offsets" in result
        assert "num_neighbors" in result
        assert result["offsets"] is None  # No PBC for isolated system

        # Check data types
        assert result["edge_index"].dtype == torch.long
        assert result["distances"].dtype == torch.float64

        # Each atom should have 2 neighbors within 4.0 Å cutoff
        assert result["num_neighbors"].sum() == 6  # 3 atoms * 2 neighbors each

        # Check O-H distances (should be ~0.9584 Å)
        edge_index = result["edge_index"]
        distances = result["distances"]

        # Find edges originating from O (atom 0)
        o_edges = edge_index[1][edge_index[0] == 0]
        o_distances = distances[edge_index[0] == 0]

        assert len(o_edges) == 2  # Two H atoms
        assert torch.allclose(
            o_distances, torch.tensor([0.9584], dtype=torch.float64), atol=1e-3
        )

    def test_single_atom(self):
        """Test with single atom (should have no neighbors)."""
        positions = torch.randn(1, 3, dtype=torch.float64)

        nbl = TorchNeighborList(cutoff=2.0, device="cpu")
        result = nbl.get_neighbors(positions)

        assert result["edge_index"].shape == (2, 0)
        assert result["distances"].shape == (0,)
        assert result["num_neighbors"][0] == 0

    def test_no_neighbors_within_cutoff(self):
        """Test atoms far apart (no neighbors within cutoff)."""
        # Two atoms 10 Å apart
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float64
        )

        nbl = TorchNeighborList(cutoff=2.0, device="cpu")
        result = nbl.get_neighbors(positions)

        assert result["edge_index"].shape == (2, 0)
        assert result["distances"].shape == (0,)
        assert result["num_neighbors"].sum() == 0

    def test_all_neighbors_within_cutoff(self):
        """Test small cluster where all atoms are neighbors."""
        # 4 atoms in a small cluster
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )

        nbl = TorchNeighborList(cutoff=2.0, device="cpu")
        result = nbl.get_neighbors(positions)

        # Each atom should see 3 neighbors
        assert result["num_neighbors"].sum() == 12  # 4 atoms * 3 neighbors


class TestNeighborListPBC:
    """Test neighbor list with periodic boundary conditions."""

    def test_cubic_cell_simple(self):
        """Test simple cubic cell."""
        # 2 atoms in a cubic cell (5 Å × 5 Å × 5 Å)
        positions = torch.tensor(
            [
                [0.1, 0.1, 0.1],  # Fractional coordinates
                [0.9, 0.1, 0.1],
            ],
            dtype=torch.float64,
        )

        cell = torch.eye(3, dtype=torch.float64) * 5.0  # 5 Å cubic cell

        nbl = TorchNeighborList(cutoff=2.0, device="cpu")
        result = nbl.get_neighbors(positions, cell=cell)

        # Check output structure
        assert "edge_index" in result
        assert "distances" in result
        assert "offsets" in result
        assert result["offsets"] is not None  # PBC system

        # Atoms at 0.1 and 0.9 in fractional coords are 1.0 Å
        # apart (across boundary). They should see each other
        assert result["num_neighbors"].sum() > 0

    def test_pbc_partial(self):
        """Test with PBC only in some directions."""
        positions = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)

        cell = torch.eye(3, dtype=torch.float64) * 3.0

        # PBC only in x and y directions
        pbc = torch.tensor([True, True, False], dtype=torch.bool)

        nbl = TorchNeighborList(cutoff=2.0, device="cpu")
        result = nbl.get_neighbors(positions, cell=cell, pbc=pbc)

        # Single atom should have no neighbors
        assert result["num_neighbors"][0] == 0

    def test_triclinic_cell(self):
        """Test with non-orthogonal (triclinic) cell."""
        # Simple triclinic cell
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64
        )

        # Triclinic cell with 60° angles
        a = 5.0
        cell = torch.tensor(
            [
                [a, 0.0, 0.0],
                [a * 0.5, a * 0.866, 0.0],
                [a * 0.5, a * 0.289, a * 0.816],
            ],
            dtype=torch.float64,
        )

        nbl = TorchNeighborList(cutoff=4.0, device="cpu")
        result = nbl.get_neighbors(positions, cell=cell)

        # Should find some neighbors
        assert result["edge_index"].shape[1] > 0

    def test_cell_offsets(self):
        """Test that cell offsets are computed correctly."""
        # Two atoms that interact across periodic boundary
        positions = torch.tensor(
            [[0.05, 0.5, 0.5], [0.95, 0.5, 0.5]], dtype=torch.float64
        )

        cell = torch.eye(3, dtype=torch.float64) * 5.0

        nbl = TorchNeighborList(cutoff=1.0, device="cpu")
        result = nbl.get_neighbors(positions, cell=cell)

        # Atoms should interact across boundary
        assert result["edge_index"].shape[1] > 0

        # Check that offsets are non-zero for periodic interactions
        offsets = result["offsets"]
        assert torch.any(offsets != 0)


class TestNeighborListEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_positions(self):
        """Test with empty position array."""
        positions = torch.empty((0, 3), dtype=torch.float64)

        nbl = TorchNeighborList(cutoff=2.0, device="cpu")

        # Should handle gracefully - returns empty results
        result = nbl.get_neighbors(positions)
        assert result["edge_index"].shape == (2, 0)
        assert result["distances"].shape == (0,)
        assert result["num_neighbors"].shape == (0,)

    def test_large_cutoff(self):
        """Test with cutoff larger than cell size."""
        positions = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)

        cell = torch.eye(3, dtype=torch.float64) * 2.0

        nbl = TorchNeighborList(cutoff=10.0, device="cpu")  # Large cutoff
        result = nbl.get_neighbors(positions, cell=cell)

        # Should complete without error
        assert result["edge_index"].shape[0] == 2

    def test_dtype_consistency(self):
        """Test that output dtypes match input."""
        positions = torch.randn(5, 3, dtype=torch.float32)

        nbl = TorchNeighborList(cutoff=2.0, device="cpu", dtype=torch.float32)
        result = nbl.get_neighbors(positions)

        assert result["distances"].dtype == torch.float32

    def test_device_consistency(self):
        """Test that outputs are on correct device."""
        positions = torch.randn(5, 3, dtype=torch.float64)

        nbl = TorchNeighborList(cutoff=2.0, device="cpu")
        result = nbl.get_neighbors(positions)

        assert result["edge_index"].device.type == "cpu"
        assert result["distances"].device.type == "cpu"


class TestNeighborListGPU:
    """Test GPU acceleration (if available)."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_gpu_isolated(self):
        """Test GPU computation for isolated system."""
        positions = torch.randn(100, 3, dtype=torch.float64, device="cuda")

        nbl = TorchNeighborList(cutoff=3.0, device="cuda")
        result = nbl.get_neighbors(positions)

        assert result["edge_index"].device.type == "cuda"
        assert result["distances"].device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_gpu_pbc(self):
        """Test GPU computation for periodic system."""
        positions = torch.rand(50, 3, dtype=torch.float64, device="cuda")
        cell = torch.eye(3, dtype=torch.float64, device="cuda") * 10.0

        nbl = TorchNeighborList(cutoff=3.0, device="cuda")
        result = nbl.get_neighbors(positions, cell=cell)

        assert result["edge_index"].device.type == "cuda"
        assert result["distances"].device.type == "cuda"
        assert result["offsets"].device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cpu_gpu_consistency(self):
        """Test that CPU and GPU give same results (order-independent)."""
        torch.manual_seed(42)
        positions = torch.randn(20, 3, dtype=torch.float64)

        # CPU computation
        nbl_cpu = TorchNeighborList(cutoff=3.0, device="cpu")
        result_cpu = nbl_cpu.get_neighbors(positions)

        # GPU computation
        nbl_gpu = TorchNeighborList(cutoff=3.0, device="cuda")
        result_gpu = nbl_gpu.get_neighbors(positions.to("cuda"))

        # Results should match (order-independent)
        # GPU and CPU may return neighbors in different orders,
        # but the sets of distances and edges should be identical

        # Compare sorted distances (order-independent)
        cpu_dists_sorted = torch.sort(result_cpu["distances"])[0]
        gpu_dists_sorted = torch.sort(result_gpu["distances"].cpu())[0]
        assert torch.allclose(
            cpu_dists_sorted, gpu_dists_sorted, rtol=1e-12), \
            "CPU and GPU distances don't match (after sorting)"

        # Check that same number of edges found
        assert result_cpu["edge_index"
                          ].shape == result_gpu["edge_index"].shape, \
            "CPU and GPU found different number of edges"

        # Verify same unique pairs exist (order-independent)
        cpu_edges = set(
            tuple(result_cpu["edge_index"][:, i].tolist())
            for i in range(result_cpu["edge_index"].shape[1])
        )
        gpu_edges = set(
            tuple(result_gpu["edge_index"][:, i].cpu().tolist())
            for i in range(result_gpu["edge_index"].shape[1])
        )
        assert cpu_edges == gpu_edges, \
            "CPU and GPU found different neighbor pairs"


class TestNeighborListValidation:
    """Validation tests against known reference data."""

    def test_water_distances(self):
        """Validate O-H distances in water molecule."""
        # Water molecule from reference data
        positions = torch.tensor(
            [
                [0.00000, 0.00000, 0.11779],  # O
                [0.00000, 0.75545, -0.47116],  # H
                [0.00000, -0.75545, -0.47116],  # H
            ],
            dtype=torch.float64,
        )

        nbl = TorchNeighborList(cutoff=4.0, device="cpu")
        result = nbl.get_neighbors(positions)

        edge_index = result["edge_index"]
        distances = result["distances"]

        # Expected O-H distance: ~0.9584 Å
        expected_oh_dist = 0.9584

        # Find O-H distances
        oh_mask = (edge_index[0] == 0) & (
            (edge_index[1] == 1) | (edge_index[1] == 2)
        )
        oh_distances = distances[oh_mask]

        assert len(oh_distances) == 2
        assert torch.allclose(
            oh_distances,
            torch.tensor([expected_oh_dist], dtype=torch.float64),
            atol=1e-3,
        )

        # Expected H-H distance: ~1.5109 Å (from geometry)
        # H at (0, 0.75545, -0.47116) and (0, -0.75545, -0.47116)
        expected_hh_dist = np.sqrt((2 * 0.75545) ** 2)

        hh_mask = ((edge_index[0] == 1) & (edge_index[1] == 2)) | (
            (edge_index[0] == 2) & (edge_index[1] == 1)
        )
        hh_distances = distances[hh_mask]

        assert len(hh_distances) == 2  # Symmetric
        assert torch.allclose(
            hh_distances,
            torch.tensor([expected_hh_dist], dtype=torch.float64),
            atol=1e-3,
        )

    def test_symmetry(self):
        """Test that neighbor list is symmetric (i->j implies j->i)."""
        positions = torch.randn(10, 3, dtype=torch.float64)

        nbl = TorchNeighborList(cutoff=3.0, device="cpu")
        result = nbl.get_neighbors(positions)

        edge_index = result["edge_index"]

        # Create set of edges
        edges = set()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edges.add((src, dst))

        # Check symmetry
        for src, dst in edges:
            assert (dst, src) in edges, f"Edge ({src}, {dst}) not symmetric"


class TestNeighborListFCC:
    """Rigorous tests based on FCC lattice (from Fortran test suite)."""

    def test_fcc_primitive_cell(self):
        """
        Test FCC primitive unit cell.

        Based on test_fcc_primitive() from fortran/test_lclist.f90.
        Expected: Exactly 12 nearest neighbors for the single atom.
        """
        # FCC primitive lattice vectors (a = 5.0 Å)
        a = 5.0
        cell = (
            torch.tensor(
                [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
                dtype=torch.float64,
            )
            * a
        )

        # Single atom at origin (fractional coordinates)
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)

        # Nearest neighbor distance in FCC: d_NN = 0.5 * sqrt(2) * a
        d_nn = 0.5 * np.sqrt(2.0) * a
        cutoff = d_nn + 0.1  # Slightly larger than NN distance

        nbl = TorchNeighborList(cutoff=cutoff, device="cpu")
        result = nbl.get_neighbors(positions, cell=cell)

        # Should find exactly 12 nearest neighbors
        assert result["num_neighbors"][0] == 12, (
            f"Expected 12 neighbors, got {result['num_neighbors'][0]}"
        )

        # All distances should be approximately d_nn
        distances = result["distances"]
        assert torch.allclose(
            distances, torch.full_like(distances, d_nn), atol=1e-10
        ), "Not all neighbors are at nearest neighbor distance"

    def test_fcc_conventional_supercell(self):
        """
        Test FCC conventional cell as 4x4x4 supercell.

        Based on test_fcc_conventional() from fortran/test_lclist.f90.
        Every atom should have exactly 12 nearest neighbors.
        """
        # FCC conventional cell (a = 5.0 Å), scaled by 4
        a = 5.0
        cell = torch.eye(3, dtype=torch.float64) * a * 4.0

        # FCC basis in conventional cell
        basis = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            dtype=torch.float64,
        )

        # Create 4x4x4 supercell (256 atoms total)
        positions = []
        for ix in range(4):
            for iy in range(4):
                for iz in range(4):
                    offset = (
                        torch.tensor([ix, iy, iz], dtype=torch.float64) * 0.25
                    )
                    for b in basis:
                        positions.append(0.25 * b + offset)

        positions = torch.stack(positions)

        # Nearest neighbor distance
        d_nn = 0.5 * np.sqrt(2.0) * a
        cutoff = d_nn + 0.1

        nbl = TorchNeighborList(cutoff=cutoff, device="cpu")
        result = nbl.get_neighbors(positions, cell=cell)

        # Every atom should have exactly 12 neighbors
        assert torch.all(result["num_neighbors"] == 12), (
            f"Not all atoms have 12 neighbors: {result['num_neighbors']}"
        )

    def test_fcc_isolated_cube(self):
        """
        Test isolated FCC structure (2x2x2 cube without PBC).

        Based on test_fcc_isolated() from fortran/test_lclist.f90.
        Interior atoms have 12 neighbors, surface atoms have fewer.
        """
        # FCC conventional cell (a = 5.0 Å), scaled by 2
        a = 5.0
        cell = torch.eye(3, dtype=torch.float64) * a * 2.0

        # FCC basis
        basis = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            dtype=torch.float64,
        )

        # Create 2x2x2 cube (32 atoms total)
        positions = []
        for ix in range(2):
            for iy in range(2):
                for iz in range(2):
                    offset = (
                        torch.tensor([ix, iy, iz], dtype=torch.float64) * 0.5
                    )
                    for b in basis:
                        # Fractional -> Cartesian for isolated system
                        cart_pos = (0.5 * b + offset) @ cell
                        positions.append(cart_pos)

        positions = torch.stack(positions)

        # Nearest neighbor distance
        d_nn = 0.5 * np.sqrt(2.0) * a
        cutoff = d_nn + 0.1

        # No PBC for isolated system
        nbl = TorchNeighborList(cutoff=cutoff, device="cpu")
        result = nbl.get_neighbors(positions)

        # All atoms should have <= 12 neighbors (boundary atoms have fewer)
        assert torch.all(result["num_neighbors"] <= 12), (
            "Some atoms have more than 12 "
            + f"neighbors: {result['num_neighbors']}"
        )

        # At least some atoms should have fewer than 12 (surface atoms)
        assert torch.any(result["num_neighbors"] < 12), (
            "All atoms have 12 neighbors, but some should be on surface"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
