"""
Tests for per-atom neighbor access and type-dependent cutoffs.
"""

import pytest
import torch

torch_cluster = pytest.importorskip("torch_cluster")

from aenet.torch_featurize.neighborlist import TorchNeighborList  # noqa: E402


class TestPerAtomNeighbors:
    """Test per-atom neighbor access methods."""

    def test_get_neighbors_of_atom_basic(self):
        """Test basic per-atom neighbor query."""
        # Water molecule
        positions = torch.tensor(
            [
                [0.00000, 0.00000, 0.11779],  # O
                [0.00000, 0.75545, -0.47116],  # H
                [0.00000, -0.75545, -0.47116],  # H
            ],
            dtype=torch.float64,
        )

        nbl = TorchNeighborList(cutoff=2.0, device="cpu")

        # Get neighbors of oxygen
        result = nbl.get_neighbors_of_atom(0, positions)

        assert "indices" in result
        assert "distances" in result
        assert "offsets" in result
        assert result["offsets"] is None  # No PBC

        # Oxygen should have 2 neighbors (both H)
        assert len(result["indices"]) == 2
        assert torch.all(torch.isin(result["indices"], torch.tensor([1, 2])))

    def test_get_neighbors_by_atom(self):
        """Test getting neighbors for all atoms."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        )

        nbl = TorchNeighborList(cutoff=1.5, device="cpu")
        results = nbl.get_neighbors_by_atom(positions)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert "indices" in result
            assert "distances" in result
            assert "offsets" in result


class TestTypeDependentCutoffs:
    """Test type-dependent cutoff filtering."""

    def test_validation_missing_type(self):
        """Test that validation catches missing types."""
        atom_types = torch.tensor([8, 1, 1])
        cutoff_dict = {
            (1, 1): 2.0,
            (1, 8): 2.5,
            (8, 6): 3.0,  # Type 6 not in atom_types!
        }

        with pytest.raises(ValueError, match="Type 6.*not found"):
            TorchNeighborList(
                cutoff=5.0, atom_types=atom_types, cutoff_dict=cutoff_dict
            )

    def test_validation_cutoff_exceeds_max(self):
        """Test that validation catches cutoffs exceeding maximum."""
        atom_types = torch.tensor([8, 1, 1])
        cutoff_dict = {
            (1, 1): 2.0,
            (1, 8): 6.0,  # Exceeds max cutoff of 5.0!
        }

        with pytest.raises(ValueError, match="exceeds maximum cutoff"):
            TorchNeighborList(
                cutoff=5.0, atom_types=atom_types, cutoff_dict=cutoff_dict
            )

    def test_type_filtering_water(self):
        """Test type-dependent filtering on water molecule."""
        # Water molecule
        positions = torch.tensor(
            [
                [0.00000, 0.00000, 0.11779],  # O
                [0.00000, 0.75545, -0.47116],  # H1
                [0.00000, -0.75545, -0.47116],  # H2
            ],
            dtype=torch.float64,
        )

        atom_types = torch.tensor([8, 1, 1])  # O, H, H

        # Set different cutoffs for different pairs
        cutoff_dict = {
            (1, 1): 1.0,  # H-H: short cutoff
            (1, 8): 2.5,  # O-H: medium cutoff
            (8, 8): 3.0,  # O-O: long cutoff
        }

        nbl = TorchNeighborList(
            cutoff=5.0,  # Max cutoff
            atom_types=atom_types,
            cutoff_dict=cutoff_dict,
            device="cpu",
        )

        # Get O neighbors with type filtering
        result_o = nbl.get_neighbors_of_atom(0, positions)

        # O should see both H atoms (O-H distance ~0.958 Å < 2.5 Å)
        assert len(result_o["indices"]) == 2

        # Get H1 neighbors with type filtering
        result_h1 = nbl.get_neighbors_of_atom(1, positions)

        # H1 should see O (distance ~0.958 Å < 2.5 Å)
        # H1 should NOT see H2 (distance ~1.511 Å > 1.0 Å)
        assert len(result_h1["indices"]) == 1
        assert result_h1["indices"][0] == 0  # Should only see O

    def test_type_filtering_override(self):
        """Test that cutoff_dict can be overridden per call."""
        positions = torch.tensor(
            [
                [0.00000, 0.00000, 0.11779],  # O
                [0.00000, 0.75545, -0.47116],  # H
                [0.00000, -0.75545, -0.47116],  # H
            ],
            dtype=torch.float64,
        )

        atom_types = torch.tensor([8, 1, 1])

        # Store restrictive cutoffs
        cutoff_dict_restrictive = {
            (1, 1): 0.5,  # Very short
            (1, 8): 0.5,  # Very short
            (8, 8): 0.5,  # Very short
        }

        nbl = TorchNeighborList(
            cutoff=5.0,
            atom_types=atom_types,
            cutoff_dict=cutoff_dict_restrictive,
            device="cpu",
        )

        # With restrictive cutoffs, O should have no neighbors
        result_restrictive = nbl.get_neighbors_of_atom(0, positions)
        assert len(result_restrictive["indices"]) == 0

        # Override with permissive cutoffs
        cutoff_dict_permissive = {
            (1, 1): 3.0,
            (1, 8): 3.0,
            (8, 8): 3.0,
        }

        result_permissive = nbl.get_neighbors_of_atom(
            0, positions, cutoff_dict=cutoff_dict_permissive
        )

        # Now O should see both H atoms
        assert len(result_permissive["indices"]) == 2


class TestCaching:
    """Test caching mechanism."""

    def test_cache_reuse(self):
        """Test that cache is reused for same positions."""
        positions = torch.randn(10, 3, dtype=torch.float64)

        nbl = TorchNeighborList(cutoff=3.0, device="cpu")

        # First call - computes
        result1 = nbl.get_neighbors_of_atom(0, positions)

        # Second call - should use cache
        result2 = nbl.get_neighbors_of_atom(1, positions)

        # Results should be consistent
        assert result1["indices"].device == result2["indices"].device

    def test_cache_invalidation(self):
        """Test that cache is invalidated for different positions."""
        positions1 = torch.randn(10, 3, dtype=torch.float64)
        positions2 = torch.randn(10, 3, dtype=torch.float64)

        nbl = TorchNeighborList(cutoff=3.0, device="cpu")

        # Call with first positions
        result1 = nbl.get_neighbors_of_atom(0, positions1)

        # Call with different positions - should recompute
        result2 = nbl.get_neighbors_of_atom(0, positions2)

        # Results should be different
        assert not torch.equal(result1["indices"], result2["indices"])


class TestPerAtomPBC:
    """Test per-atom access with periodic boundary conditions."""

    def test_per_atom_pbc(self):
        """Test per-atom neighbors with PBC."""
        # Two atoms in cubic cell
        positions = torch.tensor(
            [[0.1, 0.1, 0.1], [0.9, 0.1, 0.1]], dtype=torch.float64
        )

        cell = torch.eye(3, dtype=torch.float64) * 5.0

        nbl = TorchNeighborList(cutoff=2.0, device="cpu")
        result = nbl.get_neighbors_of_atom(0, positions, cell=cell)

        # Should find neighbor across boundary
        assert len(result["indices"]) > 0
        assert result["offsets"] is not None

    def test_type_filtering_with_pbc(self):
        """Test type filtering combined with PBC."""
        positions = torch.tensor(
            [[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=torch.float64
        )

        cell = torch.eye(3, dtype=torch.float64) * 5.0
        atom_types = torch.tensor([8, 1])

        cutoff_dict = {
            (1, 1): 1.0,
            (1, 8): 2.0,
            (8, 8): 3.0,
        }

        nbl = TorchNeighborList(
            cutoff=5.0,
            atom_types=atom_types,
            cutoff_dict=cutoff_dict,
            device="cpu",
        )

        result = nbl.get_neighbors_of_atom(0, positions, cell=cell)

        # Check that result contains the expected keys
        assert "indices" in result
        assert "distances" in result
        assert "offsets" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
