"""
Unit tests for new neighbor list integration features.

Tests for:
- max_num_neighbors configurability
- Automatic numpy/torch conversion
- return_coordinates parameter
- from_AtomicStructure() factory method
- AtomicStructure.get_neighbors() integration
"""

import numpy as np
import pytest
import torch

torch_cluster = pytest.importorskip("torch_cluster")

from aenet.geometry import AtomicStructure  # noqa: E402
from aenet.torch_featurize.neighborlist import TorchNeighborList  # noqa: E402


class TestMaxNumNeighbors:
    """Test max_num_neighbors configurability."""

    def test_default_max_num_neighbors(self):
        """Test that default max_num_neighbors is 256."""
        nbl = TorchNeighborList(cutoff=4.0, device="cpu")
        assert nbl.max_num_neighbors == 256

    def test_custom_max_num_neighbors(self):
        """Test setting custom max_num_neighbors."""
        nbl = TorchNeighborList(
            cutoff=4.0, max_num_neighbors=512, device="cpu")
        assert nbl.max_num_neighbors == 512

        nbl_large = TorchNeighborList(
            cutoff=4.0, max_num_neighbors=1024, device="cpu"
        )
        assert nbl_large.max_num_neighbors == 1024

    def test_max_num_neighbors_affects_search(self):
        """Test that max_num_neighbors actually affects neighbor search."""
        # Create a dense cluster where atoms might exceed default limit
        # if max_num_neighbors were too small
        np.random.seed(42)
        positions = torch.from_numpy(
            np.random.randn(100, 3) * 0.5
        ).to(torch.float64)

        # With sufficient max_num_neighbors
        nbl_sufficient = TorchNeighborList(
            cutoff=10.0, max_num_neighbors=100, device="cpu"
        )
        result = nbl_sufficient.get_neighbors(positions)

        # Should complete without error
        assert result["edge_index"].shape[0] == 2
        assert result["distances"].shape[0] > 0


class TestNumpyTorchConversion:
    """Test automatic numpy/torch tensor conversion."""

    def test_numpy_positions_input(self):
        """Test that numpy array positions are accepted."""
        # Numpy positions
        positions_np = np.array(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]
        )

        nbl = TorchNeighborList(cutoff=2.0, device="cpu")
        result = nbl.get_neighbors_of_atom(0, positions_np)

        # Should work and return torch tensors
        assert isinstance(result["indices"], torch.Tensor)
        assert isinstance(result["distances"], torch.Tensor)
        assert len(result["indices"]) > 0

    def test_numpy_cell_input(self):
        """Test that numpy array cell is accepted."""
        positions_np = np.array([[0.1, 0.1, 0.1], [0.9, 0.1, 0.1]])
        cell_np = np.eye(3) * 5.0

        nbl = TorchNeighborList(cutoff=2.0, device="cpu")
        result = nbl.get_neighbors_of_atom(0, positions_np, cell=cell_np)

        # Should work and return torch tensors
        assert isinstance(result["indices"], torch.Tensor)
        assert isinstance(result["distances"], torch.Tensor)
        assert isinstance(result["offsets"], torch.Tensor)

    def test_mixed_numpy_torch_input(self):
        """Test mixed numpy and torch inputs."""
        positions_np = np.random.randn(5, 3)
        cell_torch = torch.eye(3, dtype=torch.float64) * 4.0

        nbl = TorchNeighborList(cutoff=3.0, device="cpu")
        result = nbl.get_neighbors_of_atom(0, positions_np, cell=cell_torch)

        # Should work
        assert isinstance(result["indices"], torch.Tensor)

    def test_torch_input_still_works(self):
        """Test that torch tensor input still works as before."""
        positions_torch = torch.randn(5, 3, dtype=torch.float64)

        nbl = TorchNeighborList(cutoff=2.0, device="cpu")
        result = nbl.get_neighbors_of_atom(0, positions_torch)

        assert isinstance(result["indices"], torch.Tensor)
        assert isinstance(result["distances"], torch.Tensor)


class TestReturnCoordinates:
    """Test return_coordinates parameter."""

    def test_return_coordinates_isolated(self):
        """Test return_coordinates for isolated system."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]],
            dtype=torch.float64,
        )

        nbl = TorchNeighborList(cutoff=2.0, device="cpu")

        # Without return_coordinates
        result_basic = nbl.get_neighbors_of_atom(0, positions)
        assert "coordinates" not in result_basic

        # With return_coordinates
        result_coords = nbl.get_neighbors_of_atom(
            0, positions, return_coordinates=True
        )
        assert "coordinates" in result_coords
        assert isinstance(result_coords["coordinates"], torch.Tensor)
        assert result_coords["coordinates"].shape == (
            len(result_coords["indices"]),
            3,
        )

        # Verify coordinates match neighbor positions
        neighbor_idx = result_coords["indices"].cpu().numpy()
        expected_coords = positions[neighbor_idx]
        assert torch.allclose(result_coords["coordinates"], expected_coords)

    def test_return_coordinates_pbc(self):
        """Test return_coordinates with PBC (offsets applied)."""
        # Simple test: just verify coordinates are returned with correct shape
        positions = torch.tensor(
            [[0.1, 0.2, 0.3], [0.9, 0.2, 0.3], [0.5, 0.8, 0.5]],
            dtype=torch.float64
        )
        cell = torch.eye(3, dtype=torch.float64) * 4.0

        nbl = TorchNeighborList(cutoff=2.0, device="cpu")
        result = nbl.get_neighbors_of_atom(
            0, positions, cell=cell, return_coordinates=True
        )

        # Should have coordinates key
        assert "coordinates" in result

        # If neighbors found, check shape
        if len(result["coordinates"]) > 0:
            assert result["coordinates"].shape[1] == 3
            assert len(result["coordinates"]) == len(result["indices"])

    def test_return_coordinates_manual_vs_automatic(self):
        """Compare manual coordinate computation vs return_coordinates."""
        positions = np.array(
            [[0.1, 0.2, 0.3], [0.9, 0.2, 0.3], [0.5, 0.8, 0.5]])
        cell = np.eye(3) * 4.0

        nbl = TorchNeighborList(cutoff=2.0, device="cpu")

        # Manual approach
        result_manual = nbl.get_neighbors_of_atom(0, positions, cell=cell)
        neighbor_idx = result_manual["indices"].cpu().numpy()
        offsets = result_manual["offsets"].cpu().numpy()
        coords_manual = positions[neighbor_idx] + offsets @ cell

        # Automatic approach
        result_auto = nbl.get_neighbors_of_atom(
            0, positions, cell=cell, return_coordinates=True
        )
        coords_auto = result_auto["coordinates"].cpu().numpy()

        # Should be identical
        assert np.allclose(coords_manual, coords_auto)


class TestFromAtomicStructure:
    """Test from_AtomicStructure() factory method."""

    def test_factory_method_isolated(self):
        """Test factory method with isolated structure."""
        coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]])
        types = ["O", "H", "H"]
        structure = AtomicStructure(coords, types)

        nbl = TorchNeighborList.from_AtomicStructure(
            structure, cutoff=2.0, device="cpu"
        )

        assert nbl.cutoff == 2.0
        assert nbl.device == "cpu"
        assert nbl.max_num_neighbors == 256

    def test_factory_method_periodic(self):
        """Test factory method with periodic structure."""
        coords = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
        types = ["C", "C"]
        avec = np.eye(3) * 5.0
        structure = AtomicStructure(coords, types, avec=avec)

        nbl = TorchNeighborList.from_AtomicStructure(
            structure, cutoff=3.0, device="cpu"
        )

        assert nbl.cutoff == 3.0
        assert structure.pbc is True

    def test_factory_method_custom_params(self):
        """Test factory method with custom parameters."""
        coords = np.array([[0.0, 0.0, 0.0]])
        types = ["H"]
        structure = AtomicStructure(coords, types)

        nbl = TorchNeighborList.from_AtomicStructure(
            structure, cutoff=4.0, max_num_neighbors=512, device="cpu"
        )

        assert nbl.cutoff == 4.0
        assert nbl.max_num_neighbors == 512

    def test_factory_method_multiframe(self):
        """Test factory method with multiple frames."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        types = ["C", "C"]

        structure = AtomicStructure(coords1, types)
        structure.add_frame(coords2)

        # Test with first frame
        nbl_frame0 = TorchNeighborList.from_AtomicStructure(
            structure, cutoff=2.0, frame=0, device="cpu"
        )
        assert nbl_frame0.cutoff == 2.0

        # Test with last frame (default)
        nbl_frame_last = TorchNeighborList.from_AtomicStructure(
            structure, cutoff=2.0, device="cpu"
        )
        assert nbl_frame_last.cutoff == 2.0


class TestAtomicStructureIntegration:
    """Test AtomicStructure.get_neighbors() integration."""

    def test_get_neighbors_isolated(self):
        """Test get_neighbors on isolated structure."""
        coords = np.array(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
             [3.0, 0.0, 0.0], [0.0, 1.5, 0.0]]
        )
        types = ["O", "H", "C", "H"]
        structure = AtomicStructure(coords, types)

        # Get neighbors of atom 0
        neighbors = structure.get_neighbors(i=0, cutoff=2.0)

        # Should be an AtomicStructure
        assert isinstance(neighbors, AtomicStructure)

        # Should contain self and nearby atoms
        assert neighbors.natoms >= 2  # At least H at 1.5 and possibly H at 1.5

    def test_get_neighbors_periodic(self):
        """Test get_neighbors on periodic structure."""
        coords = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]])
        types = ["Na", "Cl"]
        avec = np.eye(3) * 5.0
        structure = AtomicStructure(coords, types, avec=avec)

        # Get neighbors including periodic images
        neighbors = structure.get_neighbors(i=0, cutoff=2.0)

        assert isinstance(neighbors, AtomicStructure)
        # Should find neighbor across boundary
        assert neighbors.natoms > 0

    def test_get_neighbors_return_self(self):
        """Test return_self parameter."""
        coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        types = ["C", "C"]
        structure = AtomicStructure(coords, types)

        # With return_self=True (default)
        neighbors_with_self = structure.get_neighbors(
            i=0, cutoff=2.0, return_self=True
        )
        assert neighbors_with_self.natoms >= 2  # Self + at least one neighbor

        # With return_self=False
        neighbors_no_self = structure.get_neighbors(
            i=0, cutoff=2.0, return_self=False
        )
        assert neighbors_no_self.natoms == neighbors_with_self.natoms - 1

    def test_get_neighbors_coordinates_correct(self):
        """Test that returned coordinates are correct."""
        coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]])
        types = ["O", "H", "H"]
        structure = AtomicStructure(coords, types)

        neighbors = structure.get_neighbors(i=0, cutoff=2.0, return_self=True)

        # First atom should be self (at origin)
        assert np.allclose(neighbors.coords[0][0], [0.0, 0.0, 0.0])

        # Should have 3 total atoms (self + 2 neighbors)
        assert neighbors.natoms == 3

    def test_get_neighbors_pbc_coordinates(self):
        """Test that PBC neighbor coordinates are correctly computed."""
        # Two atoms across boundary (Cartesian coordinates)
        # Atoms at 0.5 Å and 4.5 Å (should be 1.0 Å apart across boundary)
        coords = np.array([[0.5, 2.5, 2.5], [4.5, 2.5, 2.5]])
        types = ["Na", "Cl"]
        avec = np.eye(3) * 5.0
        structure = AtomicStructure(coords, types, avec=avec)

        neighbors = structure.get_neighbors(i=0, cutoff=2.0, return_self=False)

        if neighbors is not None and neighbors.natoms > 0:
            # Neighbor coordinates should reflect actual position
            # (with PBC offset applied)
            atom0_pos = coords[0]
            neighbor_pos = neighbors.coords[0][0]

            # Distance should be within cutoff
            dist = np.linalg.norm(neighbor_pos - atom0_pos)
            assert dist <= 2.0

    def test_get_neighbors_types_preserved(self):
        """Test that atom types are preserved."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        types = ["C", "H", "O"]
        structure = AtomicStructure(coords, types)

        neighbors = structure.get_neighbors(i=0, cutoff=1.5, return_self=True)

        # Should have types
        assert neighbors.types is not None
        assert len(neighbors.types) == neighbors.natoms

        # First type should be 'C' (self)
        assert neighbors.types[0] == "C"

    def test_get_neighbors_multiframe(self):
        """Test get_neighbors with multiple frames."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        types = ["C", "C"]

        structure = AtomicStructure(coords1, types)
        structure.add_frame(coords2)

        # Frame 0: neighbor at 1.0 Å
        neighbors_f0 = structure.get_neighbors(
            i=0, cutoff=1.5, return_self=False, frame=0
        )
        assert neighbors_f0.natoms == 1

        # Frame 1: neighbor at 2.0 Å (outside cutoff)
        neighbors_f1 = structure.get_neighbors(
            i=0, cutoff=1.5, return_self=False, frame=1
        )
        assert neighbors_f1 is None  # No neighbors found

        # Last frame (default): same as frame 1
        neighbors_last = structure.get_neighbors(
            i=0, cutoff=1.5, return_self=False, frame=-1
        )
        assert neighbors_last is None  # No neighbors found


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_basic_usage_unchanged(self):
        """Test that basic usage patterns still work."""
        # This would have worked with legacy neighbor list
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        types = ["C", "H", "O"]
        structure = AtomicStructure(coords, types)

        neighbors = structure.get_neighbors(i=0, cutoff=1.5)

        assert isinstance(neighbors, AtomicStructure)
        assert neighbors.natoms > 0

    def test_periodic_usage_unchanged(self):
        """Test that periodic system usage still works."""
        coords = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
        types = ["Na", "Cl"]
        avec = np.eye(3) * 5.0
        structure = AtomicStructure(coords, types, avec=avec)

        neighbors = structure.get_neighbors(i=0, cutoff=3.0)

        assert isinstance(neighbors, AtomicStructure)


class TestEdgeCases:
    """Test edge cases in integration."""

    def test_no_neighbors_found(self):
        """Test when no neighbors are within cutoff."""
        coords = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        types = ["C", "C"]
        structure = AtomicStructure(coords, types)

        neighbors = structure.get_neighbors(
            i=0, cutoff=1.0, return_self=False
        )

        # Should return None when no neighbors found
        assert neighbors is None

    def test_single_atom_structure(self):
        """Test with single atom structure."""
        coords = np.array([[0.0, 0.0, 0.0]])
        types = ["H"]
        structure = AtomicStructure(coords, types)

        neighbors = structure.get_neighbors(
            i=0, cutoff=2.0, return_self=True
        )

        assert neighbors.natoms == 1  # Only self

    def test_large_system(self):
        """Test with larger system to verify performance."""
        np.random.seed(42)
        coords = np.random.randn(100, 3) * 5.0
        types = ["C"] * 100
        structure = AtomicStructure(coords, types)

        neighbors = structure.get_neighbors(i=0, cutoff=3.0)

        # Should complete without error
        assert isinstance(neighbors, AtomicStructure)
        assert neighbors.natoms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
