import pytest
import torch
from aenet.torch_nblist.neighborlist import TorchNeighborList


class TestCellListValidation:
    """
    Compare the new cell list implementation against the legacy
    full-replication method.
    """

    @pytest.fixture
    def nbl(self):
        return TorchNeighborList(cutoff=4.0, device="cpu", dtype=torch.float64)

    def test_cubic_cell_consistency(self, nbl):
        """Test consistency for a simple cubic cell."""
        # Create a simple cubic system
        cell = torch.eye(3, dtype=torch.float64) * 10.0

        # Random positions
        torch.manual_seed(42)
        positions = torch.rand(50, 3, dtype=torch.float64) * 10.0

        # Get neighbors using new method (default)
        res_new = nbl.get_neighbors_pbc(positions, cell, fractional=False)
        edge_index_new, dists_new, offsets_new = res_new

        # Get neighbors using old method
        res_old = nbl._get_neighbors_pbc_old(positions, cell, fractional=False)
        edge_index_old, dists_old, offsets_old = res_old

        # Compare number of edges
        assert edge_index_new.shape[1] == edge_index_old.shape[1]

        # Sort edges to compare (order might differ)
        # Sort by source, then target
        if edge_index_new.shape[1] > 0:
            # Create sort keys
            def get_sort_indices(edge_index, dists):
                # Sort by source, then target, then distance
                # (to handle degenerate cases)
                # Simple sort by source then target
                # Multiply source by large number to prioritize
                sort_key = edge_index[0] * 1000000 + edge_index[1]
                return torch.argsort(sort_key)

            idx_new = get_sort_indices(edge_index_new, dists_new)
            idx_old = get_sort_indices(edge_index_old, dists_old)

            # Compare sorted results
            assert torch.equal(
                edge_index_new[:, idx_new], edge_index_old[:, idx_old])
            assert torch.allclose(dists_new[idx_new], dists_old[idx_old])

            # Offsets might differ by periodic shifts if atoms are exactly
            # on boundaries but for random positions inside, they should match
            assert torch.equal(offsets_new[idx_new], offsets_old[idx_old])

    def test_triclinic_cell_consistency(self, nbl):
        """Test consistency for a triclinic cell."""
        # Triclinic cell
        cell = torch.tensor([
            [10.0, 0.0, 0.0],
            [2.0, 10.0, 0.0],
            [1.0, 2.0, 10.0]
        ], dtype=torch.float64)

        # Random fractional positions
        torch.manual_seed(123)
        frac_positions = torch.rand(50, 3, dtype=torch.float64)

        # Get neighbors using new method (fractional input)
        res_new = nbl.get_neighbors_pbc(frac_positions, cell, fractional=True)
        edge_index_new, dists_new, offsets_new = res_new

        # Get neighbors using old method
        res_old = nbl._get_neighbors_pbc_old(
            frac_positions, cell, fractional=True)
        edge_index_old, dists_old, offsets_old = res_old

        # Compare number of edges
        assert edge_index_new.shape[1] == edge_index_old.shape[1]

        if edge_index_new.shape[1] > 0:
            # Sort and compare
            sort_key_new = edge_index_new[0] * 1000000 + edge_index_new[1]
            idx_new = torch.argsort(sort_key_new)

            sort_key_old = edge_index_old[0] * 1000000 + edge_index_old[1]
            idx_old = torch.argsort(sort_key_old)

            assert torch.equal(
                edge_index_new[:, idx_new], edge_index_old[:, idx_old])
            assert torch.allclose(dists_new[idx_new], dists_old[idx_old])

    def test_gradient_consistency(self, nbl):
        """Verify that gradients flow correctly and match."""
        cell = torch.eye(3, dtype=torch.float64) * 8.0
        positions = torch.rand(20, 3, dtype=torch.float64) * 8.0
        positions.requires_grad = True

        # New method gradient
        positions.grad = None
        res_new = nbl.get_neighbors_pbc(positions, cell, fractional=False)
        loss_new = res_new[1].sum()  # Sum of distances
        loss_new.backward()
        grad_new = positions.grad.clone()

        # Old method gradient
        positions.grad = None
        res_old = nbl._get_neighbors_pbc_old(positions, cell, fractional=False)
        loss_old = res_old[1].sum()
        loss_old.backward()
        grad_old = positions.grad.clone()

        assert torch.allclose(grad_new, grad_old)

    def test_fractional_vs_cartesian(self, nbl):
        """Test that fractional and Cartesian inputs yield same results."""
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        frac_pos = torch.rand(30, 3, dtype=torch.float64)
        cart_pos = frac_pos @ cell

        # Fractional input
        res_frac = nbl.get_neighbors_pbc(frac_pos, cell, fractional=True)

        # Cartesian input
        res_cart = nbl.get_neighbors_pbc(cart_pos, cell, fractional=False)

        # Should be identical
        assert res_frac[0].shape == res_cart[0].shape

        # Sort and compare
        idx_frac = torch.argsort(res_frac[0][0] * 1000000 + res_frac[0][1])
        idx_cart = torch.argsort(res_cart[0][0] * 1000000 + res_cart[0][1])

        assert torch.equal(res_frac[0][:, idx_frac], res_cart[0][:, idx_cart])
        assert torch.allclose(res_frac[1][idx_frac], res_cart[1][idx_cart])
