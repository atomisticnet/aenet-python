"""
Unit tests for Python NeighborList (src/aenet/nblist).

Tests the pure Python neighbor list implementation and compares
with ASE neighbor list for validation.
"""

import unittest
import numpy as np

from aenet.nblist import NeighborList


# Check if ASE is available for comparison
try:
    import ase
    from ase.neighborlist import NeighborList as ASENeighborList
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


class TestNeighborListIsolated(unittest.TestCase):
    """Test neighbor list for isolated systems (molecules/clusters)."""

    def test_simple_dimer(self):
        """Test simplest case: two atoms."""
        # Two atoms 2.0 Å apart
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0]
        ])

        # Create neighbor list with cutoff 3.0 Å
        nl = NeighborList(coords, cartesian=True, interaction_range=3.0)

        # Each atom should see the other
        for i in range(2):
            nbl, dist, Tvecs = nl.get_neighbors_and_distances(
                i, r=3.0, return_self=False
            )

            self.assertEqual(len(nbl), 1, f"Atom {i} should have 1 neighbor")
            self.assertAlmostEqual(dist[0], 2.0, places=6)

            # No periodic images for isolated system
            expected_T = np.array([0, 0, 0])
            np.testing.assert_array_equal(Tvecs[0], expected_T)

    def test_triangle(self):
        """Test equilateral triangle."""
        # 3 atoms in equilateral triangle with side length 2.0 Å
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, np.sqrt(3.0), 0.0]
        ])

        nl = NeighborList(coords, cartesian=True, interaction_range=2.5)

        # Each atom should see 2 neighbors
        for i in range(3):
            nbl, dist, Tvecs = nl.get_neighbors_and_distances(
                i, r=2.5, return_self=False
            )
            self.assertEqual(len(nbl), 2,
                             f"Atom {i} should have 2 neighbors")

            # Distances should be approximately 2.0 Å
            for d in dist:
                self.assertAlmostEqual(d, 2.0, places=5)

    def test_return_coords(self):
        """Test that return_coords works correctly."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0]
        ])

        nl = NeighborList(coords, cartesian=True, interaction_range=3.0)

        # Get neighbors with coordinates
        nbl, coords_out, dist, Tvecs = nl.get_neighbors_and_distances(
            0, r=3.0, return_coords=True, return_self=False
        )

        self.assertEqual(len(nbl), 1)
        self.assertEqual(len(coords_out), 1)

        # Neighbor coordinate should be [2.0, 0.0, 0.0]
        expected_coord = np.array([2.0, 0.0, 0.0])
        np.testing.assert_allclose(coords_out[0], expected_coord, atol=1e-10)

    def test_return_self(self):
        """Test return_self parameter."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0]
        ])

        nl = NeighborList(coords, cartesian=True, interaction_range=3.0)

        # With return_self=True
        nbl_with, dist_with, Tvecs_with = nl.get_neighbors_and_distances(
            0, r=3.0, return_self=True
        )

        # Should include self with distance 0
        self.assertIn(0, nbl_with)
        self.assertIn(0.0, dist_with)

        # Without return_self
        nbl_without, dist_without, Tvecs_without = \
            nl.get_neighbors_and_distances(
                0, r=3.0, return_self=False
            )

        # Should not include self
        self.assertNotIn(0, nbl_without)
        self.assertNotIn(0.0, dist_without)

    @unittest.skipIf(not ASE_AVAILABLE, "ASE not available")
    def test_compare_with_ase_isolated(self):
        """Compare with ASE neighbor list for isolated system."""
        # Random cluster of 10 atoms
        np.random.seed(42)
        coords = np.random.rand(10, 3) * 10.0

        cutoff = 3.0

        # Python neighbor list
        nl_py = NeighborList(coords, cartesian=True,
                             interaction_range=cutoff)

        # ASE neighbor list
        cell = np.eye(3) * 20.0  # Large box
        atoms = ase.Atoms(
            symbols=['H'] * 10,
            positions=coords,
            cell=cell,
            pbc=False
        )
        cutoffs_ase = [cutoff / 2.0] * 10
        nl_ase = ASENeighborList(
            cutoffs_ase, skin=0.0, sorted=False,
            self_interaction=False, bothways=True
        )
        nl_ase.update(atoms)

        # Compare for each atom
        for i in range(10):
            # Python NL
            nbl_py, dist_py, _ = nl_py.get_neighbors_and_distances(
                i, r=cutoff, return_self=False
            )
            nbl_py_set = set(nbl_py)

            # ASE NL
            indices_ase, offsets_ase = nl_ase.get_neighbors(i)
            nbl_ase_set = set(indices_ase)

            # Should find same neighbors
            self.assertEqual(
                nbl_py_set, nbl_ase_set,
                f"Atom {i}: Python and ASE neighbor lists differ\n"
                f"  Python: {sorted(nbl_py_set)}\n"
                f"  ASE:    {sorted(nbl_ase_set)}"
            )


class TestNeighborListPeriodic(unittest.TestCase):
    """Test neighbor list with periodic boundary conditions."""

    def test_simple_cubic(self):
        """Test simple cubic lattice."""
        # Single atom in cubic cell
        coords = np.array([[0.5, 0.5, 0.5]])
        avec = np.eye(3) * 5.0

        # Cutoff to reach nearest neighbors through PBC
        cutoff = 3.0

        nl = NeighborList(
            coords, lattice_vectors=avec,
            cartesian=False, interaction_range=cutoff
        )

        # Get neighbors (should find periodic images)
        nbl, dist, Tvecs = nl.get_neighbors_and_distances(
            0, r=cutoff, return_self=False
        )

        # Should find neighbors through PBC
        self.assertGreater(len(nbl), 0,
                           "Should find neighbors through PBC")

    def test_two_atom_pbc(self):
        """Test two atoms with PBC."""
        # Two atoms at opposite ends of cell
        coords = np.array([
            [0.1, 0.5, 0.5],
            [0.9, 0.5, 0.5]
        ])
        avec = np.eye(3) * 5.0

        cutoff = 2.0

        nl = NeighborList(
            coords, lattice_vectors=avec,
            cartesian=False, interaction_range=cutoff
        )

        # Atoms should see each other through PBC
        # Distance through boundary: 0.2 * 5.0 = 1.0 Å
        nbl, dist, Tvecs = nl.get_neighbors_and_distances(
            0, r=cutoff, return_self=False
        )

        self.assertIn(1, nbl, "Atom 0 should see atom 1 through PBC")

        # Find the distance for atom 1
        idx = nbl.index(1)
        self.assertAlmostEqual(dist[idx], 1.0, places=5,
                               msg="Distance through PBC incorrect")


class TestNeighborListEdgeCases(unittest.TestCase):
    """Test edge cases and special situations."""

    def test_single_atom(self):
        """Test with single atom."""
        coords = np.array([[0.0, 0.0, 0.0]])

        nl = NeighborList(coords, cartesian=True, interaction_range=5.0)

        # Should have no neighbors (except possibly self)
        nbl, dist, Tvecs = nl.get_neighbors_and_distances(
            0, r=5.0, return_self=False
        )

        self.assertEqual(len(nbl), 0, "Single atom should have no neighbors")

    def test_no_neighbors_within_cutoff(self):
        """Test atoms far apart."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0]
        ])

        nl = NeighborList(coords, cartesian=True, interaction_range=2.0)

        nbl, dist, Tvecs = nl.get_neighbors_and_distances(
            0, r=2.0, return_self=False
        )

        self.assertEqual(len(nbl), 0,
                         "Should find no neighbors beyond cutoff")

    def test_large_cluster(self):
        """Test with larger cluster."""
        # FCC-like structure
        np.random.seed(123)
        coords = np.random.rand(50, 3) * 10.0

        nl = NeighborList(coords, cartesian=True, interaction_range=3.0)

        # Check that all atoms can be queried
        for i in range(50):
            nbl, dist, Tvecs = nl.get_neighbors_and_distances(
                i, r=3.0, return_self=False
            )
            # Just check it completes without error
            self.assertIsInstance(nbl, list)


class TestNeighborListCoordinateHandling(unittest.TestCase):
    """Test coordinate transformations and handling."""

    def test_fractional_to_cartesian(self):
        """Test fractional coordinate conversion."""
        avec = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ])

        # Fractional coordinates
        frac_coords = np.array([[0.5, 0.5, 0.5]])

        nl = NeighborList(
            frac_coords, lattice_vectors=avec,
            cartesian=False, interaction_range=3.0
        )

        # Convert to Cartesian
        cart_coords = nl.frac2cart(frac_coords)

        expected = np.array([[2.5, 2.5, 2.5]])
        np.testing.assert_allclose(cart_coords, expected, atol=1e-10)

    def test_cartesian_to_fractional(self):
        """Test Cartesian coordinate conversion."""
        avec = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ])

        # Cartesian coordinates
        cart_coords = np.array([[2.5, 2.5, 2.5]])

        nl = NeighborList(
            cart_coords, lattice_vectors=avec,
            cartesian=True, interaction_range=3.0
        )

        # Convert to fractional
        frac_coords = nl.cart2frac(cart_coords)

        expected = np.array([[0.5, 0.5, 0.5]])
        np.testing.assert_allclose(frac_coords, expected, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
