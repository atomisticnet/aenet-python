import unittest
import numpy as np

from ..nblist.neighborlist import NeighborList


class NblistTest(unittest.TestCase):

    def test_fcc_nearest_nb(self):
        avec = np.array([[0.0, 0.5, 0.5],
                         [0.5, 0.0, 0.5],
                         [0.5, 0.5, 0.0]])*1.0
        coo = np.array([[0.0, 0.0, 0.0]])
        nbl = NeighborList(coo, lattice_vectors=avec)
        # number of nearest neighbors is 12 for FCC
        (nn, dist, T) = nbl.get_nearest_neighbors(0)
        self.assertEqual(len(nn), 12)

    def test_fcc_cutoff(self):
        a = 1.5
        # primitive unit cell and fractional coordinates
        avec = np.array([[0.0, 0.5, 0.5],
                         [0.5, 0.0, 0.5],
                         [0.5, 0.5, 0.0]])*a
        coo = np.array([[0.0, 0.0, 0.0]])
        nbl = NeighborList(coo, lattice_vectors=avec, interaction_range=2.0*a)
        (nn1, dist1, _) = nbl.get_neighbors_and_distances(0)

        # conventional unit cell and Cartesian coordinates
        avec = np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]])*a
        coo = np.array([[0.0, 0.0, 0.0],
                        [0.0, 0.75, 0.75],
                        [0.75, 0.0, 0.75],
                        [0.75, 0.75, 0.0]])
        nbl = NeighborList(coo, lattice_vectors=avec, cartesian=True,
                           interaction_range=2.0*a)
        (nn2, dist2, _) = nbl.get_neighbors_and_distances(0)

        self.assertEqual(len(nn1), len(nn2))

        dist1 = np.sort(dist1)
        dist2 = np.sort(dist2)
        diff = np.abs(dist1 - dist2)

        self.assertTrue(np.all(diff < 1.0e-6))


if __name__ == "__main__":
    unittest.main()
