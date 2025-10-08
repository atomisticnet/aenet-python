"""
Tests for the XSF parser

"""

__author__ = "Alexander Urban, Nongnuch Artrith"
__email__ = "alexurba@mit.edu, nartrith@mit.edu"
__date__ = "2014-10-13"
__version__ = "0.1"

import unittest
import os

from ...io import structure

fixtures = os.path.join(os.path.dirname(__file__), 'fixtures')
periodic_xsf = os.path.join(fixtures, 'periodic.xsf')
isolated_xsf = os.path.join(fixtures, 'isolated.xsf')
trajectory_xsf = os.path.join(fixtures, 'trajectory.xsf')


class XSFTest(unittest.TestCase):

    def test_isolated(self):
        struc = structure.read(isolated_xsf)
        self.assertEqual(struc.natoms, 36)
        self.assertEqual(struc.ntypes, 3)
        self.assertFalse(struc.pbc)
        c = struc.composition
        self.assertEqual(c['Li'], 12)
        self.assertEqual(c['Mo'], 6)
        self.assertEqual(c['O'],  18)
        structure.write(struc, filename='TEST.xsf')
        struc2 = structure.read('TEST.xsf')
        self.assertEqual(struc, struc2)
        os.unlink('TEST.xsf')

    def test_periodic(self):
        struc = structure.read(periodic_xsf)
        self.assertEqual(struc.natoms, 36)
        self.assertEqual(struc.ntypes, 3)
        self.assertTrue(struc.pbc)
        c = struc.composition
        self.assertEqual(c['Li'], 12)
        self.assertEqual(c['Mo'], 6)
        self.assertEqual(c['O'],  18)
        structure.write(struc, filename='TEST.xsf')
        struc2 = structure.read('TEST.xsf')
        self.assertEqual(struc, struc2)
        os.unlink('TEST.xsf')

    def test_trajectory(self):
        struc = structure.read(trajectory_xsf)
        self.assertEqual(struc.natoms, 14)
        self.assertEqual(struc.ntypes, 3)
        self.assertTrue(struc.pbc)
        c = struc.composition
        self.assertEqual(c['Li'], 2)
        self.assertEqual(c['Cr'], 4)
        self.assertEqual(c['O'],  8)


if __name__ == "__main__":
    unittest.main()
