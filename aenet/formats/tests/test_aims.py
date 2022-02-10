import unittest
import tempfile
import os

from ...io import structure

fixtures = os.path.join(os.path.dirname(__file__), 'fixtures')
isolated = os.path.join(fixtures, 'geometry.in.iso')
periodic = os.path.join(fixtures, 'geometry.in.pbc')
output = os.path.join(fixtures, 'fhiaims-01.out')


class AimsTest(unittest.TestCase):

    def test_isolated(self):
        struc = structure.read(isolated, frmt='aims')
        self.assertEqual(struc.natoms, 36)
        self.assertEqual(struc.ntypes, 3)
        self.assertFalse(struc.pbc)
        c = struc.composition
        self.assertEqual(c['Li'], 12)
        self.assertEqual(c['Mo'], 6)
        self.assertEqual(c['O'],  18)
        with tempfile.TemporaryDirectory() as d:
            tmp = os.path.join(d, 'geometry.in')
            structure.write(struc, filename=tmp)
            struc2 = structure.read(tmp)
            self.assertEqual(struc, struc2)

    def test_periodic(self):
        struc = structure.read(periodic, frmt='aims')
        self.assertEqual(struc.natoms, 52)
        self.assertEqual(struc.ntypes, 2)
        self.assertTrue(struc.pbc)
        c = struc.composition
        self.assertEqual(c['Au'], 38)
        self.assertEqual(c['Cu'], 14)
        with tempfile.TemporaryDirectory() as d:
            tmp = os.path.join(d, 'geometry.in')
            structure.write(struc, filename=tmp)
            struc2 = structure.read(tmp)
            self.assertEqual(struc, struc2)

    def test_output(self):
        struc = structure.read(output, frmt='aimsout')
        self.assertEqual(struc.natoms, 21)
        self.assertEqual(struc.ntypes, 3)
        self.assertEqual(struc.nframes, 37)
        self.assertFalse(struc.pbc)


if __name__ == "__main__":
    unittest.main()
