import unittest
import tempfile
import os
import bz2

from ...io import structure

fixtures = os.path.join(os.path.dirname(__file__), 'fixtures')
periodic = os.path.join(fixtures, 'POSCAR')
output = os.path.join(fixtures, 'vasprun.xml.bz2')


class AimsTest(unittest.TestCase):

    def test_periodic(self):
        struc = structure.read(periodic)
        self.assertEqual(struc.ntypes, 3)
        c = struc.composition
        self.assertEqual(c['Li'], 22)
        self.assertEqual(c['Mo'], 12)
        self.assertEqual(c['O'], 36)
        with tempfile.TemporaryDirectory() as d:
            tmp = os.path.join(d, 'POSCAR')
            structure.write(struc, filename=tmp)
            struc2 = structure.read(tmp)
            self.assertEqual(struc.composition, struc2.composition)

    def test_output(self):
        with bz2.open(output, mode='r') as fp:
            struc = structure.read(fp, frmt='vasprun')
            self.assertEqual(struc.natoms, 36)
            self.assertEqual(struc.ntypes, 3)
            self.assertEqual(struc.nframes, 3)


if __name__ == "__main__":
    unittest.main()
