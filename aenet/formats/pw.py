"""
Read pw.x output.

"""

import re
import numpy as np

from ..geometry import AtomicStructure
from .. import units
from .parser_abc import ParserABC

__author__ = "Nongnuch Artrith"
__date__ = "2014-04-02"


class PWParser(ParserABC):
    def __init__(self):
        self.name = 'pw'
        self.description = 'pw.x output format (QE)'
        self.extensions = []
        self.default_file_names = ['pw.out']

    def read(self, infile, **kwargs):
        """
        Read trajectory in the `cp.pos' format of Quantum Espresso's `cp.x'.

        Arguments:
          infile     name of the input file (pw.x output format)

        Returns:
          an instance of the AtomicStructure class
        """

        self._check_amend_args(**kwargs)

        re_alat = re.compile(r"^ *lattice parameter")
        re_avec = re.compile(r"^ *crystal axes: ")
        re_atoms = re.compile(r"^ *site n. *atom *positions")
        re_energy = re.compile(r"^!")
        re_forces = re.compile(r"^ *Forces acting on atoms")
        re_end = re.compile(r"^ *JOB DONE.")
        re_avec_up = re.compile(r"^CELL_PARAMETERS \(alat= *([0-9.]*)")
        re_atoms_up = re.compile(r"^ATOMIC_POSITIONS")
        # re_bfgs_step = re.compile(r"^ *BFGS Geometry Optimization")
        re_bfgs_step = re.compile(r"^ *number of bfgs steps")

        energy = 0.0
        avec = None
        coords = None
        forces = None
        types = None
        fractional = False
        struc = None

        with open(infile, 'r') as f:
            while True:
                line = f.readline()
                if re_end.search(line):
                    if struc is None:
                        struc = AtomicStructure(
                            coords.copy(), types, avec=avec,
                            fractional=fractional, energy=energy,
                            forces=forces)
                    elif energy is not None:
                        struc.add_frame(
                            coords.copy(), avec=avec, energy=energy,
                            forces=forces, fractional=fractional)
                    break
                elif re_bfgs_step.search(line):
                    if struc is None:
                        struc = AtomicStructure(
                            coords.copy(), types, avec=avec,
                            fractional=fractional, energy=energy,
                            forces=forces)
                    else:
                        struc.add_frame(
                            coords.copy(), avec=avec, energy=energy,
                            forces=forces, fractional=fractional)
                elif re_alat.search(line):
                    # lattice constant/scaling factor in Bohr
                    alat = float(line.split()[4])*units.Bohr2Ang
                elif re_avec.search(line):
                    # initial lattice vectors
                    avec = []
                    for i in range(3):
                        line = f.readline()
                        avec.append([float(el) for el in line.split()[3:6]])
                    avec = alat*np.array(avec)
                elif re_atoms.search(line):
                    # initial coordinates
                    coords = []
                    types = []
                    line = f.readline()
                    while (len(line.strip()) > 0):
                        coo = [float(el) for el in line.split()[6:9]]
                        coords.append(coo)
                        types.append(line.split()[1])
                        line = f.readline()
                    fractional = False
                    coords = np.array(coords)*alat
                    forces = np.zeros(np.shape(coords))
                elif re_energy.search(line):
                    # final total energy in Rydberg
                    energy = float(line.split()[4])*units.Ry2eV
                elif re_forces.search(line):
                    i = 0
                    while i < len(forces):
                        line = f.readline()
                        match = re.search(
                            "^ *atom.* force =  *([0-9. -]*)$", line)
                        if match:
                            forces[i][:] = [
                                float(el) for el in match.group(1).split()]
                            forces[i] *= units.Ry2eV/units.Bohr2Ang
                            i += 1
                elif re_avec_up.search(line):
                    m = re_avec_up.search(line)
                    alat = float(m.group(1))*units.Bohr2Ang
                    for v in avec:
                        line = f.readline()
                        v[:] = [float(el) for el in line.split()]
                    avec = alat*np.array(avec)
                elif re_atoms_up.search(line):
                    for coo in coords:
                        line = f.readline()
                        coo[:] = [float(el) for el in line.split()[1:4]]
                    fractional = True
                    energy = None  # discard previous energy

        self._amend(struc, **kwargs)
        return struc
