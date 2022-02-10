#!/usr/bin/env python

"""
Read and write files in ASE's traj format.

"""

import sys
import numpy as np

try:
    import ase.io
    import ase.calculators
    import ase.io.trajectory
    import ase.constraints
    has_ase = True
except ImportError:
    has_ase = False

from ..geometry import AtomicStructure
from .parser_abc import ParserABC

__author__ = "Alexander Urban"
__date__ = "2019-08-18"


class AseParser(ParserABC):
    def __init__(self):
        self.name = 'ase'
        self.description = "ASE's trajectory (traj) format"
        self.extensions = ['traj']
        self.default_file_names = []

    def read(self, infile, **kwargs):
        """
        Read atomic structure file in the ASE 'traj' format.  This routine
        relies on ASE's own parser, so ASE has to be installed.

        Arguments:
          infile   name of the input file

        Rerturns:
          instance of the AtomicStructure class

        """

        if not has_ase:
            raise RuntimeError(
                "ASE not found.  Please make sure ASE is installed.")

        self._check_amend_args(**kwargs)

        trajec = ase.io.read(infile, index=":")

        types = list(trajec[0].symbols)

        # handle constrained coordinates
        fixed = np.zeros(trajec[0].positions.shape, dtype=bool)
        for c in trajec[0].constraints:
            if isinstance(c, ase.constraints.FixAtoms):
                fixed[c.get_indices()] = True
            elif isinstance(c, ase.constraints.FixScaled
                            ) and np.all(c.cell == np.array(trajec[0].cell)):
                fixed[c.a] = c.mask
            else:
                raise Warning("Unsupported constraint of type "
                              "{} ignored".format(type(c)))
        if not np.any(fixed):
            fixed = None

        pbc = all(trajec[0].pbc)
        if not pbc:
            avec = None

        for i, atoms in enumerate(trajec):
            coords = atoms.arrays["positions"]

            if pbc:
                avec = atoms.cell.array

            try:
                energy = atoms.get_potential_energy()
            except RuntimeError:
                energy = None

            try:
                forces = atoms.get_forces()
            except (RuntimeError, ValueError):
                forces = None

            if i == 0:
                struc = AtomicStructure(coords, types=types, avec=avec,
                                        fractional=False, energy=energy,
                                        forces=forces, fixed=fixed)
            else:
                struc.add_frame(coords, avec=avec, fractional=False,
                                energy=energy, forces=forces)

        self._amend(struc, **kwargs)
        return struc

    def write(self, struc, outfile=None, frame=None, **kwargs):
        """
        Write atomic structure to file in ASE's trajectory format.  Relies
        on ASE.

        Arguments:
          struc       instance of the AtomicStructure class
          outfile     name of the output file
          frame       number of frame to write out; if None, all frames
                      will be written to a trajectory file

        """

        if not has_ase:
            raise RuntimeError(
                "ASE not found.  Please make sure ASE is installed.")

        if outfile is None:
            raise RuntimeError(
                "ASE trajectories are binary and can only be written to "
                "files, not to stdout.")

        for kw in kwargs:
            sys.stderr.write("Warning: unsupported argument: {}\n".format(kw))

        symbols = struc.types
        pbc = struc.pbc
        if not pbc:
            cell = None

        trajec = []
        if frame is None:
            frames = range(struc.nframes)
        else:
            frames = [frame]
        for iframe in frames:
            positions = struc.coords[iframe]
            if pbc:
                cell = struc.avec[iframe]
            # in ASE some constraints are defined in terms of the lattice
            # vectors, so unfortunately we have to set up the constraints
            # again for each MD frame
            if struc.fixed is not None and np.any(struc.fixed):
                constraints = []
                fixed_atoms = []
                for i, mask in enumerate(struc.fixed):
                    if np.all(mask):
                        fixed_atoms.append(i)
                    else:
                        c = ase.constraints.FixScaled(
                                            struc.avec[iframe], i, mask)
                        constraints.append(c)
                if len(fixed_atoms) > 0:
                    constraints.append(ase.constraints.FixAtoms(fixed_atoms))
            else:
                constraints = None
            atoms = ase.Atoms(symbols=symbols, positions=positions, pbc=pbc,
                              cell=cell, constraint=constraints)
            calc = ase.calculators.singlepoint.SinglePointCalculator(
                atoms, energy=struc.energy[-1], forces=struc.forces[-1])
            atoms.set_calculator(calc)
            trajec.append(atoms)

        traj = ase.io.trajectory.Trajectory(outfile, 'w', trajec[0])
        for atoms in trajec:
            traj.write(atoms)
        traj.close()
