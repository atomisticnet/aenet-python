"""
Interact with aenet's `predict.x` and its input and output files.

"""

from typing import List

import numpy as np
import os
import re

__author__ = "The aenet developers"
__email__ = "aenet@atomistic.net"
__date__ = "2023-04-17"
__version__ = "0.1"


class PredictIn(object):

    def __init__(self, types, networks, files, verbosity='low',
                 calc_forces=False, save_forces=False,
                 save_energies=False):
        self.types = types
        self.networks = networks
        self.files = files
        self.verbosity = verbosity
        self.calc_forces = calc_forces
        self.save_forces = save_forces
        self.save_energies = save_energies

    def __str__(self):
        out = "TYPES\n"
        out += "{}\n".format(self.num_types)
        for t in self.types:
            out += "{}\n".format(t)
        out += "\nNETWORKS\n"
        for t in self.networks:
            out += "{} {}\n".format(t, self.networks[t])
        out += "\nVERBOSITY {}\n".format(self.verbosity)
        if self.calc_forces:
            out += "\nNETWORKS\n"
        if self.save_forces:
            out += "\nSAVE_FORCES\n"
        if self.save_energies:
            out += "\nSAVE_ENERGIES\n"
        out += "\nFILES\n"
        out += "{}\n".format(self.num_files)
        for f in self.files:
            out += "{}\n".format(f)
        return out

    @property
    def num_types(self):
        return len(self.types)

    @property
    def num_files(self):
        return len(self.files)

    @classmethod
    def from_file(cls, filename: os.PathLike):
        if not os.path.exists(filename):
            raise FileNotFoundError("File not found: {}".format(filename))
        types = []
        networks = {}
        files = []
        verbosity = 'low'
        calc_forces = False
        save_forces = False
        save_energies = False
        with open(filename) as fp:
            line = "\n"
            while True:
                line = fp.readline()
                if line == '':
                    break
                if len(line.strip()) == 0:
                    continue
                kwd = line.strip().lower()
                if kwd == 'types':
                    num_types = int(fp.readline().strip())
                    for i in range(num_types):
                        types.append(fp.readline().strip())
                elif kwd == 'networks':
                    line = fp.readline()
                    while line and len(line.strip()) > 0:
                        networks.update([line.split()])
                        line = fp.readline()
                elif kwd == 'forces':
                    calc_forces = True
                elif kwd == 'save_forces':
                    save_forces = True
                elif kwd == 'save_energies':
                    save_energies = True
                elif kwd == 'verbosity':
                    verbosity = line.strip().split()[1].lower()
                elif kwd == 'files':
                    num_files = int(fp.readline().strip())
                    for i in range(num_files):
                        files.append(fp.readline().strip())
        return cls(types=types, networks=networks, files=files,
                   verbosity=verbosity, calc_forces=calc_forces,
                   save_forces=save_forces, save_energies=save_energies)


class PredictOut(object):

    def __init__(self, coords, forces, atom_types, cohesive_energy,
                 total_energy, predict_in_path=None):
        self.coords = coords
        self.forces = forces
        self.atom_types = atom_types
        self.cohesive_energy = cohesive_energy
        self.total_energy = total_energy
        if predict_in_path is not None:
            self.inputs = PredictIn.from_file(predict_in_path)
        else:
            self.inputs = None

    def __str__(self):
        return

    @property
    def paths(self):
        if self.inputs is not None:
            return self.inputs.files
        else:
            return None

    @property
    def num_structures(self):
        return len(self.cohesive_energy)

    @classmethod
    def from_file(cls, filename: os.PathLike, **kwargs):
        if not os.path.exists(filename):
            raise FileNotFoundError("File not found: {}".format(filename))
        coords = []
        forces = []
        atom_types = []
        cohesive_energy = []
        total_energy = []
        with open(filename) as fp:
            line = "\n"
            while line != '':
                line = fp.readline()
                if re.match(r'^ Cartesian atomic coordinates.*', line):
                    for i in range(5):
                        line = fp.readline()
                    coords_here = []
                    forces_here = []
                    types_here = []
                    while len(line.strip()) > 0:
                        fields = line.strip().split()
                        types_here.append(fields[0])
                        coords_here.append([float(a) for a in fields[1:4]])
                        forces_here.append([float(a) for a in fields[4:7]])
                        line = fp.readline()
                    atom_types.append(np.array(types_here))
                    coords.append(np.array(coords_here))
                    forces.append(np.array(forces_here))
                elif m := re.match(
                        r'^ *Cohesive energy *: *([0-9.-]*) eV.*$', line):
                    cohesive_energy.append(float(m.groups()[0]))
                elif m := re.match(
                        r'^ *Total energy *: *([0-9.-]*) eV.*$', line):
                    total_energy.append(float(m.groups()[0]))
        return cls(coords=coords, forces=forces, atom_types=atom_types,
                   cohesive_energy=cohesive_energy,
                   total_energy=total_energy, **kwargs)


class PredictOutAnalyzer(object):

    def __init__(self, pout_list: List[PredictOut]):
        self.pouts = pout_list
        if not self._check_compatible():
            raise ValueError("The 'predict.out' files are not compatible.")

    def __str__(self):
        return

    @property
    def num_structures(self):
        return self.pouts[0].num_structures

    def energy_uncertainty(self, i: int):
        """
        Return energy ucertainty for a selected structure.

        """
        energies = [po.total_energy[i] for po in self.pouts]
        return np.std(energies)

    def force_uncertainty(self, i: int):
        """
        Return energy ucertainty for a selected structure.

        """
        forces = [po.forces[i] for po in self.pouts]
        return np.std(forces, axis=0)

    def _check_compatible(self):
        if not all([po.num_structures == self.pouts[0].num_structures
                    for po in self.pouts[1:]]):
            return False
        if all(po.paths is not None for po in self.pouts):
            paths0 = self.pouts[0].paths
            for po in self.pouts[1:]:
                if not all([po.paths[i] == paths0[i]
                            for i in range(len(paths0))]):
                    return False
        return True
