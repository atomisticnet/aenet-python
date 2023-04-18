"""
Interact with aenet's `predict.x` and its input and output files.

"""

from typing import List

import numpy as np
import os
import re

from .io.structure import read as read_structure
from .io.structure import write as write_structure

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
        base_path = os.path.dirname(filename)
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
                        p = fp.readline().strip()
                        files.append(os.path.join(base_path, p))
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

    def num_atoms(self, i):
        """ number of atoms in a select structure """
        return len(self.atom_types[i])

    def structure(self, i, frmt=None, **kwargs):
        """ return selected atomic structure """
        if self.paths is None:
            raise ValueError("File paths unknown. Cannot read structure.")
        struc = read_structure(self.paths[i], frmt=frmt, **kwargs)
        return struc

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

    @property
    def paths(self):
        return self.pouts[0].paths

    def num_atoms(self, i):
        return self.pouts[0].num_atoms(i)

    def structure(self, i, frmt=None, **kwargs):
        return self.pouts[0].structure(i, frmt=frmt, **kwargs)

    def write_pdb_with_force_uncertainty(self, i, filename, frmt=None,
                                         **kwargs):
        struc = self.structure(i)
        force_u = self.force_uncertainty(i)
        write_structure(struc, filename=filename, frmt=frmt,
                        atom_attrib=force_u, **kwargs)

    def energy_stats(self, i: int, normalize: bool = True):
        """
        Return energy ucertainty for a selected structure.

        """
        energies = np.array([po.total_energy[i] for po in self.pouts])
        if normalize:
            energies /= self.num_atoms(i)
        E_min = np.min(energies)
        E_max = np.max(energies)
        E_avg = np.mean(energies)
        E_std = np.std(energies)
        return E_min, E_max, E_avg, E_std

    def all_energy_stats(self, **kwargs):
        data = []
        for i in range(self.num_structures):
            data.append(self.energy_stats(i))
        columns = ['E_min', 'E_max', 'E_mean', 'E_std']
        return np.array(data), columns

    def force_stats(self, i: int):
        """
        Return energy ucertainty for a selected structure.

        """
        forces = [po.forces[i] for po in self.pouts]
        return np.std(forces, axis=0)

    def force_uncertainty(self, i: int):
        """
        Return an atomic uncertainty for a selected structure.

        """
        forces = [po.forces[i] for po in self.pouts]
        return np.mean(np.std(forces, axis=0), axis=1)

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
