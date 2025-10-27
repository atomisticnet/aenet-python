"""
Atomic structure featurization.

"""

import inspect
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List
import numpy as np

from . import config
from .util import cd
from .trainset import TrnSet
from .geometry import AtomicStructure
from .formats.xsf import XSFParser

__author__ = "Alexander Urban"
__date__ = "2022-10-04"


class AtomicFeaturizer(object):

    def __init__(self, typenames: List[str], **kwargs):
        self.typenames = typenames
        for arg in kwargs:
            print("Warning: unknown keyword `{}`".format(arg))

    @classmethod
    def from_structure(cls, struc: AtomicStructure, **kwargs):
        if not struc.names_known:
            raise ValueError("Structure without type name information.")
        return cls(typenames=struc.typenames)

    @property
    def ntypes(self):
        return len(self.typenames)


class AenetAUCFeaturizer(AtomicFeaturizer):

    def __init__(self, typenames: List[str],
                 rad_order: int = 0, rad_cutoff: float = 0.0,
                 ang_order: int = 0, ang_cutoff: float = 0.0,
                 min_cutoff: float = 0.55, **kwargs):
        super().__init__(typenames, **kwargs)
        self.rad_order = rad_order
        self.rad_cutoff = rad_cutoff
        self.ang_order = ang_order
        self.ang_cutoff = ang_cutoff
        self.min_cutoff = min_cutoff

    def setup_file_strings(self):
        header = """
        DESCR
          Featurization set-up file for species {species} using the
          AUC method [1].
          This file was generated using the aenet-python package.
          Please cite the following reference when publishing results
          based on this input file.
          [1] N. Artrith, A. Urban and Ceder, Phys. Rev. B 96, 2017, 014112,
              https://doi.org/10.1103/PhysRevB.96.014112
        END DESCR

        ATOM {species}
        """

        env = "ENV {}".format(self.ntypes)
        for t in self.typenames:
            env += "\n{}".format(t)

        footer = """
        RMIN {min_cutoff}

        BASIS type=Chebyshev
        radial_Rc = {rc}  radial_N = {ro} angular_Rc = {ac}  angular_N = {ao}
        """.format(min_cutoff=self.min_cutoff,
                   rc=self.rad_cutoff, ro=self.rad_order,
                   ac=self.ang_cutoff, ao=self.ang_order)

        setup_files = {}
        for t in self.typenames:
            s = inspect.cleandoc(header.format(species=t)) + "\n\n"
            s += env + "\n\n" + inspect.cleandoc(footer)
            setup_files[t] = s

        return setup_files

    def generate_input_string(self,
                              xsf_files: List[os.PathLike],
                              output_file: str = 'data.train',
                              atomic_energies: Dict[str, float] = None,
                              workdir: os.PathLike = '.',
                              debug: bool = False,
                              forces: bool = False,
                              forcespercent: float = 100.0,
                              deriv_method: str = None,
                              deriv_delta: float = 0.01,
                              deriv_fraction: float = 16.0,
                              deriv_outfile: str = 'data.deriv.train',
                              deriv_disp_all: bool = True,
                              deriv_N_max: int = 100000,
                              **kwargs):

        for arg in kwargs:
            raise TypeError("Unexpected keyword argument '{}'".format(arg))

        if deriv_method not in ["taylor", "analytical", None]:
            raise ValueError(
                'Invalid derivative method: {}'.format(deriv_method))

        if atomic_energies is None:
            atomic_energies = {}
        for t in self.typenames:
            if t not in atomic_energies:
                atomic_energies[t] = 0.0

        # check if all xsf files listed exist and make paths relative to
        # workdir
        xsf_file_paths = []
        for f in xsf_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f)
            xsf_file_paths.append(
                os.path.relpath(os.path.abspath(f), os.path.abspath(workdir)))

        generate_in = "OUTPUT {}\n\nTYPES\n{}\n".format(
            output_file, self.ntypes)
        for t in self.typenames:
            generate_in += "{:2s}  {}\n".format(t, atomic_energies[t])

        generate_in += "\nSETUPS\n"
        for t in self.typenames:
            generate_in += "{:2s}  {}.stp\n".format(t, t)

        if deriv_method is not None:
            if deriv_method == 'analytical':
                forces = True
                raise Warning(
                    "Analytical derivatives requested.  The `FORCES' "
                    "option for training with PyTorch will be activated.  "
                    "This option is only supported by aenet-PyTorch.")
            elif deriv_method == 'taylor':
                generate_in += "\nDERIVATIVES\n"
                generate_in += ("method=taylor delta={delta} fraction={frac} "
                                + "outfile={out} disp_all={all} N_max={max}\n"
                                ).format(delta=deriv_delta,
                                         frac=deriv_fraction,
                                         out=deriv_outfile,
                                         all=(1 if deriv_disp_all else 0),
                                         max=deriv_N_max)
            else:
                raise ValueError(
                    "Unexpected derivative method '{}'".format(deriv_method))

        if forces:
            if not 0 <= forcespercent <= 100:
                raise ValueError(
                    "Forces percent must be between 0 and 100.")
            generate_in += "\nFORCES"
            generate_in += "\nFORCESPERCENT {}\n".format(forcespercent)

        if debug:
            generate_in += "\nDEBUG\n"

        generate_in += "\nFILES\n{}\n".format(len(xsf_file_paths))
        for p in xsf_file_paths:
            generate_in += "{}\n".format(p)

        return generate_in

    def write_generate_input_files(self,
                                   xsf_files: List[os.PathLike],
                                   filename: str = 'generate.in',
                                   workdir: os.PathLike = '.', **kwargs):
        """
        All keyword args are forwarded to `generate_input_string`.  See
        this method for documentation.

        """
        stp_strings = self.setup_file_strings()
        generate_in = self.generate_input_string(
            xsf_files, workdir=workdir, **kwargs)

        if not os.path.exists(workdir):
            os.makedirs(workdir, exist_ok=True)

        outfile = os.path.join(workdir, filename)
        if os.path.exists(outfile):
            raise AssertionError('File already exists: {}'.format(outfile))

        with open(outfile, 'w') as fp:
            fp.write(generate_in)

        for t in self.typenames:
            outfile = os.path.join(workdir, '{}.stp'.format(t))
            if os.path.exists(outfile):
                raise AssertionError('File already exists: {}'.format(outfile))
            with open(outfile, 'w') as fp:
                fp.write(stp_strings[t])

    def run_aenet_generate(self, xsf_files: List[os.PathLike],
                           workdir=None, hdf5_filename='features.h5',
                           output_file='generate.out', **kwargs):
        """
        Run aenet's `generate.x` tool to evaluate the features of a list
        of XSF files. The features are written to an HDF5 output file.

        Args:
            xsf_files: List of paths to XSF files in aenet's format.
            workdir: Directory for the creation of input/output files.
                Defaults to None.
            hdf5_filename: Name of the output file with generated
                features. Defaults to 'features.h5'.
            output_file: Name of the `generate.x` output file. Will only
                be accessible if `workdir` is not None.
                Defaults to 'generate.out'.

        Raises:
            FileNotFoundError: Raised when the `generate.x` tool is not found.
        """
        aenet_paths = config.read('aenet')
        if not os.path.exists(aenet_paths['generate_x_path']):
            raise FileNotFoundError(
                "Cannot find `generate.x`. Configure with `aenet config`.")

        if workdir is None:
            workdir = tempfile.mkdtemp(dir='.')
            rm_tmp_files = True
        else:
            if not os.path.exists(workdir):
                os.makedirs(workdir, exist_ok=True)
            rm_tmp_files = False

        self.write_generate_input_files(
            xsf_files, workdir=workdir, output_file='data.train', **kwargs)

        with cd(workdir) as cm:
            outfile = os.path.join(cm['origin'], output_file)
            errfile = 'errors.out'
            with open(outfile, 'w') as out, open(errfile, 'w') as err:
                subprocess.run([aenet_paths['generate_x_path'],
                                'generate.in'], stdout=out, stderr=err)

        ts = TrnSet.from_fortran_binary_file(
            os.path.join(workdir, 'data.train'),
            origin=workdir)
        ts.to_hdf5(hdf5_filename)
        ts.close()

        if rm_tmp_files:
            shutil.rmtree(workdir)

    def featurize_structures(self, structures: List[AtomicStructure],
                             hdf5_filename=None, **kwargs):
        """
        Runs aenet's `generate.x` tool and returns the feature vectors.

        Args:
            structures: List of .geometry.AtomicStructure
            hdf5_filename: Path to the generated HDF5 file with the
                featurized structures. If None, a temporary file will
                be created and deleted when the featurization is done.

        Returns:
            a list of .trainset.FeaturizedAtomicStructure objects

        """
        # generate XSF files for featurization with generate.x
        xsf_dir = tempfile.mkdtemp(dir='.')
        xsf = XSFParser()
        xsf_files = []
        for i, s in enumerate(structures):
            struc = s.copy()
            xsf_filename = os.path.join(xsf_dir, 'structure{}.xsf'.format(i))
            xsf_files.append(xsf_filename)
            # If the structure is not labeled with energies and forces,
            # it can still be featurized.  Setting these labels to zero
            # when they aren't set.
            if struc.energy[-1] is None:
                struc.energy[-1] = 0.0
            if struc.forces[-1] is None or len(struc.forces[-1]) == 0:
                struc.forces[-1] = np.zeros(struc.coords[-1].shape)
            xsf.write(struc, outfile=xsf_filename)

        # now perform the featurization
        if hdf5_filename is None:
            workdir = tempfile.mkdtemp(dir='.')
            trnset_file = os.path.join(workdir, 'features.h5')
            rm_tmp_files = True
        else:
            if os.path.exists(hdf5_filename):
                raise IOError('File already exists: {}'.format(hdf5_filename))
            trnset_file = hdf5_filename
            rm_tmp_files = False

        self.run_aenet_generate(xsf_files, hdf5_filename=trnset_file, **kwargs)
        featurized_structures = []
        with TrnSet.from_file(trnset_file) as ts:
            for s in ts:
                featurized_structures.append(s)

        if rm_tmp_files:
            shutil.rmtree(workdir)
        shutil.rmtree(xsf_dir)

        return featurized_structures

    def featurize_structure(self, struc: AtomicStructure, **kwargs):
        """
        Runs aenet's `generate.x` tool and returns the feature vectors.
        This is the same as `featurize_structures()` but for a single
        structure.  See the docstring of `featurize_structures()` for
        additional arguments.

        Args:
            struc: Instance of .geometry.AtomicStructure

        Returns:
            An instance of .trainset.FeaturizedAtomicStructure

        """
        return self.featurize_structures([struc], **kwargs)[-1]
