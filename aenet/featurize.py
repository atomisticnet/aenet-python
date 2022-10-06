"""
Atomic structure featurization.

"""

import inspect
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List

from . import config
from .util import cd

__author__ = "Alexander Urban"
__date__ = "2022-10-04"


class AtomicFeaturizer(object):

    def __init__(self, typenames: List[str], **kwargs):
        self.typenames = typenames
        for arg in kwargs:
            print("Warning: unknown keyword `{}`".format(arg))

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
          Featurization set-up file for species {species} using the AUC method [1].
          This file was generated using the aenet Python package.
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
                              debug=False):
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

        generate_in = "OUTPUT {}\n\nTYPES\n{}\n".format(output_file, self.ntypes)
        for t in self.typenames:
            generate_in += "{:2s}  {}\n".format(t, atomic_energies[t])

        generate_in += "\nSETUPS\n"
        for t in self.typenames:
            generate_in += "{:2s}  {}.stp\n".format(t, t)

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
        generate_in = self.generate_input_string(xsf_files, **kwargs)

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

    def run_aenet_generate(self, xsf_files: List[os.PathLike]):
        aenet_paths = config.read('aenet')
        if not os.exists(aenet_paths['generate_x_path']):
            raise FileNotFoundError(
                "Cannot find `generate.x`. Configure with `aenet config`.")

        workdir = tempfile.mkdtemp(dir='.')
        self.write_generate_input_files(
            xsf_files, workdir=workdir, output_file='data.train')
        with cd(workdir) as cm:
            with open(os.join(cm['origin'], 'generate.out'), 'w') as fp:
                subprocess.run([aenet_paths['generate_x_path'], 
                                'generate.in'], stdout=fp)
                shutil.move('data.train', cm['origin'])
            
        shutil.rmtree(workdir)

