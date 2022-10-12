#!/usr/bin/env python3

import pandas as pd

from .tools import AenetToolABC
from ..trainset import TrnSet

__author__ = "The aenet developers"
__email__ = "aenet@atomistic.net"
__date__ = "2020-11-30"


class SFP(AenetToolABC):
    """ Compute structure fingerprints """

    def _set_arguments(self):
        self.parser.add_argument(
            "training_set_file",
            help="aenet training set in ASCII, HDF5, or binary format.")

        self.parser.add_argument(
            "-m", "--moment",
            help="Maximal moment for fingerprint expansion (default: 2).",
            type=int,
            default=2)

        self.parser.add_argument(
            "-o", "--output-file",
            help="Path to the CSV output file (default: structures.csv). "
                 "Note: if file exists, data will be appended.",
            type=str,
            default="structures.csv")

        self.parser.add_argument(
            "-t", "--atom-types",
            help="Selected atom types (default: use all).",
            type=str,
            default=None,
            nargs="+")

    def _man(self):
        return """
        Featurize atomic structures by calculating structure
        'fingerprints' using the approach of reference [1].

        [1] H. Guo, Q. Wang, A. Urban, N. Artrith, arXiv:2201.11203, 2022,
            https://arxiv.org/abs/2201.11203

        Parses a training-set file produced by ``generate.x`` and
        converted to ASCII format with ``trnset2ascii.x``.  The atomic
        environment features of a structure are combined by calculating
        moments of their distribution (mean, standard deviation, etc.).
        For further details, see the original publication [1].

        The dimension of the resulting structural fingerprints is

           :math:`D = D_{atom} * N_{moment}`

        where :math:`D_{atom}` is the dimension of the atomic
        environment descriptor, and :math:`N_{moment}` is the maximal
        moment used for the fingerprint expansion.

        Note: Usually, the dimension of the structure fingerprints can
        be significanlty reduced using standard dimension reduction
        methods, such as principal component analysis.

        """

    def analyze(self, training_set_file, moment, output_file, atom_types):
        ts = TrnSet.from_file(training_set_file)
        print(ts)
        print("Maximal moment for structure fingerprints: {}".format(moment))
        print("Writing structure fingerprints to '{}'.".format(output_file))

        with open(output_file, 'a') as fp:
            s = ts.read_next_structure()
            sfp = s.moment_fingerprint(
                sel_atom_types=atom_types, moment=moment)
            columns = list(range(len(sfp))) + ["num_atoms", "energy", "path"]
            df = pd.DataFrame([sfp + [s.num_atoms, s.energy, s.path]],
                              columns=columns)
            df.to_csv(fp, header=True)
            for i in range(ts.num_structures - 1):
                s = ts.read_next_structure()
                sfp = s.moment_fingerprint(sel_atom_types=atom_types,
                                           moment=moment)
                df = pd.DataFrame([sfp + [s.num_atoms, s.energy, s.path]],
                                  columns=columns)
                df.to_csv(fp, header=False)

    def run(self, args):
        self.analyze(args.training_set_file, args.moment,
                     args.output_file, args.atom_types)


if __name__ == "__main__":
    tool = SFP()
    args = tool.parser.parse_args()
    tool.run(args)
