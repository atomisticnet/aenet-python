#!/usr/bin/env python

"""
Parse aenet training set file and write out structure fingerprints
(as opposed to atomic fingerprints) as a CSV file.

"""

import pandas as pd
import argparse

from aenet.trainset import TrnSet

__author__ = "The aenet developers"
__email__ = "aenet@atomistic.net"
__date__ = "2020-10-31"
__version__ = "0.1"


def analyze(training_set_file, moment, output_file):

    ts = TrnSet.from_ascii_file(training_set_file)
    print(ts)
    print("Maximal moment for structure fingerprints: {}".format(moment))
    print("Writing structure fingerprints to '{}'.".format(output_file))

    with open(output_file, 'a') as fp:
        s = ts.read_next_structure()
        dim = s.max_descriptor_length
        columns = list(range(dim*moment*ts.num_types)
                       ) + ["num_atoms", "energy", "path"]
        sfp = s.moment_fingerprint(ts.atom_types, moment=2)
        df = pd.DataFrame([sfp + [s.num_atoms, s.energy, s.path]],
                          columns=columns)
        df.to_csv(fp, header=True)
        for i in range(ts.num_structures - 1):
            s = ts.read_next_structure()
            sfp = s.moment_fingerprint(ts.atom_types, moment=2)
            df = pd.DataFrame([sfp + [s.num_atoms, s.energy, s.path]],
                              columns=columns)
            df.to_csv(fp, header=False)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__+"\n{} {}".format(__date__, __author__),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "training_set_file",
        help="aenet training set in ASCII format.")

    parser.add_argument(
        "-m", "--moment",
        help="Maximal moment for fingerprint expansion (default: 2).",
        type=int,
        default=2)

    parser.add_argument(
        "-o", "--output-file",
        help="Path to the CSV output file (default: structures.csv). "
             "Note: if file exists, data will be appended.",
        type=str,
        default="structures.csv")

    args = parser.parse_args()

    analyze(args.training_set_file, args.moment, args.output_file)


if (__name__ == "__main__"):
    main()
