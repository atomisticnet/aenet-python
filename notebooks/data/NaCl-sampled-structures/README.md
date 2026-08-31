# UMA NaCl Sampled Structures

This directory makes the input and provenance for
`example-09-sampled-structures-downselection.ipynb` available from a clean checkout.

## Dataset

`sampled_structures.tar.xz` contains 20,000 XSF structures with 64 atoms each:

| Temperature | Structures |
| --- | ---: |
| 550 K | 5,000 |
| 700 K | 5,000 |
| 850 K | 5,000 |
| 1000 K | 5,000 |

The archive SHA-256 is
`00e5f8d8313f05a92a96e775e11a33b66fe5c8d37aab0b62319be0562f67a5e2`.
Each XSF file stores the cell, atomic positions, UMA total energy, and UMA
forces. The archive is the authoritative dataset used by the notebook.

Extract it with:

```bash
tar -xJf sampled_structures.tar.xz
```

## Precomputed Features

`sampled_structure_features.npz` contains the global-moment Chebyshev
fingerprints for all 20,000 tracked structures. Its `features` array has shape
`(20000, 60)`; `paths` preserves the source filenames used to label rows, and
`source_indices` maps each row to its original sorted input index. The notebook
loads this file by default so the full analysis does not repeat featurization.

The features were generated on CPU with `float64`, radial order 10 and cutoff
4.8 Angstrom, angular order 3 and cutoff 3.75 Angstrom, and minimum cutoff 0.5
Angstrom. Global moments used outer and inner moment 1 with weighted moments
enabled and appended. The file SHA-256 is
`ed4c22dc0ba2065320c503b3ddd84102929f400a3b2b61b2422a68e1ae8e07fa`.

Set `FEATURE_FILE = None` in the notebook to rebuild the same representation
from `sampled_structures.tar.xz` instead.

## Generation Protocol

`NaCl_2x2x2.vasp` is the 64-atom starting structure. For each temperature,
`scripts/uma_md.py` used the FairChem `uma-s-1p1` predictor with the `omat`
task on CUDA, a 2 fs time step, 20 ps of NPT equilibration, and 100 ps of NPT
production at zero external stress. It wrote 5,000 production frames.

The original runs did not record the random seed used to initialize velocities.
Consequently, the scripts document and reproduce the protocol but cannot
regenerate the archived coordinates byte for byte. This limitation is why the
complete XSF dataset is tracked.

Run `uma_md.py` from a directory such as `550K/run`, with the VASP file in that
directory. The parent directory supplies the temperature. Then convert the
trajectory directly into the tracked naming scheme:

```bash
python scripts/traj_to_xsf.py --output-dir sampled_structures
```

The conversion skips trajectory frame 0, which is the post-equilibration
structure, and writes production frame 1 as `snapshot_0001_<temperature>K.xsf`.

## Descriptor Cutoffs

`scripts/analyze_chebyshev_cutoffs.py` analyzes periodic neighbor-shell
distances in the extracted XSF files:

```bash
python scripts/analyze_chebyshev_cutoffs.py \
  --input-dir sampled_structures \
  --output-dir cutoff_analysis
```

The tracked report and plots in `cutoff_analysis/` support the rounded settings
used by example 09:

```python
rad_cutoff = 4.8
ang_cutoff = 3.75
```

The fitted values were 4.770 Angstrom and 3.735 Angstrom, respectively. See
`cutoff_analysis/chebyshev_cutoff_report.md` for shell-overlap limitations and
the full per-temperature statistics.
