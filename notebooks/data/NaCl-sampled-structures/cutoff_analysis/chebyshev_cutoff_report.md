# NaCl Chebyshev Cutoff Analysis

Analyzed `20000` XSF files with `64` atoms per structure.

## Recommended Cutoffs

- Fitted angular cutoff: `3.735` Angstrom
- Fitted radial cutoff: `4.770` Angstrom
- Practical descriptor setting: `ang_cutoff = 3.75` Angstrom
- Practical descriptor setting: `rad_cutoff = 4.8` Angstrom

Use these rounded descriptor settings:

```python
rad_cutoff = 4.8
ang_cutoff = 3.75
```

The exact fitted values are reported below so the recommendation remains traceable to the data.

## Interpretation

The angular cutoff is placed between neighbor ranks 1-6 and ranks 7-18. For rocksalt NaCl, ranks 1-6 are the first unlike-ion coordination shell, while ranks 7-18 are the second same-sublattice shell.

The radial cutoff is placed between ranks 7-18 and ranks 19-26, so radial features include the first and second shells while avoiding the next shell.

At 850 K and 1000 K the shells overlap substantially. No single hard cutoff can include every thermally broadened second-shell distance while excluding the next shell. The recommended radial cutoff is therefore a balanced separator, not a perfect high-temperature second-shell envelope.

## Cutoff Selection Statistics

| Boundary | Lower shell 99.9% | Upper shell 0.1% | Lower max | Upper min | Method |
|---|---:|---:|---:|---:|---|
| 1st/2nd shell, angular | 4.4626 | 3.3984 | 5.2833 | 2.8694 | maximized desired-shell capture minus next-shell leakage |
| 2nd/3rd shell, radial | 5.8648 | 4.4708 | 6.5309 | 4.2225 | maximized desired-shell capture minus next-shell leakage |

## Per-Temperature Shell Summary

| Temperature | Shell | 0.1% | Median | 99.9% | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| 550K | first | 2.3924 | 2.9221 | 3.7857 | 2.2285 | 4.0612 |
| 550K | second | 3.4056 | 4.1549 | 4.8418 | 3.0578 | 5.1069 |
| 550K | third | 4.4285 | 5.0898 | 5.7238 | 4.2225 | 5.9587 |
| 700K | first | 2.3539 | 2.9478 | 3.8688 | 2.1784 | 4.1487 |
| 700K | second | 3.3819 | 4.2010 | 4.9420 | 3.0556 | 5.2523 |
| 700K | third | 4.4629 | 5.1505 | 5.8051 | 4.2471 | 6.0617 |
| 850K | first | 2.2844 | 3.0476 | 4.4631 | 2.1001 | 4.8748 |
| 850K | second | 3.3931 | 4.6739 | 5.8260 | 2.8694 | 6.2854 |
| 850K | third | 4.8724 | 5.6974 | 6.5411 | 4.5128 | 7.0138 |
| 1000K | first | 2.2618 | 3.0972 | 4.5854 | 2.0633 | 5.2833 |
| 1000K | second | 3.4233 | 4.7750 | 5.9873 | 3.0160 | 6.5309 |
| 1000K | third | 4.9463 | 5.8116 | 6.7503 | 4.5241 | 7.3610 |

## Cutoff Coverage Check

| Temperature | 1st shell <= angular cutoff | 2nd shell <= angular cutoff | 2nd shell <= radial cutoff | 3rd shell <= radial cutoff |
|---|---:|---:|---:|---:|
| 550K | 99.75% | 5.22% | 99.60% | 12.59% |
| 700K | 99.13% | 5.88% | 98.36% | 9.50% |
| 850K | 85.43% | 2.37% | 56.53% | 0.02% |
| 1000K | 82.20% | 1.71% | 49.64% | 0.01% |
| all | 91.63% | 3.79% | 76.03% | 5.53% |

## Species-Pair Check

- first: Cl-Cl: 6.81%, Na-Cl: 85.72%, Na-Na: 7.47%
- second: Cl-Cl: 40.92%, Na-Cl: 19.22%, Na-Na: 39.86%
- third: Cl-Cl: 14.15%, Na-Cl: 70.69%, Na-Na: 15.17%

## Validation Plots

- `shell_distance_histograms_by_temperature.png`: first, second, and third rank-shell distance distributions by temperature.
- `shell_quantile_summary.png`: shell medians and 0.1-99.9% ranges by temperature.
- `species_pair_distance_histograms.png`: Na-Cl, Na-Na, and Cl-Cl pair-distance distributions by temperature.
- `cutoff_tradeoff_curves.png`: desired-shell capture and next-shell leakage as a function of cutoff.

Dashed vertical line = angular cutoff. Solid vertical line = radial cutoff.

## Notes

- Distances are periodic minimum-image distances computed from each XSF cell.
- Used exact 27-image periodic minimum-image search around wrapped fractional displacements for every pair; the faster rounded-fraction method failed validation for these skewed cells.
- Shell labels use rocksalt NaCl coordination counts: 6 first-shell neighbors, 12 second-shell neighbors, and 8 next-shell neighbors.
- The recommendation is intentionally conservative against high-temperature broadening by using all snapshots from every temperature.
