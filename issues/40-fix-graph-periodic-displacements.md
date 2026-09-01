# Issue 40: Fix periodic displacement reconstruction in graph-based descriptors

**Type:** Bug
**Priority:** High
**Status:** Open
**Created:** 2026-09-01

## Problem

The graph-based descriptor path reconstructs periodic edge displacement vectors
from a neighbor index, an integer lattice offset, and the original Cartesian
positions. The default periodic neighbor-list backend defines those offsets
relative to fractional coordinates that it has first wrapped into the primary
cell. `aenet.torch_featurize.graph._compute_r_ij`, however, currently applies
the offsets directly to the original, potentially unwrapped Cartesian
positions:

```text
r_ij = (positions[j] + offsets @ cell) - positions[i].
```

That expression is valid only when the input positions are already wrapped in
the same image convention used by the neighbor-list backend. For periodic
structures containing atoms outside the primary cell, such as unwrapped
molecular-dynamics snapshots, the returned neighbor distance can therefore
disagree with the norm of the reconstructed edge vector. The graph path then
uses an incorrect edge direction and can produce descriptors that differ from
the standard featurization path for a physically equivalent structure.

The backend contract requires reconstruction from consistently wrapped
fractional coordinates:

```text
frac = remainder(positions @ inverse(cell), 1.0)
r_ij = ((frac[j] + offsets) @ cell) - (frac[i] @ cell).
```

This defect affects the differentiable graph route used for direct-force
training. It can corrupt angular descriptor values and their coordinate
derivatives, and it can also change energies evaluated through that graph
route. It must not be described as a force-only error. Energy-only and Taylor-
expanded energy training use the standard descriptor path and are not affected
by this specific reconstruction defect.

## Current evidence

The index and offset flow has been traced through the default neighbor-list and
graph builders:

- the neighbor-list backend converts Cartesian positions to fractional
  coordinates and wraps them into `[0, 1)` before constructing periodic ghost
  images;
- `edge_index[0]` remains the central atom index, `edge_index[1]` remains the
  original neighbor atom index, and the returned offset is the neighbor-image
  lattice offset relative to those wrapped coordinates;
- edge filtering and stable sorting apply the same selection and permutation
  to indices, distances, offsets, and displacement vectors;
- CSR and angular-triplet construction preserve this edge alignment; and
- the standard descriptor implementation wraps fractional positions before
  applying the same offsets, while `_compute_r_ij` does not.

A controlled two-atom test in a 10-Angstrom periodic cell isolates the failure.
For positions at `9.8` and `0.2` Angstrom, the backend and graph path both
produce the expected `0.4`-Angstrom neighbor vector. Replacing the second
coordinate with the physically equivalent unwrapped value `10.2` Angstrom
leaves the backend indices, offset, and distance unchanged, but the current
graph reconstruction produces a vector with norm `10.4` Angstrom. Thus one
edge simultaneously reports `d_ij = 0.4` Angstrom and `|r_ij| = 10.4`
Angstrom.

A controlled three-atom angular test gives agreement between the standard and
graph descriptors to approximately `2.2e-12` for wrapped coordinates. Moving
one atom to an equivalent unwrapped image produces a maximum descriptor
difference of approximately `0.428`; the radial block remains equal while the
angular block differs. Reconstructing only `r_ij` with the wrapped-fractional
formula restores agreement to approximately `2.2e-12` without changing edge
indices, offsets, sorting, CSR construction, or triplet construction.

An independent production audit of the Issue 38 NaCl runs found the same class
of inconsistency on real data:

- standard and graph descriptors differed by as much as `0.417431`;
- graph- and standard-path model energies differed by as much as
  `0.108228 eV/atom`; and
- the final training-loop validation energy MAE (`0.013883 eV/atom`) did not
  agree with post-training standard-path validation (`0.093238 eV/atom`).

The same audit verified that the test structures, energies, forces, atom-to-
structure mapping, and saved prediction ordering were aligned. Direct-force
predictions also outperformed shifted, atom-shuffled, and sign-reversed
controls. These checks rule out label mismatching as the explanation for the
observed graph/standard discrepancy.

## Impact

Until this issue is fixed:

- direct-force models trained on unwrapped periodic structures may optimize a
  representation that is inconsistent with standard inference;
- graph-path energy and force metrics cannot be compared reliably with
  energy-only or Taylor-trained models evaluated through the standard path;
- angular descriptors and force derivatives can depend on the arbitrary image
  chosen for an atom rather than only on the physical periodic structure;
- training-loop validation can appear substantially better than post-training
  validation when the two stages use different descriptor paths; and
- production direct-force results affected by the defect must be treated as
  invalid and retrained after the fix.

## Proposed approach

1. Define one explicit periodic-offset contract for all neighbor-list
   backends. State whether offsets are relative to original Cartesian
   positions or wrapped fractional positions and ensure each backend documents
   and satisfies that contract.
2. Update `_compute_r_ij` to reconstruct vectors in the convention of the
   selected backend. For the default backend, wrap fractional coordinates
   before applying integer lattice offsets. Preserve dtype, device, gradient
   flow, batched-cell behavior, and nonperiodic behavior.
3. Keep edge indices, offsets, distances, CSR row pointers, and angular
   triplets unchanged unless a new failing test demonstrates a separate
   defect. The current evidence localizes the error to vector reconstruction,
   not index ordering.
4. Add invariant checks in tests that compare the returned neighbor distance
   with `norm(r_ij)` for periodic edges and verify that both are unchanged by
   adding arbitrary integer lattice vectors to individual atomic positions.
5. Add standard-versus-graph descriptor parity tests for wrapped and unwrapped
   representations of the same orthorhombic and triclinic periodic structures,
   including an angular three-body environment.
6. Add energy and force regression tests using one fixed model. Equivalent
   periodic images must give equivalent graph-path predictions, and graph- and
   standard-path energies must agree within the established numerical
   tolerance.
7. Audit all call sites and alternative neighbor-list backends before changing
   the shared helper. If backend offset conventions differ, normalize their
   outputs at the neighbor-list boundary rather than adding undocumented
   backend-specific assumptions to descriptor code.
8. After the unit and integration tests pass, rerun a small direct-force smoke
   training job. Confirm that the training-loop and post-training validation
   metrics use equivalent representations before repeating production runs.

## Acceptance criteria

- `_compute_r_ij` satisfies the documented neighbor-list offset convention for
  periodic and nonperiodic structures.
- For every tested periodic edge, `norm(r_ij)` agrees with the neighbor-list
  distance within dtype-appropriate tolerances.
- Adding independent integer lattice translations to atomic coordinates does
  not change neighbor distances, radial descriptors, angular descriptors,
  graph-path energies, or graph-derived forces beyond numerical tolerance.
- Standard and graph descriptor paths agree for equivalent wrapped and
  unwrapped orthorhombic and triclinic structures.
- Tests exercise boundary-crossing pairs and at least one angular triplet; a
  radial-only test is insufficient because the observed descriptor failure is
  concentrated in the angular block.
- Autograd remains connected through positions and cells where currently
  supported, and existing direct-force tests continue to pass.
- A small direct-force training smoke test reports consistent validation
  energy metrics during training and after model reload/evaluation through the
  standard path.
- The affected Issue 38 direct-force production runs are marked for retraining;
  energy-only and Taylor-expanded runs are not invalidated by this defect.

## Notes

This issue is a prerequisite for a scientifically valid direct-force baseline
in Issue 38. It is limited to periodic displacement reconstruction and the
resulting descriptor-path inconsistency; it does not implement Taylor sampling,
Gaussian-process regression, or the ensemble uncertainty-quantification work
reserved for Issue 39.
