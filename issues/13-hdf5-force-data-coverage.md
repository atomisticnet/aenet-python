# Issue 13: Add build-time controls for persisted force-data coverage in HDF5 datasets

**Type:** API  
**Priority:** Medium  
**Status:** Open  
**Created:** 2026-03-25

## Problem

Issue 12 moved runtime force-sampling and runtime-cache behavior into
`TorchTrainingConfig`, but users may still want independent control over how
much force-related information is persisted when an HDF5 dataset is built.

This is a storage-policy question, not a runtime training-policy question. A
single dataset may be prepared for many later runs, and users may want to
limit persisted force-related payloads for disk usage, preprocessing time, or
workflow-specific reuse patterns.

Build-time coverage of persisted force payloads should therefore be
configurable without reintroducing ambiguity with runtime training controls
such as `force_fraction` and `force_sampling`.

## Proposed approach

Introduce explicit build-time controls for force-data persistence coverage in
the HDF5 preparation workflow, keeping that policy separate from
`TorchTrainingConfig`.

The design should determine:

- whether the controls apply only to persisted derivative caches or also to
  other optional force-related payloads;
- whether a dedicated build-time option or configuration object is appropriate
  for `HDF5StructureDataset.build_database(...)`;
- how defaults avoid accidentally discarding reusable force information; and
- how build-time storage coverage interacts with runtime training policy.

## Acceptance criteria

- Build-time force-data coverage can be configured independently of runtime
  force supervision and sampling.
- Defaults preserve the current conservative behavior unless a user explicitly
  requests reduced coverage.
- Persisted payloads remain readable through the existing HDF5 dataset API.
- Unit tests cover default, partial-coverage, and reopened-dataset behavior.
- Public API and Sphinx documentation clearly distinguish storage policy from
  runtime training policy.

## Notes

- This issue is intentionally separate from completed Issue 12.
- The implementation must not reintroduce dataset-versus-config ambiguity for
  runtime training policy.

