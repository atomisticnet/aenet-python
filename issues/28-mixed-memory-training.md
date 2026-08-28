# Issue 28: Implement a real mixed-memory training mode for large force-training datasets

**Type:** Feature  
**Priority:** Medium  
**Status:** Open  
**Created:** 2026-03-27

## Problem

The PyTorch backend reserves `memory_mode="mixed"` for a future real
mixed-memory execution mode and raises `NotImplementedError` when it is
requested.

This leaves a capability gap for large force-training workloads:

- force training has much larger per-batch payloads than energy-only training
  because descriptor derivatives and graph/triplet data are large;
- large datasets may fit poorly in fully GPU-resident workflows; and
- the legacy `aenet-PyTorch` package provided useful prior art through a disk
  mode that staged batch payloads outside GPU memory and transferred them on
  demand.

## Proposed approach

Implement a real mixed-memory training mode with explicit semantics, starting
with force-training use cases that benefit most from reduced peak GPU memory
pressure.

The design should define:

- whether mixed mode means CPU-resident batches, disk-backed staged batches,
  or both;
- what stays on `descriptor.device`, what stays in host memory or on disk, and
  what is transferred to `config.device` for an active batch;
- how the mode interacts with `num_workers`, HDF5-backed datasets, persisted
  feature/derivative reuse, and trainer-owned runtime caches;
- how temporary staged payloads are managed and cleaned up; and
- checkpoint and resume guarantees.

## Acceptance criteria

- `memory_mode="mixed"` has documented, implemented semantics and no longer
  fails at configuration time.
- At least one large force-training workflow reduces peak accelerator memory
  relative to GPU mode without changing numerical results beyond documented
  tolerances.
- HDF5-backed and in-memory datasets follow clearly defined transfer and cache
  behavior.
- Temporary resources are cleaned up deterministically after success or
  failure.
- Tests cover transfer behavior, training correctness, resource cleanup, and
  checkpoint/resume behavior.
- User and API documentation explain supported configurations and limitations.

## Notes

- Completed Issue 27 made the current fail-fast API decision; this issue covers
  the actual implementation.
- The legacy implementation is prior art, but the new design should fit the
  current dataset and trainer architecture rather than reproduce the old batch
  preparation layer verbatim.
- Correctness and explicit semantics take priority over aggressive optimization
  or distributed-training support in the first implementation.

