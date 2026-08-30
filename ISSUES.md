# Global Issues

This file is the high-level index of active, repository-wide issues. Detailed
descriptions live under [`issues/`](issues/). Local implementation work is
tracked separately in the untracked `LOCAL_ISSUES.md` file.

**Last issue created:** 38

**Next issue ID:** 39

## Active issues

| ID | Priority | Summary |
| --- | --- | --- |
| [13](issues/13-hdf5-force-data-coverage.md) | Medium | Add build-time controls for persisted force-data coverage in HDF5 datasets |
| [28](issues/28-mixed-memory-training.md) | Medium | Implement a real mixed-memory training mode for large force-training datasets |
| [36](issues/36-structure-library-generation.md) | Medium | Provide an end-to-end structure-library generation workflow |
| [37](issues/37-representative-structure-sampling.md) | Medium | Implement representative and random structure sampling |
| [38](issues/38-taylor-expansion-force-sampling.md) | Medium | Implement force-informed local Taylor sampling for ANN training |

## Legacy migration notes

- The previous local-only closed tracker is preserved unchanged at
  [`closed-issues/CLOSED_ISSUES.md`](closed-issues/CLOSED_ISSUES.md).
- Legacy ID 30 was assigned twice. ID 30 is retained for the completed HDF5
  worker-handle cleanup bug recorded in the closed tracker. The reference-energy
  helper umbrella that also used ID 30 is considered completed and superseded
  by completed Issues 34 and 35; it is not an active issue and has not been
  assigned a replacement ID.
