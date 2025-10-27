# Test Data Fixtures

This directory contains minimal test fixtures for the `torch_train` module tests.

## Files

### `sample.h5`
- **Format**: HDF5 training set (aenet-python format)
- **Purpose**: Test HDF5 format detection and loading
- **Contents**: Small training set with a few structures
- **Usage**: Used to test the `HDF5Loader` adapter

### `sample.train`
- **Format**: Binary training set (Fortran format)
- **Purpose**: Test binary format detection and loading with aenet-pytorch
- **Contents**: Training set in original aenet binary format
- **Usage**: Used to test the `BinaryLoader` adapter

### `sample.train.ascii`
- **Format**: ASCII training set (converted format)
- **Purpose**: Test ASCII format detection and loading
- **Contents**: ASCII representation of training set data
- **Usage**: Used to test the `ASCIILoader` adapter

### `features_with_neighbors.h5`
- **Format**: HDF5 training set generated with the PyTorch implementation
- **Purpose**: Test reading of neighbor information
- **Contents**: Small training set with a few structures
- **Usage**: Used to test the `HDF5Loader` adapter

## Maintenance

These test fixtures should remain small (< 1MB each) to keep tests fast and the repository lean. If you need to update them:

1. Ensure they contain valid, minimal data
2. Test that all three formats work with their respective loaders
3. Document any changes to the structure or content

## Generation

These fixtures were created from example aenet training sets. To regenerate or modify:

```python
from aenet.trainset import TrnSet

# For HDF5: Load and save subset of structures
with TrnSet.from_file("source.h5") as trnset:
    # ... extract subset and save
    trnset.to_hdf5("sample.h5")
```

For binary/ASCII formats, use aenet's `generate.x` and conversion tools.
