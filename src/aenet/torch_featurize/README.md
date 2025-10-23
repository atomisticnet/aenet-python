# PyTorch Featurization Module

This module provides GPU-accelerated implementations of neighbor list construction, Chebyshev polynomial evaluation, and atomic featurization for aenet.

## Phase 1: Neighbor List (COMPLETED)

### Implementation Status

✅ **Core Implementation Complete**
- `TorchNeighborList` class with full PBC and isolated system support
- GPU acceleration via torch_cluster
- Double precision (float64) support
- Comprehensive unit tests
- Validation and benchmarking scripts

### Features

- **Isolated Systems**: Fast neighbor finding for molecules
- **Periodic Boundary Conditions**: Support for arbitrary (triclinic) cells
- **GPU Acceleration**: Leverages CUDA for large systems
- **Flexible PBC**: Support for partial PBC (e.g., periodic in XY, not Z)
- **High Precision**: Double precision (torch.float64) for accuracy

### Installation

#### Basic Installation (CPU only)

```bash
pip install torch>=2.0.0
```

#### GPU Support

For GPU-accelerated neighbor finding, install torch_cluster:

```bash
# For CUDA 11.8
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# For CUDA 12.1
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# For CPU only
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

Replace `torch-2.1.0` with your installed PyTorch version and `cu118`/`cu121` with your CUDA version.

### Usage

#### Isolated System (Molecule)

```python
import torch
from aenet.torch_featurize import TorchNeighborList

# Water molecule positions (Cartesian coordinates in Angstroms)
positions = torch.tensor([
    [0.00000, 0.00000, 0.11779],   # O
    [0.00000, 0.75545, -0.47116],  # H
    [0.00000, -0.75545, -0.47116]  # H
], dtype=torch.float64)

# Create neighbor list
nbl = TorchNeighborList(cutoff=4.0, device='cpu')
result = nbl.get_neighbors(positions)

# Access results
edge_index = result['edge_index']  # (2, num_edges) neighbor pairs
distances = result['distances']    # (num_edges,) distances in Angstroms
num_neighbors = result['num_neighbors']  # (N,) neighbors per atom
```

#### Periodic System (Crystal)

```python
import torch
from aenet.torch_featurize import TorchNeighborList

# Fractional coordinates [0, 1)
positions = torch.rand(100, 3, dtype=torch.float64)

# Lattice vectors as rows (in Angstroms)
cell = torch.tensor([
    [10.0, 0.0, 0.0],
    [0.0, 10.0, 0.0],
    [0.0, 0.0, 10.0]
], dtype=torch.float64)

# Create neighbor list
nbl = TorchNeighborList(cutoff=5.0, device='cpu')
result = nbl.get_neighbors(positions, cell=cell)

# Access results
edge_index = result['edge_index']  # (2, num_edges)
distances = result['distances']    # (num_edges,)
offsets = result['offsets']        # (num_edges, 3) cell offsets
```

#### GPU Acceleration

```python
import torch
from aenet.torch_featurize import TorchNeighborList

# Create data on GPU
positions = torch.randn(1000, 3, dtype=torch.float64, device='cuda')

# Create neighbor list on GPU
nbl = TorchNeighborList(cutoff=3.0, device='cuda')
result = nbl.get_neighbors(positions)

# All outputs are on GPU
assert result['edge_index'].device.type == 'cuda'
assert result['distances'].device.type == 'cuda'
```

### Testing

Run unit tests:

```bash
# Run all tests
pytest src/aenet/torch_featurize/tests/test_neighborlist.py -v

# Run specific test class
pytest src/aenet/torch_featurize/tests/test_neighborlist.py::TestNeighborListIsolated -v

# Run GPU tests (if CUDA available)
pytest src/aenet/torch_featurize/tests/test_neighborlist.py::TestNeighborListGPU -v
```

### Validation

Validate against current implementation:

```bash
python src/aenet/torch_featurize/tests/validate_against_current.py
```

### Performance Benchmarking

Run performance benchmarks:

```bash
python src/aenet/torch_featurize/tests/benchmark_neighborlist.py
```

Expected speedups:
- **CPU**: 3-10× faster than current implementation
- **GPU**: 50-100× faster for large systems (>1000 atoms)

### Output Format

The `get_neighbors()` method returns a dictionary with:

- `edge_index`: `(2, num_edges)` tensor
  - Row 0: source atom indices
  - Row 1: target atom indices
  - Example: `edge_index[:, 0] = [0, 1]` means atom 0 → atom 1

- `distances`: `(num_edges,)` tensor
  - Pairwise distances in Angstroms
  - Corresponds to edges in `edge_index`

- `offsets`: `(num_edges, 3)` tensor or `None`
  - Cell offsets for periodic systems
  - `None` for isolated systems
  - Example: `[1, 0, -1]` means +1 cell in x, 0 in y, -1 in z

- `num_neighbors`: `(N,)` tensor
  - Number of neighbors for each atom
  - Useful for debugging and statistics

### API Reference

#### `TorchNeighborList`

```python
class TorchNeighborList:
    def __init__(self, cutoff: float, device: str = 'cpu',
                 dtype: torch.dtype = torch.float64)
```

**Parameters:**
- `cutoff`: Interaction cutoff radius in Angstroms
- `device`: 'cpu' or 'cuda'
- `dtype`: torch.float32 or torch.float64 (recommended: float64)

**Methods:**

```python
def get_neighbors(positions, cell=None, pbc=None) -> dict
```

**Parameters:**
- `positions`: `(N, 3)` tensor
  - Cartesian coordinates for isolated systems
  - Fractional coordinates [0, 1) for periodic systems
- `cell`: `(3, 3)` tensor or `None`
  - Lattice vectors as rows
  - `None` for isolated systems
- `pbc`: `(3,)` boolean tensor or `None`
  - Periodic boundary conditions in each direction
  - Default: `[True, True, True]` if cell is provided

**Returns:**
Dictionary with keys: `edge_index`, `distances`, `offsets`, `num_neighbors`

### Technical Details

#### Neighbor Finding Algorithm

1. **Isolated Systems**: Direct radius search using torch_cluster.radius_graph
2. **Periodic Systems**:
   - Determine required periodic images based on cutoff
   - Create replicated positions for periodic images
   - Find neighbors in extended system
   - Filter to central cell and compute cell offsets

#### Precision

- Uses `torch.float64` by default for numerical accuracy
- Matches Fortran implementation to ~1e-12 tolerance
- Can use `torch.float32` for faster computation if needed

#### Memory Efficiency

For periodic systems with large cutoffs, memory usage scales with the number of periodic images. The implementation:
- Dynamically determines minimum required periodic images
- Creates replicated positions only once
- Filters results efficiently on GPU

### Next Steps

Phase 1 (Neighbor List) is complete. Next phases:
- **Phase 2**: Chebyshev polynomial evaluation
- **Phase 3**: Full featurization pipeline
- **Phase 4**: Gradient support (optional)

See `PHASE2_CHEBYSHEV_POLYNOMIALS.md` in the project root for details.

### Contributing

When making changes:
1. Run unit tests: `pytest src/aenet/torch_featurize/tests/`
2. Run validation: `python src/aenet/torch_featurize/tests/validate_against_current.py`
3. Update tests if adding new features
4. Maintain double precision by default

### References

- **torch_cluster**: https://github.com/rusty1s/pytorch_cluster
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **AENET**: http://ann.atomistic.net/
