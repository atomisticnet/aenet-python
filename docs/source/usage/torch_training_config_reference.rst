Training Configuration Reference
==================================

This reference documents all parameters of the
:class:`~aenet.torch_training.TorchTrainingConfig` class used to configure
PyTorch-based training of machine learning interatomic potentials.

Overview
--------

The ``TorchTrainingConfig`` class centralizes all training parameters with
built-in validation. Parameters are organized by category:

1. **Training basics**: Core training settings
2. **Force training**: Force supervision parameters
3. **Performance & caching**: Optimization strategies
4. **DataLoader**: Parallel data loading settings
5. **Memory & precision**: Resource management
6. **Energy normalization**: Target space and preprocessing
7. **Output & monitoring**: Logging and diagnostics

Quick Reference
---------------

Common configurations by use case:

.. code-block:: python

   from aenet.torch_training import TorchTrainingConfig, Adam

   # Minimal energy-only training
   config = TorchTrainingConfig(
       iterations=100,
       force_weight=0.0
   )

   # Force training with default settings
   config = TorchTrainingConfig(
       iterations=100,
       force_weight=0.1
   )

   # Optimized force training (large dataset)
   config = TorchTrainingConfig(
       iterations=100,
       force_weight=0.1,
       force_fraction=0.3,
       cache_neighbors=True,
       cache_triplets=True,
       num_workers=8
   )

Training Basics
---------------

iterations
~~~~~~~~~~

**Type**: ``int``

**Default**: ``100``

**Description**: Maximum number of training epochs. One epoch processes all
structures in the training set once.

**Guidelines**:

* Small datasets (< 100 structures): 100-200 epochs
* Medium datasets (100-500 structures): 50-100 epochs
* Large datasets (> 500 structures): 20-50 epochs
* Monitor validation metrics; stop if no improvement after many epochs

.. code-block:: python

   config = TorchTrainingConfig(iterations=100)

method
~~~~~~

**Type**: ``TrainingMethod`` (``Adam`` or ``SGD``)

**Default**: ``Adam()`` with default parameters

**Description**: Optimization algorithm and its parameters.

**Available methods**:

* **Adam** (recommended): Adaptive learning rate optimizer

  .. code-block:: python

     from aenet.torch_training import Adam

     method = Adam(
         mu=0.001,           # Learning rate
         batchsize=32,       # Structures per batch
         beta1=0.9,          # First moment decay
         beta2=0.999,        # Second moment decay
         epsilon=1e-8,       # Numerical stability
         weight_decay=0.0    # L2 regularization
     )

* **SGD**: Stochastic gradient descent with momentum

  .. code-block:: python

     from aenet.torch_training import SGD

     method = SGD(
         lr=0.01,            # Learning rate
         batchsize=32,
         momentum=0.9,       # Momentum coefficient
         weight_decay=0.0
     )

**Recommendations**:

* Use Adam for most applications (robust convergence)
* SGD may work better for some systems with tuned hyperparameters
* Start with default Adam parameters, adjust learning rate if needed

testpercent
~~~~~~~~~~~

**Type**: ``int``

**Default**: ``10``

**Range**: 0-100

**Description**: Percentage of structures to hold out for validation (test set).
The remaining structures form the training set.

**Guidelines**:

* Use 10-20% for most cases
* Larger test sets (20-30%) for small datasets (< 100 structures)
* Smaller test sets (5-10%) for very large datasets (> 1000 structures)
* 0% uses all data for training (not recommended, no validation)

.. code-block:: python

   config = TorchTrainingConfig(testpercent=10)

Force Training Parameters
--------------------------

force_weight
~~~~~~~~~~~~

**Type**: ``float``

**Default**: ``0.0``

**Range**: 0.0-1.0

**Description**: Weight (α) balancing energy and force contributions to the loss:

.. math::

   \text{Loss} = (1 - \alpha) \cdot \text{RMSE}_{\text{energy}} + \alpha \cdot \text{RMSE}_{\text{forces}}

**Common values**:

* ``0.0``: Energy-only training (fastest)
* ``0.1``: Primarily energy with force regularization (typical)
* ``0.5``: Equal energy and force weighting
* ``1.0``: Force-only training (rarely used)

**Guidelines**:

* Start with 0.1 for most applications
* Increase if force accuracy is critical (e.g., MD simulations)
* Energy-only (0.0) is sufficient for many property predictions

.. code-block:: python

   config = TorchTrainingConfig(force_weight=0.1)

force_fraction
~~~~~~~~~~~~~~

**Type**: ``float``

**Default**: ``1.0``

**Range**: 0.0-1.0

**Description**: Fraction of structures (or atoms) to use for force supervision
each epoch. Reduces computational cost of force training.

**Performance impact**:

* ``force_fraction=1.0``: Use all forces (baseline)
* ``force_fraction=0.3``: ~3× speedup in force computation
* ``force_fraction=0.1``: ~10× speedup in force computation

**Convergence**:

Research has shown that ``force_fraction=0.1-0.3`` maintains good convergence
while significantly reducing training time [1,2]. The loss is rescaled to
remain unbiased with respect to the full supervision.

**Recommendations**:

* Start with 0.5-0.7 for initial experiments
* Use 0.1-0.3 for production training (validated effective range)
* Monitor validation metrics to ensure convergence quality

.. code-block:: python

   config = TorchTrainingConfig(
       force_weight=0.1,
       force_fraction=0.3  # Use 30% of forces per epoch
   )

**References**:

[1] López-Zorrilla, J.M., et al. "ænet-PyTorch: a GPU-supported implementation
for machine learning atomic potentials training." *J. Chem. Phys.* **158**,
164105 (2023). https://doi.org/10.1063/5.0146803

[2] Yeu, I.W., et al. "Scalable training of neural network potentials for
complex interfaces through data augmentation." *npj Comput Mater* **11**, 156
(2025). https://doi.org/10.1038/s41524-025-01651-0

force_sampling
~~~~~~~~~~~~~~

**Type**: ``Literal['random', 'fixed']``

**Default**: ``'random'``

**Description**: Strategy for selecting structures/atoms for force supervision
when ``force_fraction < 1.0``.

**Options**:

* ``'random'``: Randomly resample each epoch (recommended)
* ``'fixed'``: Use fixed subset throughout training

**Guidelines**:

* Use ``'random'`` for better generalization
* ``'fixed'`` may be useful for debugging or comparing runs

.. code-block:: python

   config = TorchTrainingConfig(
       force_fraction=0.3,
       force_sampling='random'
   )

force_resample_each_epoch
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Type**: ``bool``

**Default**: ``True``

**Description**: When ``force_sampling='random'``, resample the subset at the
beginning of each epoch. If ``False``, samples once at initialization.

**Recommendation**: Keep as ``True`` for better coverage of force data.

.. code-block:: python

   config = TorchTrainingConfig(
       force_fraction=0.3,
       force_sampling='random',
       force_resample_each_epoch=True
   )

force_min_structures_per_epoch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Type**: ``int`` or ``None``

**Default**: ``1``

**Description**: Minimum number of force-labeled structures to include per
epoch, regardless of ``force_fraction``. Prevents completely excluding force
data for small fractions.

**Recommendation**: Keep default value.

force_scale_unbiased
~~~~~~~~~~~~~~~~~~~~

**Type**: ``bool``

**Default**: ``False``

**Description**: If ``True``, apply ``sqrt(1/f)`` scaling to force RMSE to
approximate constant scale under sub-sampling.

**Recommendation**: Keep default (``False``). The loss is already scaled
appropriately.

Performance & Caching
---------------------

cached_features
~~~~~~~~~~~~~~~

**Type**: ``bool``

**Default**: ``False``

**Description**: Precompute and cache features for all structures before
training. Provides massive speedup for energy-only training.

**Performance impact**:

* **Energy-only training**: 100× or more speedup
* **Force training**: Not applicable (requires on-the-fly gradients)

**Memory impact**: Proportional to dataset size (n_structures × n_atoms × n_features)

**Restrictions**:

* Only works when ``force_weight=0.0`` (energy-only)
* Cannot be used with force training

**Recommendation**: Always enable for energy-only training if memory allows.

.. code-block:: python

   # Energy-only with caching
   config = TorchTrainingConfig(
       force_weight=0.0,
       cached_features=True  # 100× speedup
   )

cached_features_for_force
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Type**: ``bool``

**Default**: ``False``

**Description**: In mixed energy/force training with partial force supervision
(``force_fraction < 1.0``), cache computed features for structures that are NOT
selected for force supervision in the current epoch.

**How it works**:

* When ``force_fraction=0.3``, only 30% of force-labeled structures are used for
  force gradients each epoch
* The remaining 70% are used only for energy prediction
* With ``cached_features_for_force=True``, features for these energy-only
  structures are cached and reused across batches within the epoch/window

**Performance impact**:

* Modest speedup (≈ 5–15%) when ``force_fraction`` is significantly below 1.0
* Memory usage increases proportionally to the number of cached structures
* Works in both in-memory and HDF5-backed datasets

**What is NOT cached**:

* Gradients are NOT cached (they change every optimizer step)
* Only forward-pass features are cached
* Neighbor information is recomputed unless ``cache_neighbors=True``

**Recommendations**:

* Keep default (``False``) unless using small ``force_fraction`` (e.g., 0.1–0.3)
  with many epochs and you observe repeated computation for energy-only samples
* Not applicable for energy-only training — use ``cached_features`` instead

cache_neighbors
~~~~~~~~~~~~~~~

**Type**: ``bool``

**Default**: ``False``

**Description**: Cache per-structure neighbor graphs (indices and displacement
vectors) to avoid repeated neighbor searches across epochs. Applicable when
geometries are fixed during training.

**Performance impact**:

* 6-20% speedup for force training
* Most beneficial with multiple epochs (3+)

**Memory impact**: Proportional to number of edges (neighbors) per structure

**Restrictions**:

* Only useful when training on fixed structures
* Not applicable for on-the-fly augmentation

**Recommendation**: Enable for force training with epochs ≥ 3.

.. code-block:: python

   config = TorchTrainingConfig(
       force_weight=0.1,
       cache_neighbors=True
   )

cache_triplets
~~~~~~~~~~~~~~

**Type**: ``bool``

**Default**: ``False``

**Description**: Build and cache CSR neighbor graphs and precomputed angular
triplet indices per structure. Enables fully vectorized featurization and
gradient computation.

**Performance impact**:

* Removes Python-level enumeration overhead
* Most beneficial with higher ``ang_order`` or ``ang_cutoff``
* Works synergistically with ``cache_neighbors``

**Memory impact**: Additional storage for triplet indices

**Recommendation**: Enable together with ``cache_neighbors`` for force training.

.. code-block:: python

   config = TorchTrainingConfig(
       force_weight=0.1,
       cache_neighbors=True,
       cache_triplets=True  # Vectorized operations
   )

.. important::

   **What is NOT cached**: Gradients are NOT cached by any of these parameters.

   * Gradients are model-parameter-specific and change every optimizer step
   * They are computed on-demand via PyTorch's autograd during backpropagation
   * Caching gradients would provide no benefit and would waste memory

   The caching parameters focus on:

   * **Input structures** (neighbors, triplets) that enable efficient gradient computation
   * **Forward features** for energy-only prediction paths
   * **NOT** the gradients themselves

cache_persist_dir
~~~~~~~~~~~~~~~~~

**Type**: ``str`` or ``None``

**Default**: ``None``

**Description**: Optional directory for persisting caches to disk (future feature).

**Recommendation**: Keep default (``None``) unless using experimental persistence.

cache_scope
~~~~~~~~~~~

**Type**: ``Literal['train', 'val', 'all']``

**Default**: ``'all'``

**Description**: Which dataset split(s) should be cached/persisted.

**Options**:

* ``'train'``: Cache training set only
* ``'val'``: Cache validation set only
* ``'all'``: Cache both splits

**Recommendation**: Keep default (``'all'``).

epochs_per_force_window
~~~~~~~~~~~~~~~~~~~~~~~

**Type**: ``int``

**Default**: ``1``

**Description**: Resample the random subset of force-supervised structures every
this many epochs (when ``force_sampling='random'``). Values > 1 amortize cached
features across multiple epochs.

**Recommendation**: Keep default (``1``) for standard use.

DataLoader Settings
-------------------

These parameters control parallel data loading, which is beneficial for large
datasets when using on-the-fly featurization.

num_workers
~~~~~~~~~~~

**Type**: ``int``

**Default**: ``0``

**Description**: Number of parallel worker processes for data loading.

**Performance impact**:

* **Small datasets** (< 128 structures): Workers add overhead; use ``0``
* **Large datasets** (≥ 128 structures): 2-3× speedup with 4-8 workers

**Platform considerations**:

* macOS multiprocessing overhead can negate benefits for small workloads
* Linux typically shows better scaling

**Recommendation**:

* Small datasets: ``num_workers=0``
* Large datasets: ``num_workers=4`` to ``8`` (≈ physical cores / 2)
* Experiment to find optimal value for your system

.. code-block:: python

   # Large dataset with parallel loading
   config = TorchTrainingConfig(
       iterations=100,
       num_workers=8,
       prefetch_factor=4
   )

.. note::

   The benefits of parallel data loading only apply to on-the-fly featurization.
   When ``cached_features=True``, data loading is negligible and workers provide
   no benefit.

prefetch_factor
~~~~~~~~~~~~~~~

**Type**: ``int``

**Default**: ``2``

**Description**: Number of batches to prefetch per worker. Only used when
``num_workers > 0``.

**Recommendation**:

* Use 2-4 for most cases
* Higher values use more memory for minimal additional benefit

.. code-block:: python

   config = TorchTrainingConfig(
       num_workers=8,
       prefetch_factor=4  # Prefetch 4 batches per worker
   )

persistent_workers
~~~~~~~~~~~~~~~~~~

**Type**: ``bool``

**Default**: ``True``

**Description**: Keep worker processes alive between epochs. Only used when
``num_workers > 0``.

**Benefits**:

* Avoids overhead of spawning workers each epoch
* Particularly beneficial for training with many epochs

**Recommendation**: Keep default (``True``) when using workers.

.. code-block:: python

   config = TorchTrainingConfig(
       num_workers=8,
       persistent_workers=True
   )

Memory & Precision
------------------

device
~~~~~~

**Type**: ``str`` or ``None``

**Default**: ``None`` (auto-detect)

**Description**: Device for training: ``'cpu'``, ``'cuda'``, or ``'cuda:0'``,
``'cuda:1'``, etc.

**Auto-detection**: If ``None``, automatically uses CUDA if available, otherwise CPU.

**Recommendation**:

* Let auto-detection handle device selection
* Explicitly set to ``'cpu'`` to force CPU usage
* Use ``'cuda:N'`` to select specific GPU

.. code-block:: python

   # Auto-detect (recommended)
   config = TorchTrainingConfig(device=None)

   # Force CPU
   config = TorchTrainingConfig(device='cpu')

   # Specific GPU
   config = TorchTrainingConfig(device='cuda:0')

precision
~~~~~~~~~

**Type**: ``Literal['auto', 'float32', 'float64']``

**Default**: ``'auto'``

**Description**: Numeric precision for training.

**Auto behavior**:

* CPU: Uses ``float32`` (single precision)
* GPU: Uses ``float64`` (double precision)

**Trade-offs**:

* ``float32``: Lower memory usage, faster on GPU, may have numerical issues
* ``float64``: Higher accuracy, more memory, slightly slower

**Recommendation**:

* Use ``'auto'`` (default) for most cases
* Force ``float64`` if you encounter numerical instability
* Use ``float32`` to reduce memory usage on GPU

.. code-block:: python

   # Auto-select based on device
   config = TorchTrainingConfig(precision='auto')

   # Force double precision
   config = TorchTrainingConfig(precision='float64')

memory_mode
~~~~~~~~~~~

**Type**: ``Literal['cpu', 'gpu', 'mixed']``

**Default**: ``'gpu'``

**Description**: Memory management strategy.

**Options**:

* ``'cpu'``: Keep data on CPU
* ``'gpu'``: Keep data on GPU
* ``'mixed'``: Hybrid approach (advanced)

**Recommendation**: Keep default unless you have specific memory constraints.

Energy Normalization
--------------------

These parameters control preprocessing of energies before training.

energy_target
~~~~~~~~~~~~~

**Type**: ``Literal['cohesive', 'total']``

**Default**: ``'cohesive'``

**Description**: Target energy space for training.

**Options**:

* ``'cohesive'``: Train on cohesive energies (relative to atomic references)
* ``'total'``: Train on total energies

**Recommendation**:

* Use ``'cohesive'`` for most applications
* Requires ``E_atomic`` to be specified

.. code-block:: python

   config = TorchTrainingConfig(
       energy_target='cohesive',
       E_atomic={'O': -432.1, 'H': -13.6}  # eV
   )

E_atomic
~~~~~~~~

**Type**: ``Dict[str, float]`` or ``None``

**Default**: ``None``

**Description**: Atomic reference energies for each species (in eV). Required
when ``energy_target='cohesive'``.

**Guidelines**:

* Use DFT-computed isolated atom energies
* Must include all species in your structures
* Energies should match your DFT method

.. code-block:: python

   E_atomic = {
       'Ti': -1604.57,
       'O': -432.10
   }

   config = TorchTrainingConfig(
       energy_target='cohesive',
       E_atomic=E_atomic
   )

normalize_features
~~~~~~~~~~~~~~~~~~

**Type**: ``bool``

**Default**: ``True``

**Description**: Standardize features to zero mean and unit variance.

**Recommendation**: Keep default (``True``) for stable training.

normalize_energy
~~~~~~~~~~~~~~~~

**Type**: ``bool``

**Default**: ``True``

**Description**: Normalize energies (per-atom shift and scaling).

**Recommendation**: Keep default (``True``) for better convergence.

feature_stats
~~~~~~~~~~~~~

**Type**: ``Dict[str, Any]`` or ``None``

**Default**: ``None``

**Description**: Pre-computed feature statistics (mean, std). If ``None``,
computed from training set.

**Format**:

.. code-block:: python

   feature_stats = {
       'mean': np.array([...]),  # Shape: (n_features,)
       'std': np.array([...])     # Shape: (n_features,)
   }

**Recommendation**: Let the training code compute statistics automatically.

E_shift
~~~~~~~

**Type**: ``float`` or ``None``

**Default**: ``None``

**Description**: Per-atom energy shift for normalization. If ``None``, computed
from training set.

**Recommendation**: Let the training code compute automatically.

E_scaling
~~~~~~~~~

**Type**: ``float`` or ``None``

**Default**: ``None``

**Description**: Energy scaling factor for normalization. If ``None``, computed
from training set.

**Recommendation**: Let the training code compute automatically.

Output & Monitoring
-------------------

max_energy
~~~~~~~~~~

**Type**: ``float`` or ``None``

**Default**: ``None``

**Description**: Exclude structures with energy above this threshold (per atom).

**Use case**: Filter outliers or high-energy configurations.

.. code-block:: python

   config = TorchTrainingConfig(
       max_energy=10.0  # Exclude structures with E > 10 eV/atom
   )

max_forces
~~~~~~~~~~

**Type**: ``float`` or ``None``

**Default**: ``None``

**Description**: Exclude structures with maximum force above this threshold.

**Use case**: Filter structures with unrealistic forces.

.. code-block:: python

   config = TorchTrainingConfig(
       max_forces=50.0  # Exclude structures with F_max > 50 eV/Å
   )

save_energies
~~~~~~~~~~~~~

**Type**: ``bool``

**Default**: ``False``

**Description**: Save predicted energies for training and test sets to files
(``energies.train.predicted``, ``energies.test.predicted``).

**Recommendation**: Enable for detailed analysis of predictions.

.. code-block:: python

   config = TorchTrainingConfig(save_energies=True)

save_forces
~~~~~~~~~~~

**Type**: ``bool``

**Default**: ``False``

**Description**: Save predicted forces for training and test sets to files
(``forces.train.predicted``, ``forces.test.predicted``).

**Recommendation**: Enable for force training analysis.

.. code-block:: python

   config = TorchTrainingConfig(
       force_weight=0.1,
       save_forces=True
   )

timing
~~~~~~

**Type**: ``bool``

**Default**: ``False``

**Description**: Enable detailed timing output for performance profiling.

**Use case**: Identify bottlenecks during training.

**Recommendation**: Enable when optimizing performance, disable for production.

.. code-block:: python

   config = TorchTrainingConfig(timing=True)

show_progress
~~~~~~~~~~~~~

**Type**: ``bool``

**Default**: ``True``

**Description**: Display progress bar during training.

**Recommendation**: Keep enabled for interactive use, disable for batch scripts.

.. code-block:: python

   config = TorchTrainingConfig(show_progress=True)

show_batch_progress
~~~~~~~~~~~~~~~~~~~

**Type**: ``bool``

**Default**: ``False``

**Description**: Display progress bar for individual batches within each epoch.

**Recommendation**: Enable only for debugging; adds visual clutter.

.. code-block:: python

   config = TorchTrainingConfig(
       show_progress=True,
       show_batch_progress=True  # Verbose output
   )

Configuration Examples
----------------------

Energy-Only Training (Small Dataset)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from aenet.torch_training import TorchTrainingConfig, Adam

   config = TorchTrainingConfig(
       iterations=200,
       method=Adam(mu=0.001, batchsize=16),
       testpercent=20,
       force_weight=0.0,
       device='cpu'
   )

Energy-Only Training (Large Dataset, Optimized)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = TorchTrainingConfig(
       iterations=50,
       method=Adam(mu=0.001, batchsize=64),
       testpercent=10,
       force_weight=0.0,
       cached_features=True,  # 100× speedup
       device='cuda'
   )

Force Training (Standard)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = TorchTrainingConfig(
       iterations=100,
       method=Adam(mu=0.001, batchsize=32),
       testpercent=10,
       force_weight=0.1,
       device='cuda',
       save_forces=True
   )

Force Training (Optimized for Large Dataset)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = TorchTrainingConfig(
       iterations=100,
       method=Adam(mu=0.001, batchsize=32),
       testpercent=10,
       force_weight=0.1,
       force_fraction=0.3,      # 3× speedup
       cache_neighbors=True,    # Cache neighbor lists
       cache_triplets=True,     # Vectorized operations
       num_workers=8,           # Parallel data loading
       prefetch_factor=4,
       device='cuda',
       save_energies=True,
       save_forces=True
   )

Production Training with All Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = TorchTrainingConfig(
       iterations=100,
       method=Adam(mu=0.001, batchsize=32, weight_decay=1e-5),
       testpercent=10,
       force_weight=0.1,
       force_fraction=0.3,
       force_sampling='random',
       cache_neighbors=True,
       cache_triplets=True,
       num_workers=8,
       prefetch_factor=4,
       persistent_workers=True,
       precision='auto',
       device='cuda',
       energy_target='cohesive',
       E_atomic={'Ti': -1604.57, 'O': -432.10},
       save_energies=True,
       save_forces=True,
       show_progress=True
   )

Dataset Selection for Large-Scale Training
-------------------------------------------

For datasets that don't fit in memory, use :class:`~aenet.torch_training.HDF5StructureDataset`
instead of loading structures directly. This provides lazy loading from an HDF5 database.

HDF5StructureDataset Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The HDF5-backed dataset stores serialized torch Structures in a compressed HDF5
file with per-entry metadata, enabling training on datasets with 10M+ structures.

**Key features**:

* Lazy loading: Only batch-sized chunks loaded into RAM at a time
* Multiprocessing-safe: Each DataLoader worker opens its own read-only handle
* Compressed storage: Uses zlib/blosc for 3–5× disk space reduction
* LRU in-memory cache: Configurable per-worker cache for hot entries
* Compatible with all caching parameters: ``cache_neighbors``, ``cache_triplets``, etc.

**When to use**:

* Dataset > 1 GB in-memory size
* > 1,000 structures with forces
* > 10,000 structures for energy-only training
* Distributed training across multiple machines (build once, share DB)

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   from aenet.torch_training import (
       HDF5StructureDataset,
       TorchANNPotential,
       TorchTrainingConfig
   )
   import aenet.io
   from pathlib import Path

   # Define parser (top-level function if using num_workers > 0)
   def parse_structure(path):
       # Handle single-frame or trajectory (pick first frame)
       s = aenet.io.read(path).to_TorchStructure()
       return s[0] if isinstance(s, list) else s

   # Build HDF5 database (one-time operation)
   file_paths = [str(p) for p in Path("data").glob("*.xsf")]

   ds = HDF5StructureDataset(
       descriptor=descriptor,
       database_file="structures.h5",
       file_paths=file_paths,
       parser=parse_structure,
       mode="build",              # Create database
       compression="zlib",
       compression_level=5,
       force_fraction=1.0,
       force_sampling="random",
       cache_neighbors=True,      # Per-structure neighbor caching
       cache_triplets=True,       # Vectorized operations
   )
   ds.build_database(show_progress=True)

   # Later: Load and train (can be on different machine)
   ds = HDF5StructureDataset(
       descriptor=descriptor,
       database_file="structures.h5",
       mode="load",               # Open existing database
       force_fraction=0.3,        # Override force sampling for training
       cache_neighbors=True,
       cache_triplets=True,
   )

   config = TorchTrainingConfig(
       iterations=100,
       force_weight=0.1,
       num_workers=8,             # Parallel loading
       prefetch_factor=4,
       persistent_workers=True,
       device="cuda"
   )

   pot = TorchANNPotential(arch, descriptor)
   history = pot.train(dataset=ds, config=config)

HDF5 Dataset Caching Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All standard caching parameters work with ``HDF5StructureDataset``:

* **cache_neighbors** (recommended): Cache neighbor lists per structure
* **cache_triplets** (recommended): Enable vectorized gradient computation
* **cached_features_for_force**: Cache features for energy-only samples within epoch
* **in_memory_cache_size** (default: 2048): LRU cache size for unpickled Structures

**Caching hierarchy**:

1. **HDF5 file** (disk): Permanent storage, accessed via memory mapping
2. **LRU Structure cache** (RAM): Recently accessed structures (per worker)
3. **Neighbor/triplet cache** (RAM): Computed once, reused across epochs
4. **Feature cache** (RAM): Only when ``cached_features_for_force=True``

Performance Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For optimal performance with large HDF5 datasets:

.. code-block:: python

   # Optimized configuration for large-scale force training
   config = TorchTrainingConfig(
       iterations=100,
       force_weight=0.1,
       force_fraction=0.3,         # Reduce force computation cost
       num_workers=8,              # Parallel I/O (≈ CPU cores / 2)
       prefetch_factor=4,          # Pipeline batches
       persistent_workers=True,    # Reuse worker processes
       device="cuda"
   )

   ds = HDF5StructureDataset(
       descriptor=descriptor,
       database_file="structures.h5",
       mode="load",
       force_fraction=0.3,
       cache_neighbors=True,       # Avoid repeated neighbor searches
       cache_triplets=True,        # Vectorized operations
       in_memory_cache_size=4096,  # Cache hot structures
   )

**Expected speedup** vs. in-memory + no caching:

* ``cache_neighbors``: +10–20%
* ``cache_triplets``: +15–30%
* Combined: +30–50% for force training

Parser Requirements
~~~~~~~~~~~~~~~~~~~

When using ``num_workers > 0``, the parser must be a **picklable top-level function**:

.. code-block:: python

   # ✓ GOOD: Top-level function
   def my_parser(path):
       s = aenet.io.read(path).to_TorchStructure()
       return s[0] if isinstance(s, list) else s

   # ✗ BAD: Lambda (not picklable)
   parser = lambda p: aenet.io.read(p).to_TorchStructure()[0]

   # ✗ BAD: Closure (captures local state)
   def make_parser(frame_idx):
       def parser(path):
           s = aenet.io.read(path).to_TorchStructure()
           return s[frame_idx] if isinstance(s, list) else s
       return parser

Summary of Key Parameters
--------------------------

This table summarizes the most important parameters and when to use them:

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Category
     - Default
     - When to Change
   * - ``iterations``
     - Basic
     - 100
     - Adjust based on dataset size and convergence
   * - ``force_weight``
     - Force
     - 0.0
     - Set to 0.1-0.5 for force training
   * - ``force_fraction``
     - Force
     - 1.0
     - Use 0.1-0.3 to speed up force training
   * - ``cached_features``
     - Performance
     - False
     - Enable for energy-only training (huge speedup)
   * - ``cache_neighbors``
     - Performance
     - False
     - Enable for force training with epochs ≥ 3
   * - ``cache_triplets``
     - Performance
     - False
     - Enable with ``cache_neighbors`` for vectorization
   * - ``num_workers``
     - DataLoader
     - 0
     - Use 4-8 for large datasets (≥128 structures)
   * - ``device``
     - Memory
     - None
     - Set to 'cuda' to explicitly use GPU
   * - ``precision``
     - Memory
     - 'auto'
     - Use 'float32' to reduce memory usage
   * - ``testpercent``
     - Basic
     - 10
     - Increase for small datasets (20-30%)

See Also
--------

* :doc:`torch_featurization` - Featurization parameters and settings
* :doc:`choosing_implementation` - Fortran vs PyTorch comparison
