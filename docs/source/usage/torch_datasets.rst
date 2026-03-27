PyTorch Dataset Options
========================

The PyTorch training workflow provides flexible dataset classes for different
use cases, from simple in-memory lists to large-scale HDF5-backed lazy-loading.
This page covers all available dataset options and their usage.

.. note::

   Datasets are used with :doc:`torch_training`. Make sure you understand
   the basic training workflow before diving into advanced dataset options.

Example notebook
----------------

For a file-backed training walkthrough using the TiO2 example data, explicit
``CachedStructureDataset`` objects, fixed train/test splits, and
dataset-backed ``predict_dataset()`` calls, see
`example-05-torch-training.ipynb
<https://github.com/atomisticnet/aenet-python/blob/master/notebooks/example-05-torch-training.ipynb>`_.

The ``.rst`` page below stays focused on compact API-facing examples, while the
notebook remains the home for the longer training workflow.

Structure Input Formats
-----------------------

All dataset classes accept structures in three formats:

1. **File paths** (``List[os.PathLike]``): Simplest option, recommended for most cases
2. **AtomicStructure objects** (``List[AtomicStructure]``): Use when you need to manipulate structures first
3. **torch Structure objects** (``List[Structure]``): Advanced option, direct PyTorch format

The conversion between formats happens automatically, so you can use whichever
is most convenient. The notebook above shows the most realistic file-backed
workflow; the compact page examples below use small in-memory structures.

.. code-block:: python

   from pathlib import Path
   from aenet.geometry import AtomicStructure
   from aenet.torch_training import Structure
   from aenet.torch_training.dataset import StructureDataset

   descriptor = ...  # Reuse your configured ChebyshevDescriptor

   # Option 1: file paths (simplest for real training runs)
   structure_paths = sorted(Path("xsf-TiO2").glob("*.xsf"))
   dataset = StructureDataset(structures=structure_paths, descriptor=descriptor)

   # Option 2: AtomicStructure objects (when you want to inspect or edit them)
   atomic_structures = [
       AtomicStructure.from_file(path) for path in structure_paths[:2]
   ]
   dataset = StructureDataset(
       structures=atomic_structures,
       descriptor=descriptor,
   )

   # Option 3: torch Structure objects (advanced / fully explicit)
   torch_structures = [
       structure
       for atomic in atomic_structures
       for structure in atomic.to_TorchStructure()
   ]
   dataset = StructureDataset(structures=torch_structures, descriptor=descriptor)

**Recommendation**: Use file paths (Option 1) for simplicity unless you have specific needs.


Dataset Classes Overview
------------------------

Three main dataset classes are available:

1. **StructureDataset**: On-the-fly featurization, supports force training
2. **CachedStructureDataset**: Pre-computed features for energy-only training (much faster)
3. **HDF5StructureDataset**: Lazy-loading for large datasets (10,000+ structures)


StructureDataset: On-the-Fly Featurization
-------------------------------------------

The default dataset for most use cases. Stores structures in memory and
computes features on-demand during training.

Basic Usage
~~~~~~~~~~~

.. doctest::

   >>> import numpy as np
   >>> from aenet.torch_featurize import ChebyshevDescriptor
   >>> from aenet.torch_training import Structure
   >>> from aenet.torch_training.dataset import StructureDataset
   >>> descriptor = ChebyshevDescriptor(
   ...     species=["H"],
   ...     rad_order=1,
   ...     rad_cutoff=2.0,
   ...     ang_order=0,
   ...     ang_cutoff=2.0,
   ...     min_cutoff=0.1,
   ...     device="cpu",
   ... )
   >>> structures = [
   ...     Structure(
   ...         positions=np.array(
   ...             [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.9, 0.0]]
   ...         ),
   ...         species=["H", "H", "H"],
   ...         energy=0.0,
   ...         forces=np.zeros((3, 3)),
   ...     ),
   ...     Structure(
   ...         positions=np.array(
   ...             [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
   ...         ),
   ...         species=["H", "H", "H"],
   ...         energy=0.5,
   ...         forces=np.zeros((3, 3)),
   ...     ),
   ... ]
   >>> dataset = StructureDataset(
   ...     structures=structures,
   ...     descriptor=descriptor,
   ... )
   >>> len(dataset)
   2
   >>> sample = dataset[0]
   >>> sample["features"].shape
   torch.Size([3, 3])
   >>> sample["use_forces"]
   True

Runtime Training Options
~~~~~~~~~~~~~~~~~~~~~~~~

``StructureDataset`` is now a passive data source. Force sampling and runtime
cache behavior live in ``TorchTrainingConfig``:

.. code-block:: python

   from aenet.torch_training import TorchTrainingConfig

   config = TorchTrainingConfig(
       force_weight=0.1,
       force_fraction=0.3,           # Use 30% of force-labeled structures
       force_sampling="random",      # Resample each epoch
       cache_features=True,          # Cache energy-view features
       cache_neighbors=True,         # Cache neighbor data when helpful
   )

**Parameters:**

- **force_fraction** (float, 0.0-1.0): Fraction of force structures to use. Using a subset (e.g., 0.3) can speed up training 3× while maintaining accuracy.
- **force_sampling** (str): ``'random'`` (resample each epoch) or ``'fixed'`` (static subset). Random provides better generalization.
- **cache_features** (bool): Cache features for structures not selected for force supervision in the current epoch. Useful with ``force_fraction < 1.0``.
- **cache_neighbors** (bool): Cache neighbor graphs to avoid repeated searches for energy-view reuse and legacy non-graph paths. Supported force training does not require this.
- **cache_force_triplets** (bool): Cache CSR graphs and triplets instead of rebuilding them on demand.

Manual Dataset Splitting
~~~~~~~~~~~~~~~~~~~~~~~~~

For full control over train/test splits:

.. doctest::

   >>> import numpy as np
   >>> from aenet.torch_featurize import ChebyshevDescriptor
   >>> from aenet.torch_training import Structure
   >>> from aenet.torch_training.dataset import StructureDataset, train_test_split
   >>> descriptor = ChebyshevDescriptor(
   ...     species=["H"],
   ...     rad_order=1,
   ...     rad_cutoff=2.0,
   ...     ang_order=0,
   ...     ang_cutoff=2.0,
   ...     min_cutoff=0.1,
   ...     device="cpu",
   ... )
   >>> structures = [
   ...     Structure(
   ...         positions=np.array(
   ...             [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.9, 0.0]]
   ...         ),
   ...         species=["H", "H", "H"],
   ...         energy=0.0,
   ...         forces=np.zeros((3, 3)),
   ...     ),
   ...     Structure(
   ...         positions=np.array(
   ...             [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
   ...         ),
   ...         species=["H", "H", "H"],
   ...         energy=0.5,
   ...         forces=np.zeros((3, 3)),
   ...     ),
   ... ]
   >>> dataset = StructureDataset(structures=structures, descriptor=descriptor)
   >>> train_ds, test_ds = train_test_split(
   ...     dataset,
   ...     test_fraction=0.5,
   ...     seed=42,
   ... )
   >>> (len(train_ds), len(test_ds))
   (1, 1)

Pass ``train_dataset=...`` and ``test_dataset=...`` to
``TorchANNPotential.train()`` when you want an explicit fixed split. The
notebook example above keeps the full file-backed training workflow.


CachedStructureDataset: Pre-Computed Features
----------------------------------------------

For energy-only training, features can be pre-computed once and cached for ~100× speedup. This is ideal when you don't need forces and want maximum training speed.

.. doctest::

   >>> import numpy as np
   >>> from aenet.torch_featurize import ChebyshevDescriptor
   >>> from aenet.torch_training import Structure
   >>> from aenet.torch_training.dataset import CachedStructureDataset
   >>> descriptor = ChebyshevDescriptor(
   ...     species=["H"],
   ...     rad_order=1,
   ...     rad_cutoff=2.0,
   ...     ang_order=0,
   ...     ang_cutoff=2.0,
   ...     min_cutoff=0.1,
   ...     device="cpu",
   ... )
   >>> structures = [
   ...     Structure(
   ...         positions=np.array(
   ...             [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.9, 0.0]]
   ...         ),
   ...         species=["H", "H", "H"],
   ...         energy=0.0,
   ...     ),
   ...     Structure(
   ...         positions=np.array(
   ...             [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
   ...         ),
   ...         species=["H", "H", "H"],
   ...         energy=0.5,
   ...     ),
   ... ]
   >>> dataset = CachedStructureDataset(
   ...     structures=structures,
   ...     descriptor=descriptor,
   ...     show_progress=False,
   ... )
   >>> dataset[0]["features"].shape
   torch.Size([3, 3])
   >>> dataset[0]["use_forces"]
   False

**When to use:**

- Energy-only training (``force_weight=0.0``)
- Multiple training runs with same data
- When training speed is critical
- Energy-only inference with ``TorchANNPotential.predict_dataset()``

**Automatic usage:**

The trainer automatically uses ``CachedStructureDataset`` when you pass
``structures`` with ``cache_features=True`` and ``force_weight=0.0``:

.. code-block:: python

   from aenet.torch_training import TorchTrainingConfig

   config = TorchTrainingConfig(
       iterations=100,
       force_weight=0.0,         # Energy-only required
       cache_features=True,      # Triggers CachedStructureDataset
   )

Pass this config to ``TorchANNPotential.train(structures=..., config=config)``
to take the automatic cached-features path. For an explicit
``CachedStructureDataset`` workflow with a fixed split and
``predict_dataset()``, see the training notebook linked above.

Explicit Fixed Splits with CachedStructureDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already know which structures belong in the training and test sets,
build explicit cached datasets and pass them to ``train()`` directly:

.. code-block:: python

   from aenet.torch_training import Adam, TorchANNPotential, TorchTrainingConfig
   from aenet.torch_training.dataset import CachedStructureDataset

   train_ds = CachedStructureDataset(
       structures=train_structures,
       descriptor=descriptor,
       show_progress=False,
   )
   test_ds = CachedStructureDataset(
       structures=test_structures,
       descriptor=descriptor,
       show_progress=False,
   )

   config = TorchTrainingConfig(
       iterations=100,
       method=Adam(mu=0.001, batchsize=32),
       force_weight=0.0,
       testpercent=0,  # split is already explicit
   )

   pot = TorchANNPotential(arch=arch, descriptor=descriptor)
   pot.train(train_dataset=train_ds, test_dataset=test_ds, config=config)

You can also wrap a cached dataset in ``torch.utils.data.Subset`` for manual
index-based splits, and cached feature reuse still works in that case.
However, ``CachedStructureDataset`` builds its cache for the full underlying
dataset before any ``Subset`` is applied. If you already know the split,
creating separate cached train/test datasets is usually more memory-efficient.


HDF5StructureDataset: Large-Scale Lazy-Loading
-----------------------------------------------

For very large datasets (10,000+ structures), use HDF5-backed lazy-loading to minimize memory usage. Structures are serialized to an HDF5 database once, then read on-demand during training.

Building the Database
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from aenet.torch_training.dataset import HDF5StructureDataset
   from aenet.geometry import AtomicStructure
   from glob import glob

   # Define the parser used during HDF5 database construction
   def parse_xsf(path: str):
       """Parse XSF file and return torch Structure(s)."""
       atomic_struct = AtomicStructure.from_file(path)
       torch_structs = atomic_struct.to_TorchStructure()
       # to_TorchStructure() may return list of frames
       return torch_structs if isinstance(torch_structs, list) else [torch_structs]

   # Build HDF5 database (do this once)
   file_list = glob("data/**/*.xsf", recursive=True)

   db = HDF5StructureDataset(
       descriptor=descriptor,
       database_file="datasets/training.h5",
       file_paths=file_list,
       parser=parse_xsf,
       mode="build",                # Build mode
       in_memory_cache_size=2048,   # LRU cache for unpickled structures
       compression="zlib",
       compression_level=5,
   )

   db.build_database(
       show_progress=True,
       build_workers=8,                 # optional build-time worker threads
       persist_descriptor=True,         # optional descriptor recovery step
       persist_features=True,           # optional persisted raw features
       persist_force_derivatives=True,  # optional sparse derivative cache
   )

   # ``db`` is immediately reusable after build_database(); reopening the
   # file is only needed in a later session or when you want a separate handle.

.. note::

   ``build_workers`` only affects the one-time ``build_database()`` call.
   It parallelizes parser execution and optional persisted-cache preparation
   with worker threads, while the parent process still performs all ordered
   HDF5 writes. This is
   separate from training-time ``num_workers`` on ``TorchTrainingConfig``.

.. note::

   ``persist_descriptor=True`` stores a small versioned descriptor manifest
   alongside the HDF5 training data so later ``mode="load"`` sessions can
   recover supported descriptor objects automatically. This is enabled
   automatically when ``persist_features=True`` or
   ``persist_force_derivatives=True``.

.. note::

   ``persist_features=True`` stores raw unnormalized ``(N, F)`` descriptor
   features in the HDF5 cache. During later HDF5-backed training runs,
   ``HDF5StructureDataset`` will reuse those persisted features lazily when
   they are descriptor-compatible. This sits between the trainer-owned
   ``cache_features=True`` runtime cache and full feature recomputation.

.. note::

   ``persist_force_derivatives=True`` stores the sparse local derivative
   payload for force-labeled structures in the HDF5 file under a documented,
   versioned schema. This is useful when preparing derivative caches for
   repeated fixed-geometry training workflows. During HDF5-based force
   training, the trainer now loads that payload lazily per sample and prefers
   it over on-the-fly sparse derivative recomputation when the cache is
   present and descriptor-compatible. When a force-labeled entry also has
   persisted raw features, the force path can reuse both persisted payloads
   directly. This is distinct from ``cache_force_triplets=True`` and
   ``cache_features=True``, which cache in-memory runtime data within a
   dataset instance and do not write those payloads to the HDF5 file. The
   schema is documented in
   :doc:`../dev/torch_force_hdf5_cache`.

Persisted Cache Semantics
~~~~~~~~~~~~~~~~~~~~~~~~~

``HDF5StructureDataset`` now has three distinct cache layers that serve
different purposes:

* ``persist_features=True`` writes raw unnormalized ``(N, F)`` descriptor
  tensors to ``/torch_cache/features`` so later compatible HDF5-backed runs
  can reuse them across sessions
* ``persist_force_derivatives=True`` writes sparse local derivative payloads
  for force-labeled entries to ``/torch_cache/force_derivatives``
* ``cache_features=True`` is a trainer-owned in-memory runtime cache attached
  to the current dataset instance; it speeds up repeated accesses within a run
  but does not modify the HDF5 file. Its size is controlled by
  ``cache_feature_max_entries`` on ``TorchTrainingConfig``

Runtime precedence is explicit:

* energy-view sample materialization prefers the runtime
  ``cache_features=True`` cache first, then compatible persisted HDF5
  features, then on-the-fly featurization
* force-view sample materialization reuses compatible persisted raw features
  when they exist
* when both persisted raw features and persisted local derivatives exist for a
  force-supervised entry, the force path can serve that sample without
  rebuilding graph or triplet payloads

This is separate from ``CachedStructureDataset``, which is an eager in-memory
energy-only cache for structure-list workflows rather than an on-disk HDF5
cache for reuse across runs. The developer-facing schema layout and metadata
contract are documented in :doc:`../dev/torch_force_hdf5_cache`.

Training from HDF5 Database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from aenet.torch_training.dataset import HDF5StructureDataset
   from aenet.torch_training import TorchTrainingConfig, Adam

   # Load existing database later (read-only, lazy access)
   # Reopening is optional if you still have the build-time ``db`` instance.
   dataset = HDF5StructureDataset(
       descriptor=None,                # recover from persisted manifest
       database_file="datasets/training.h5",
       mode="load",                 # Read-only mode
   )

   # Train with automatic splitting
   config = TorchTrainingConfig(
       iterations=100,
       method=Adam(mu=0.001, batchsize=32),
       testpercent=10,
       force_weight=0.1,
       force_fraction=0.3,
       force_sampling="random",
       cache_features=True,
       cache_feature_max_entries=1024,
       cache_neighbors=True,
       cache_neighbor_max_entries=512,
       num_workers=8,               # Parallel workers (each opens own handle)
       prefetch_factor=4,
       persistent_workers=True,
   )

   pot.train(dataset=dataset, config=config)

.. note::

   Prebuilt ``dataset=...`` objects are passive data sources. Runtime controls
   such as ``force_fraction``, ``force_sampling``, ``cache_features``,
   ``cache_neighbors``, and ``cache_force_triplets`` belong on
   ``TorchTrainingConfig`` and can be changed between runs over the same
   dataset object.

Key HDF5 Features
~~~~~~~~~~~~~~~~~

* **Lazy-loading**: Structures read from disk on-demand, minimizing RAM
* **Multiprocessing-safe**: Each DataLoader worker opens its own read-only handle
* **Compression**: Built-in HDF5 compression (zlib, blosc) reduces disk usage
* **LRU caching**: Configurable in-memory cache per worker for frequently accessed entries
* **Build parallelism**: ``build_workers`` accelerates parser execution and
  optional persisted-cache generation, but ordered HDF5 writes still happen
  in the parent process
* **Parser concurrency**: When using ``build_workers > 1``, make sure the
  parser callable is safe to invoke concurrently over independent file paths
* **Unified persisted cache**: Optional ``/torch_cache/features`` and
  ``/torch_cache/force_derivatives`` sections can be written once and reused
  lazily across later HDF5-backed runs
* **Separate trainer cache limits**: ``cache_feature_max_entries``,
  ``cache_neighbor_max_entries``, and ``cache_force_triplet_max_entries`` bound
  the trainer-owned runtime caches separately from HDF5
  ``in_memory_cache_size``
* **Deterministic handle cleanup**: Call ``dataset.close()`` or use
  ``with HDF5StructureDataset(...) as dataset:``


Dataset Splitting Strategies
-----------------------------

Automatic Splitting
~~~~~~~~~~~~~~~~~~~

When providing a single ``dataset`` parameter to ``train()``, the trainer automatically splits it based on ``config.testpercent``:

.. code-block:: python

   # Trainer handles split automatically
   pot.train(dataset=my_dataset, config=config)  # Uses testpercent

.. note::

   When ``testpercent > 0``, validation-driven features such as
   ``use_scheduler=True`` and ``save_best=True`` become active. For very small
   validation splits, prefer disabling those features or creating an explicit
   train/test split with enough validation structures for stable monitoring.

Manual Splitting
~~~~~~~~~~~~~~~~

For full control over train/test splits:

.. code-block:: python

   from aenet.torch_training.dataset import train_test_split_dataset

   # Generic splitter for any Dataset (returns Subset objects)
   train_ds, test_ds = train_test_split_dataset(
       dataset, test_fraction=0.1, seed=42
   )

   pot.train(train_dataset=train_ds, test_dataset=test_ds, config=config)

This works for ``CachedStructureDataset`` and ``HDF5StructureDataset`` as
well. When the split is already explicit, prefer ``testpercent=0`` in the
training config to avoid implying that another automatic split will occur.

Stratified or Custom Splits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For advanced splitting strategies, manually create your datasets:

.. code-block:: python

   from torch.utils.data import Subset

   # Custom indices (e.g., stratified by composition)
   train_indices = [0, 2, 4, 6, 8, ...]
   test_indices = [1, 3, 5, 7, 9, ...]

   train_ds = Subset(dataset, train_indices)
   test_ds = Subset(dataset, test_indices)

   pot.train(train_dataset=train_ds, test_dataset=test_ds, config=config)

``Subset`` wrappers are supported for training and dataset-backed inference.
For ``CachedStructureDataset``, the subset reuses cached samples from the
base dataset; it does not build a separate smaller cache.


Performance Optimization Tips
------------------------------

For Large Datasets (HDF5)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Efficient large-scale training
   dataset = HDF5StructureDataset(
       descriptor=descriptor,
       database_file="large_dataset.h5",
       mode="load",
       in_memory_cache_size=4096,   # Larger cache for workers
   )

   config = TorchTrainingConfig(
       force_fraction=0.3,
       cache_neighbors=True,
       num_workers=16,              # More workers for I/O
       prefetch_factor=8,           # More prefetching
       persistent_workers=True,     # Keep workers alive
   )

Caching Strategies
~~~~~~~~~~~~~~~~~~

Set these on ``TorchTrainingConfig``:

* **cache_features**: For energy-only structure-list workflows, this can
  trigger eager feature caching. For force training, it caches energy-view
  features for structures not selected for force supervision in the current
  epoch. On HDF5 datasets, this runtime cache sits above compatible persisted
  HDF5 features and does not write back to disk.
* **cache_neighbors**: Reuse neighbor search results for energy-view reuse and legacy non-graph paths
* **cache_force_triplets**: Cache CSR graphs and triplets instead of rebuilding them for the default sparse force-training path
* **cache_*_max_entries**: Bound the trainer-owned runtime caches per split and per process/worker
* **cache_warmup**: Optional single-process cache prefill before epoch 0; skipped automatically when ``num_workers > 0``

For repeated fixed-geometry HDF5 workflows, prefer build-time
``persist_features=True`` and ``persist_force_derivatives=True`` when you want
cache reuse across separate training sessions. Use
``CachedStructureDataset`` when you want a one-process eager in-memory cache
for energy-only structure-list training.


Common Pitfalls
---------------

1. **Parser not pickleable**: When using HDF5 with ``num_workers > 0``, the parser function must be defined at module top-level (not a lambda or nested function).

2. **Descriptor mismatch**: Ensure descriptor species order matches your dataset. Datasets use ``descriptor.species_to_idx`` for species indexing.

3. **Memory exhaustion**: For datasets with >100K structures, use ``HDF5StructureDataset`` instead of loading all structures into memory.

4. **Force fraction too low**: Setting ``force_fraction`` very low (< 0.1) may degrade force accuracy. Balance between speed and accuracy by testing different fractions.


See Also
--------

* :doc:`torch_training` - PyTorch training workflow
* :doc:`torch_featurization` - Structure featurization with descriptors
