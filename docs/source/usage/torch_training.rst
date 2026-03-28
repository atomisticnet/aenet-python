PyTorch-based Training
======================

This page covers training machine learning interatomic potentials (MLIPs)
using the PyTorch-based implementation in ``aenet-python``. The PyTorch
implementation provides a pure Python workflow with GPU acceleration,
and automatic differentiation for forces.

.. note::

   Training as described here makes use of PyTorch.  Make sure to
   install core torch support as described in :doc:`installation`.  Most
   descriptor-based training workflows also require the matching
   ``torch-scatter`` and ``torch-cluster`` wheels.

.. note::

   **Alternative**: For training using ænet's Fortran-based tools,
   see :doc:`training`.

Overview
--------

The PyTorch training workflow consists of three main steps:

1. **Prepare structures**: Load atomic structures with energies (and optionally forces)
2. **Configure training**: Set up the model architecture and training parameters
3. **Train the model**: Run the training loop and save the trained potential

This tutorial demonstrates both **energy-only** and training on
**energies and forces**.


Example notebooks
-----------------

Jupyter notebooks with examples can be found in the `notebooks
<https://github.com/atomisticnet/aenet-python/tree/master/notebooks>`_
directory within the repository.

For the maintained PyTorch training walkthrough, including the file-backed TiO2
workflow, explicit ``CachedStructureDataset`` usage, fixed train/test splits,
and dataset-backed prediction, see
`example-05-torch-training.ipynb
<https://github.com/atomisticnet/aenet-python/blob/master/notebooks/example-05-torch-training.ipynb>`_.


Energy-Only Training
--------------------

Here's a compact CPU-only example that keeps the full setup in memory. The
notebook linked above remains the maintained home for the file-backed TiO2
workflow, checkpoint rotation, explicit ``CachedStructureDataset`` usage,
fixed train/test splits, dataset-backed prediction, and plotting.

.. code-block:: python

   import numpy as np
   import torch

   from aenet.torch_featurize import ChebyshevDescriptor
   from aenet.torch_training import (
       Adam,
       Structure,
       TorchANNPotential,
       TorchTrainingConfig,
   )

   structures = [
       Structure(
           positions=np.array(
               [
                   [0.0, 0.0, 0.0],
                   [0.9, 0.0, 0.0],
                   [0.0, 0.9, 0.0],
               ]
           ),
           species=["H", "H", "H"],
           energy=0.0,
       ),
       Structure(
           positions=np.array(
               [
                   [0.1, 0.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
               ]
           ),
           species=["H", "H", "H"],
           energy=0.5,
       ),
   ]

   descriptor = ChebyshevDescriptor(
       species=["H"],
       rad_order=1,
       rad_cutoff=2.0,
       ang_order=0,
       ang_cutoff=2.0,
       min_cutoff=0.1,
       device="cpu",
       dtype=torch.float64,
   )
   arch = {"H": [(4, "tanh")]}

   mlp = TorchANNPotential(arch, descriptor=descriptor)

   config = TorchTrainingConfig(
       iterations=1,
       method=Adam(mu=0.001, batchsize=1),
       testpercent=50,
       force_weight=0.0,
       atomic_energies={"H": 0.0},
       normalize_features=False,
       normalize_energy=False,
       memory_mode="cpu",
       device="cpu",
       checkpoint_dir=None,
       checkpoint_interval=0,
       max_checkpoints=None,
       save_best=False,
       use_scheduler=False,
   )

   results = mlp.train(structures=structures, config=config)
   print(results.errors[["RMSE_train", "RMSE_test"]].tail(1))

This trains a neural network potential using energies only, with 50% of the
structures held out for validation. The
:meth:`~aenet.torch_training.TorchANNPotential.train` method returns a
:class:`~aenet.io.train.TrainOut` object containing training history,
statistics, and plotting helpers.

.. note::

   Setting ``testpercent > 0`` does more than hold out structures. It also
   enables any validation-driven controls in your configuration, such as
   ``use_scheduler=True`` and ``save_best=True``. On very small validation
   splits, these controls can react to noisy metrics and change the training
   behavior qualitatively.


Force Training
--------------

To include force supervision, add force arrays to the structures and set
``force_weight > 0.0``:

.. doctest::

   >>> from aenet.torch_training import Adam, TorchTrainingConfig

   >>> config = TorchTrainingConfig(
   ...     iterations=2,
   ...     method=Adam(mu=0.001, batchsize=1),
   ...     testpercent=50,
   ...     force_weight=0.1,
   ...     force_fraction=0.5,
   ...     force_sampling="fixed",
   ... )
   >>> config.force_weight
   0.1
   >>> config.force_fraction
   0.5
   >>> config.force_sampling
   'fixed'

The ``force_weight`` parameter (α) balances energy and force contributions:

.. math::

   \text{Loss} = (1 - \alpha) \cdot \text{RMSE}_{\text{energy}} + \alpha \cdot \text{RMSE}_{\text{forces}}

Common values:

* ``force_weight=0.0``: Energy-only (fastest training)
* ``force_weight=0.1``: Primarily energy, with force regularization
* ``force_weight=0.5``: Equal weighting
* ``force_weight=1.0``: Force-only (rarely used)

.. note::

   Force training requires structures with force data. Structures without
   forces will only contribute to the energy loss term.

The notebook linked above remains the maintained home for the longer
force-training workflow, including checkpoint output and plotting.


Dataset Options
---------------

The PyTorch training workflow supports flexible dataset options, from simple
structure lists to advanced HDF5-backed lazy-loading for large-scale
training.

For detailed information about dataset classes, input formats, and performance
optimization, see :doc:`torch_datasets`.

The longer file-backed dataset workflow is intentionally kept in the training
notebook above so the ``torch_datasets`` page can stay focused on compact
API-facing examples.

Execution Model
~~~~~~~~~~~~~~~~

The current trainer has two distinct runtime stages:

1. Sample preparation happens in the main process when ``num_workers=0``, or
   in ``DataLoader`` workers when ``num_workers > 0``. Structures are
   converted to tensors on ``descriptor.device``, and descriptor
   featurization, neighbor reuse, graph/triplet construction, and lazy HDF5
   cache reads happen there.
2. The collated batch is then moved onto ``config.device`` inside the
   training loop. Model forward passes, normalization, loss computation, and
   optimizer steps run on that device.

In practice, GPU training with ``num_workers > 0`` is best understood as
worker-side data preparation feeding a training loop on the selected device.
It is not currently a separate mixed CPU/GPU execution pipeline.

If ``descriptor.device`` and ``config.device`` match, featurization and model
compute happen on the same device. If they differ, samples are materialized on
``descriptor.device`` and transferred before the forward pass. The compact
examples on this page create the descriptor on CPU, so later
``device='cuda'`` examples describe CPU-side sample preparation feeding GPU
training unless you also move the descriptor to CUDA.

For HDF5-backed datasets, each worker reopens its own read-only file handle
and keeps its own bounded ``in_memory_cache_size`` LRU cache. Trainer-owned
runtime caches (``cache_features``, ``cache_neighbors``,
``cache_force_triplets``) are also per process/worker, so
``cache_warmup=True`` is skipped automatically when ``num_workers > 0``. See
:doc:`torch_datasets` for persisted HDF5 cache precedence and for the
distinction between build-time ``build_workers`` and training-time
``num_workers``.

``memory_mode='mixed'`` is reserved for a future real mixed-memory mode and
currently raises ``NotImplementedError`` if requested. Today, the supported
execution modes remain ``'cpu'`` and ``'gpu'``.

Performance Optimization Tips
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**For Energy-Only Training**

.. doctest::

   >>> from aenet.torch_training import TorchTrainingConfig
   >>> config = TorchTrainingConfig(
   ...     force_weight=0.0,
   ...     cache_features=True,
   ...     num_workers=4,
   ...     prefetch_factor=4,
   ...     persistent_workers=True,
   ... )
   >>> (config.cache_features, config.num_workers, config.prefetch_factor)
   (True, 4, 4)

**For Force Training**

.. doctest::

   >>> config = TorchTrainingConfig(
   ...     force_weight=0.1,
   ...     force_fraction=0.3,
   ...     force_sampling="random",
   ...     cache_features=True,
   ...     cache_neighbors=True,
   ...     num_workers=4,
   ...     prefetch_factor=4,
   ... )
   >>> (config.cache_neighbors, config.cache_force_triplets)
   (True, False)

**Caching Strategies**

* **cache_features**: For energy-only structure-list workflows, this can
  precompute features eagerly. For force training, it caches energy-view
  features for structures not selected for force supervision in the current
  epoch.
* **cache_neighbors**: Reuse neighbor search results for energy-view reuse
  and legacy non-graph paths
* **cache_force_triplets**: Cache CSR graphs and triplets for the default sparse
  force-training path instead of rebuilding them on demand
* **cache_*_max_entries**: Bound the trainer-owned runtime caches per split
  and per process/worker instead of letting them grow without limit
* **cache_warmup**: Optional single-process prefill of trainer-owned runtime
  caches before epoch 0; skipped automatically when ``num_workers > 0``

These runtime caches are distinct from the on-disk HDF5 persisted cache
sections created with ``HDF5StructureDataset.build_database(...)``. For HDF5
datasets, ``cache_features=True`` is still only a per-run in-memory layer; it
does not replace ``persist_features=True`` or
``persist_force_derivatives=True``, which are the build-time options for
reusing raw features or sparse local derivatives across sessions. See
:doc:`torch_datasets` for the full cache-precedence workflow.

Common Pitfalls
~~~~~~~~~~~~~~~

1. **Descriptor mismatch**: Ensure descriptor species order matches your dataset.
   Datasets use ``descriptor.species_to_idx`` for species indexing.

Training Configuration
----------------------

The :class:`~aenet.torch_training.TorchTrainingConfig` class provides extensive
control over the training process. Here are the most commonly used parameters:

Basic Settings
~~~~~~~~~~~~~~

.. doctest::

   >>> from aenet.torch_training import TorchTrainingConfig
   >>> config = TorchTrainingConfig(
   ...     iterations=100,
   ...     testpercent=10,
   ...     device="cpu",
   ...     show_progress=True,
   ... )
   >>> (config.iterations, config.device, config.show_progress)
   (100, 'cpu', True)

Optimizer Selection
~~~~~~~~~~~~~~~~~~~

Choose and configure the optimization algorithm:

.. doctest::

   >>> from aenet.torch_training import Adam, SGD, TorchTrainingConfig

   >>> method = Adam(
   ...     mu=0.001,
   ...     batchsize=32,
   ...     beta1=0.9,
   ...     beta2=0.999,
   ...     weight_decay=0.0,
   ... )
   >>> (method.method_name, method.batchsize)
   ('adam', 32)

   >>> method = SGD(
   ...     lr=0.01,
   ...     batchsize=32,
   ...     momentum=0.9,
   ...     weight_decay=0.0,
   ... )
   >>> TorchTrainingConfig(iterations=100, method=method).method.method_name
   'sgd'

**Adam** is recommended for most applications due to its adaptive learning rates
and robust convergence properties.


Common Training Patterns
-------------------------

Small Dataset (< 100 structures)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = TorchTrainingConfig(
       iterations=200,  # More epochs for small data
       method=Adam(mu=0.001, batchsize=16),  # Smaller batches
       testpercent=10,
       force_weight=0.1,
       device='cpu'  # CPU fine for small datasets
   )

Large Dataset (> 500 structures)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = TorchTrainingConfig(
       iterations=50,   # Fewer epochs needed
       method=Adam(mu=0.001, batchsize=64),  # Larger batches
       testpercent=10,
       force_weight=0.1,
       device='cuda',  # Model/loss on GPU
       # Performance optimizations
       cache_features=True,  # Runtime in-memory feature cache
       cache_feature_max_entries=1024,
       num_workers=8,         # Parallel CPU-side sample preparation
       prefetch_factor=4
   )

Energy-Only with Maximum Speed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = TorchTrainingConfig(
       iterations=100,
       method=Adam(mu=0.001, batchsize=32),
       testpercent=10,
       force_weight=0.0,  # Energy-only
       cache_features=True,  # Bounded runtime feature cache for this run
       cache_warmup=True,    # Optional single-process prefill
       device='cuda'
   )

Force Training with Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = TorchTrainingConfig(
       iterations=100,
       method=Adam(mu=0.001, batchsize=32),
       testpercent=10,
       force_weight=0.1,
       force_fraction=0.3,  # Use 30% of forces (3× faster)
       cache_neighbors=True,  # Cache worker-local neighbor lists
       num_workers=4,         # Parallel CPU-side sample preparation
       device='cuda'
   )


Advanced Configuration Reference
---------------------------------

This section documents all configuration parameters available in
:class:`~aenet.torch_training.TorchTrainingConfig`.

Checkpointing & Model Saving
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**checkpoint_dir** : str (default: 'checkpoints')
   Directory to save checkpoint files. Set to None to disable checkpointing.

**checkpoint_interval** : int (default: 1)
   Save a checkpoint every N epochs. Set to 0 to disable periodic checkpoints.

**max_checkpoints** : int (default: None)
   Maximum number of checkpoint files to keep. Older checkpoints are automatically
   deleted. None = keep all checkpoints.

**save_best** : bool (default: True)
   Save the model with the best validation loss as ``best_model.pt``.
   Requires ``testpercent > 0`` to compute validation loss.

   For very small validation sets, the selected checkpoint can be unstable.
   In that case prefer ``save_best=False`` or supply a larger or explicit
   validation split.

**Resuming Training**

To resume training from a checkpoint, pass the checkpoint path to
``train(..., resume_from="checkpoints/checkpoint_epoch_0050.pt")``. The
notebook above contains the maintained checkpoint workflow.

When ``resume_from`` is provided, ``config.iterations`` means the number of
additional epochs to run in that ``train()`` call. For example, resuming a
checkpoint with ``iterations=10`` runs 10 more epochs after the saved
checkpoint epoch, regardless of how many epochs were completed in the
original run. This applies to numbered checkpoints and ``best_model.pt``
alike.

The trainer will automatically:

* Load model and optimizer state
* Restore training history and normalization statistics
* Continue from the next epoch

.. note::

   Checkpoint files are NOT interchangeable with model files created by ``save()``.
   Checkpoints include additional training state (optimizer, history) needed for
   resuming, while model files are optimized for deployment and inference.


Learning Rate Scheduling
~~~~~~~~~~~~~~~~~~~~~~~~~

**use_scheduler** : bool (default: False)
   Enable learning rate scheduler. Uses ReduceLROnPlateau, which reduces the
   learning rate when validation loss plateaus.

**scheduler_patience** : int (default: 10)
   Number of epochs with no improvement before reducing learning rate.

**scheduler_factor** : float (default: 0.5)
   Factor by which to reduce learning rate. New LR = current LR × factor.

**scheduler_min_lr** : float (default: 1e-6)
   Minimum allowed learning rate. Scheduler stops reducing below this value.

**Example Usage**

.. doctest::

   >>> from aenet.torch_training import Adam, TorchTrainingConfig
   >>> config = TorchTrainingConfig(
   ...     iterations=200,
   ...     method=Adam(mu=0.001, batchsize=32),
   ...     testpercent=10,
   ...     use_scheduler=True,
   ...     scheduler_patience=10,
   ...     scheduler_factor=0.5,
   ...     scheduler_min_lr=1e-6,
   ... )
   >>> (config.use_scheduler, config.scheduler_patience)
   (True, 10)

The scheduler helps training converge when progress stalls, automatically
adjusting the learning rate for optimal performance.

.. note::

   The scheduler requires ``testpercent > 0`` to monitor validation loss.
   With only a few validation structures, the monitored loss can be too noisy
   for stable plateau detection. In that case prefer ``use_scheduler=False``
   or a larger or explicit validation split.


Force Training Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

**force_fraction** : float (default: 1.0)
   Fraction of structures (0.0-1.0) to use for force training. Using a subset
   can significantly speed up training while maintaining accuracy.
   Example: ``force_fraction=0.3`` uses 30% of force-labeled structures.

**force_sampling** : str (default: 'random')
   Sampling strategy for force subset: ``'random'`` (resample periodically) or
   ``'fixed'`` (static subset). Random sampling provides better generalization.

**force_resample_num_epochs** : int (default: 0)
   Number of epochs between resampling the force-trained subset when
   ``force_sampling='random'``. Controls the resampling frequency:

   * ``0`` = No resampling (use fixed subset for entire training)
   * ``1`` = Resample every epoch (maximum variety, highest computational cost)
   * ``N > 1`` = Resample every N epochs (balance between variety and efficiency)

   .. note::
      The default value of 0 (no resampling) represents a conservative choice
      that maintains consistent training dynamics and reduces computational
      overhead. Set to 1 or higher for dynamic resampling.

**force_min_structures_per_epoch** : int (default: 1)
   Minimum number of force-labeled structures per epoch, regardless of
   ``force_fraction``. Ensures force gradient signal is not lost.

**force_scale_unbiased** : bool (default: False)
   Apply sqrt(1/f) scaling to force RMSE where f is the supervised fraction,
   approximating constant scale under sub-sampling.


Performance & Caching
~~~~~~~~~~~~~~~~~~~~~~

**cache_features** : bool (default: False)
   Enable feature caching. Behavior depends on training mode:

   * For energy-only training (``force_weight=0``): Pre-computes all features once,
     providing ~100× speedup
   * For force training (``force_weight > 0``): Caches features for structures not
     selected for force supervision in current epoch (useful with ``force_fraction < 1.0``)

**cache_feature_max_entries** : int or None (default: 1024)
   Maximum number of trainer-owned energy-view feature entries to retain per
   split and per process/worker when ``cache_features=True``. Use ``None`` for
   an explicit unbounded cache or ``0`` to suppress storage.

**cache_neighbors** : bool (default: False)
   Cache per-structure neighbor graphs (indices, displacement vectors) across
   epochs. Avoids repeated neighbor searches for fixed geometries on
   energy-view reuse and legacy non-graph paths. Supported force training
   does not require this option.

**cache_neighbor_max_entries** : int or None (default: 512)
   Maximum number of trainer-owned neighbor payload entries to retain per
   split and per process/worker when ``cache_neighbors=True``. Use ``None`` for
   an explicit unbounded cache or ``0`` to suppress storage.

**cache_force_triplets** : bool (default: False)
   Cache CSR neighbor graphs and precompute angular triplet indices for the
   default sparse force-training path. Leaving this disabled still uses the
   sparse graph/triplet path, but rebuilds those graph payloads on demand.

**cache_force_triplet_max_entries** : int or None (default: 256)
   Maximum number of trainer-owned graph/triplet payload entries to retain per
   split and per process/worker when ``cache_force_triplets=True``. Use
   ``None`` for an explicit unbounded cache or ``0`` to suppress storage.

**cache_persist_dir** : str (default: None)
   Directory for persisting graph/triplet caches to disk for reuse across runs.

**cache_scope** : str (default: 'all')
   Which dataset splits to cache: ``'train'``, ``'val'``, or ``'all'``.

**cache_warmup** : bool (default: False)
   If True, pre-populate trainer-owned runtime caches before the first epoch
   in single-process training. When all enabled caches have finite entry
   limits, warmup stops once those limits are filled. Warmup is skipped
   automatically when ``num_workers > 0`` because workers own their own cache
   instances and the main-process warmup would not populate those worker-local
   caches.

**num_workers** : int (default: 0)
   Number of parallel ``DataLoader`` workers for structure loading, HDF5
   reads, and on-the-fly featurization. ``0`` keeps sample preparation in the
   main process. Values ``>0`` parallelize worker-side sample preparation; they
   do not parallelize model compute.

**prefetch_factor** : int (default: 2)
   Number of batches to prefetch per worker when ``num_workers > 0``.

**persistent_workers** : bool (default: True)
   Keep DataLoader workers alive between epochs for faster iteration.
   During training, this is disabled automatically when
   ``force_sampling='random'`` uses epoch-level resampling, because worker
   copies would otherwise keep a stale force-supervision subset. Trainer-owned
   runtime caches and HDF5 ``in_memory_cache_size`` state are also
   worker-local when ``num_workers > 0``. For HDF5-backed datasets, worker
   handles are opened lazily per worker and closed explicitly when that worker
   exits.


Data Filtering & Quality Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**max_energy** : float (default: None)
   Exclude structures with total energy above this threshold. Useful for
   removing outliers or high-energy configurations.

**max_forces** : float (default: None)
   Exclude structures with maximum atomic force magnitude above this threshold.
   Units: eV/Å.


Energy Configuration
~~~~~~~~~~~~~~~~~~~~

**atomic_energies** : dict (default: None)
   Optional atomic reference energies used to convert total energies to
   cohesive-energy targets during training.
   Format: ``{'H': -13.6, 'O': -432.0, ...}`` in eV.
   If omitted, the training target remains the total energy because all atomic
   reference energies default to 0.0.

**normalize_features** : bool (default: True)
   Normalize features to zero mean and unit variance. Improves training
   stability and convergence.

**normalize_energy** : bool (default: True)
   Normalize energies by shifting and scaling. Applied after cohesive energy
   conversion if enabled.

**E_shift** : float (default: None)
   Override per-atom energy shift for normalization. Auto-computed from
   training set if None.

**E_scaling** : float (default: None)
   Override energy scaling factor. Auto-computed from training set if None.

**feature_stats** : dict (default: None)
   Override feature normalization statistics.
   Format: ``{'mean': np.ndarray, 'std': np.ndarray}``.
   Auto-computed from training set if None.


Output & Diagnostics
~~~~~~~~~~~~~~~~~~~~

**save_energies** : bool (default: False)
   Save predicted energies for train/test sets to disk. The
   ``Path-of-input-file`` column preserves the original structure path or
   name when available; otherwise it uses a stable ``structure_XXXXXX``
   identifier from the pre-split input order. For HDF5-backed datasets,
   the identifier is synthesized from persisted source metadata as
   ``display_name#frame=N`` when a display name is available,
   ``source_id#frame=N`` otherwise, then ``name#frame=N`` when only the
   persisted structure name is available, and
   ``structure_XXXXXX#frame=N`` as the final fallback. Source metadata is
   validated at HDF5 build time so these identifiers are not silently
   truncated on write.

**save_forces** : bool (default: False)
   Save predicted forces for train/test sets to disk.

**timing** : bool (default: False)
   Enable detailed timing output for performance profiling.

**show_progress** : bool (default: True)
   Display epoch-level progress bar.

**show_batch_progress** : bool (default: False)
   Display batch-level progress bar within each epoch. Verbose for large
   datasets.


Advanced Options
~~~~~~~~~~~~~~~~

**precision** : str (default: 'auto')
   Numeric precision: ``'auto'`` (match descriptor dtype), ``'float32'``, or
   ``'float64'``. Higher precision improves accuracy but increases memory usage.

**memory_mode** : str (default: 'gpu')
   Memory management strategy: ``'cpu'``, ``'gpu'``, or ``'mixed'``.
   ``'mixed'`` is reserved for a future real mixed-memory implementation and
   currently raises ``NotImplementedError``. Use ``'cpu'`` or ``'gpu'`` with
   ``descriptor.device`` and ``device`` set explicitly to control the current
   execution path.

**device** : str (default: None)
   PyTorch device: ``'cpu'``, ``'cuda'``, or ``'cuda:0'``. Auto-detected if
   None. This selects the model/training-loop device. ``descriptor.device``
   separately controls where structures are featurized. When the two differ,
   samples are prepared on ``descriptor.device`` and moved to ``device``
   before the forward pass.


Monitoring Training Progress
-----------------------------

The :class:`~aenet.io.train.TrainOut` object returned by ``train()`` provides
built-in visualization and analysis tools:

Common entry points are:

* ``results.plot_training_summary(outfile="training_summary.png")`` for a
  combined energy/force plot
* ``results.plot_training_errors(outfile="energy_errors.png")`` for
  energy-only training curves
* ``results.plot_force_errors(outfile="force_errors.png")`` when force data
  are present
* ``results.errors`` for direct access to the underlying pandas DataFrame used
  for custom plotting

The notebook linked above demonstrates these plotting helpers in a full
training workflow.

Signs of good training:

* Steady decrease in both train and test RMSE
* Test RMSE follows train RMSE (no overfitting)
* Convergence to acceptable error levels (< 0.01 eV/atom for energy)

Signs of problems:

* Test RMSE increases while train RMSE decreases (overfitting)
* Both RMSEs plateau at high values (underfitting, poor architecture)
* Divergence or oscillation (learning rate too high)


See Also
--------

* :doc:`torch_featurization` - PyTorch-based structure featurization
* :doc:`choosing_implementation` - Fortran vs PyTorch comparison
