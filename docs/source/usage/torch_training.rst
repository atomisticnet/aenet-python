PyTorch-based Training
======================

This page covers training machine learning interatomic potentials (MLIPs)
using the PyTorch-based implementation in ``aenet-python``. The PyTorch
implementation provides a pure Python workflow with GPU acceleration,
and automatic differentiation for forces.

.. note::

   Training as described here makes use of PyTorch.  Make sure to
   install ænet with the ``[torch]`` extra as described in :doc:`installation`.

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


Energy-Only Training
--------------------

Here's a minimal example for training an energy-only potential:

.. code-block:: python

   from aenet.torch_featurize import ChebyshevDescriptor
   from aenet.torch_training import TorchANNPotential, TorchTrainingConfig, Adam
   import glob

   # Load structures
   structures = glob.glob("./xsf/*.xsf")

   # Define network architecture
   arch = {
       'O': [(30, 'tanh'), (30, 'tanh')],
       'H': [(20, 'tanh'), (20, 'tanh')]
   }

   # Define local atomic descriptor
   descr = ChebyshevDescriptor(
       species=["O", "H"],
       rad_order=10,
       rad_cutoff=6.0,
       ang_order=3,
       ang_cutoff=3.5,
       min_cutoff=0.5,
       device="cpu",
       dtype=torch.float64,
   )

   mlp = TorchANNPotential(arch, descriptor=descr)

   # Configure training
   cfg = TorchTrainingConfig(
       iterations=100,
       method=Adam(mu=0.001, batchsize=32),
       testpercent=10,
       force_weight=0.0  # Energy-only
   )

   # Train model (returns TrainOut object)
   results = mlp.train(
       structures=structures,
       config=cfg
   )

   # Access training statistics
   print(results.stats)
   # Or access specific values from the errors DataFrame:
   print(f"Final train RMSE: {results.errors['RMSE_train'].iloc[-1]:.4f} eV/atom")
   print(f"Final test RMSE: {results.errors['RMSE_test'].iloc[-1]:.4f} eV/atom")

This trains a neural network potential using energies only, with 10% of
structures held out for validation. The ``train_model()`` function returns
a :class:`~aenet.io.train.TrainOut` object containing training history and
statistics.


Force Training
--------------

To include force supervision, set ``force_weight > 0.0``:

.. code-block:: python

   from aenet.torch_training import TorchTrainingConfig, Adam

   config = TorchTrainingConfig(
       iterations=200,
       method=Adam(mu=0.001, batchsize=16),  # Smaller batch for forces
       testpercent=10,
       force_weight=0.1  # 90% energy, 10% forces
   )

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


Dataset Options
---------------

The PyTorch training workflow supports flexible dataset options, from simple structure lists to advanced HDF5-backed lazy-loading for large-scale training.

For detailed information about dataset classes, input formats, and performance optimization, see :doc:`torch_datasets`.

Performance Optimization Tips
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**For Energy-Only Training**

.. code-block:: python

   # Maximum speed with cached features
   config = TorchTrainingConfig(
       force_weight=0.0,
       cache_features=True,
       num_workers=4,
       prefetch_factor=4,
       persistent_workers=True,
   )

**For Force Training**

.. code-block:: python

   # Optimize force training with caching and subsampling
   config = TorchTrainingConfig(
       force_weight=0.1,
       force_fraction=0.3,
       force_sampling='random',          # Better generalization
       cache_features=True,              # Cache features for non-force structures
       cache_force_neighbors=True,       # Reuse neighbor searches
       cache_force_triplets=True,        # Vectorized operations
       num_workers=4,
       prefetch_factor=4,
   )

**Caching Strategies**

* **cache_features**: For energy-only training, pre-computes all features once. For
  force training, caches features for structures not selected for force supervision
  in the current epoch (useful with ``force_fraction < 1.0``)
* **cache_force_neighbors**: Reuse neighbor search results (force training only)
* **cache_force_triplets**: Precompute CSR graphs and triplets for vectorized
  operations (force training only)

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

.. code-block:: python

   config = TorchTrainingConfig(
       iterations=100,        # Number of training epochs
       testpercent=10,        # % of data for validation
       device='cuda',         # 'cpu', 'cuda', or 'cuda:0'
       show_progress=True     # Display progress bar
   )

Optimizer Selection
~~~~~~~~~~~~~~~~~~~

Choose and configure the optimization algorithm:

.. code-block:: python

   from aenet.torch_training import Adam, SGD

   # Adam optimizer (recommended)
   method = Adam(
       mu=0.001,           # Learning rate
       batchsize=32,       # Structures per batch
       beta1=0.9,          # First moment decay
       beta2=0.999,        # Second moment decay
       weight_decay=0.0    # L2 regularization
   )

   # SGD optimizer
   method = SGD(
       lr=0.01,            # Learning rate
       batchsize=32,
       momentum=0.9,
       weight_decay=0.0
   )

   config = TorchTrainingConfig(iterations=100, method=method)

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
       device='cuda',  # Use GPU for speedup
       # Performance optimizations
       cache_features=True,  # If energy-only
       num_workers=8,         # Parallel data loading
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
       cache_features=True,  # 100× speedup for energy-only
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
       cache_force_neighbors=True,  # Cache neighbor lists
       cache_force_triplets=True,   # Vectorized operations
       num_workers=4,
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

**Resuming Training**

To resume training from a checkpoint, pass the checkpoint path to ``train()``:

.. code-block:: python

   # Resume from a specific checkpoint
   results = mlp.train(
       structures=structures,
       config=cfg,
       resume_from="checkpoints/checkpoint_epoch_0050.pt"
   )

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

.. code-block:: python

   config = TorchTrainingConfig(
       iterations=200,
       method=Adam(mu=0.001, batchsize=32),
       testpercent=10,  # Required for scheduler
       use_scheduler=True,
       scheduler_patience=10,  # Reduce LR if no improvement for 10 epochs
       scheduler_factor=0.5,   # Halve the learning rate
       scheduler_min_lr=1e-6   # Stop at 1e-6
   )

The scheduler helps training converge when progress stalls, automatically
adjusting the learning rate for optimal performance.

.. note::

   The scheduler requires ``testpercent > 0`` to monitor validation loss.


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

**cache_force_neighbors** : bool (default: False)
   Cache per-structure neighbor graphs (indices, displacement vectors) across
   epochs. Avoids repeated neighbor searches for fixed geometries.
   Only applicable for force training.

**cache_force_triplets** : bool (default: False)
   Build CSR neighbor graphs and precompute angular triplet indices for
   vectorized featurization. Removes Python enumeration loops.
   Only applicable for force training.

**cache_persist_dir** : str (default: None)
   Directory for persisting graph/triplet caches to disk for reuse across runs.

**cache_scope** : str (default: 'all')
   Which dataset splits to cache: ``'train'``, ``'val'``, or ``'all'``.

**num_workers** : int (default: 0)
   Number of parallel DataLoader workers for on-the-fly featurization.
   0 = main process only. Values >0 enable parallel data loading.

**prefetch_factor** : int (default: 2)
   Number of batches to prefetch per worker when ``num_workers > 0``.

**persistent_workers** : bool (default: True)
   Keep DataLoader workers alive between epochs for faster iteration.


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

**energy_target** : str (default: 'cohesive')
   Energy reference space: ``'cohesive'`` (relative to atomic references) or
   ``'total'`` (absolute energies). Cohesive energies improve training
   stability by removing large atomic contributions.

**atomic_energies** : dict (default: None)
   Atomic reference energies for cohesive energy calculation.
   Format: ``{'H': -13.6, 'O': -432.0, ...}`` in eV.
   Required when ``energy_target='cohesive'``.

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
   Save predicted energies for train/test sets to disk.

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
   Controls where data and intermediate results are stored.

**device** : str (default: None)
   PyTorch device: ``'cpu'``, ``'cuda'``, or ``'cuda:0'``. Auto-detected if
   None.


Monitoring Training Progress
-----------------------------

The :class:`~aenet.io.train.TrainOut` object returned by ``train()`` provides
built-in visualization and analysis tools:

**Using Built-in Plotting Methods**

.. code-block:: python

   # Use TrainOut's built-in methods (recommended)
   results.plot_training_summary(outfile='training_summary.png')

   # Or plot only energy errors
   results.plot_training_errors(outfile='energy_errors.png')

   # For force training, plot force errors separately
   if results.has_force_data:
       results.plot_force_errors(outfile='force_errors.png')

**Manual Plotting with Full Control**

For custom plots, access the underlying data through the ``errors`` DataFrame:

.. code-block:: python

   import matplotlib.pyplot as plt

   # The errors DataFrame contains: RMSE_train, MAE_train, RMSE_test, MAE_test
   # and optionally RMSE_force_train, RMSE_force_test
   fig, axes = plt.subplots(1, 2 if results.has_force_data else 1,
                            figsize=(12, 4))
   if not results.has_force_data:
       axes = [axes]

   # Energy RMSE
   results.errors.plot(y=['RMSE_train', 'RMSE_test'], ax=axes[0], logy=True)
   axes[0].set_xlabel('Epoch')
   axes[0].set_ylabel('Energy RMSE (eV/atom)')
   axes[0].legend(['Train', 'Test'])

   # Force RMSE (if available)
   if results.has_force_data:
       results.errors.plot(y=['RMSE_force_train', 'RMSE_force_test'],
                          ax=axes[1], logy=True)
       axes[1].set_xlabel('Epoch')
       axes[1].set_ylabel('Force RMSE (eV/Å)')
       axes[1].legend(['Train', 'Test'])

   plt.tight_layout()
   plt.savefig('training_curves.png')

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
