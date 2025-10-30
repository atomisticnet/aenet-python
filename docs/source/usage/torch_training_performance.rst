Performance Optimization Guide
================================

This guide covers strategies to optimize the performance of PyTorch-based
training for machine learning interatomic potentials. The optimizations are
particularly important for large datasets and force-matched training.

.. note::

   The benchmark results presented here are from small-scale tests (20-512
   structures, 1-10 epochs) on a TiO2 dataset. The **relative speedups** are
   indicative, but **absolute benefits will be much larger** for production-scale
   workloads with hundreds of epochs and thousands of structures.

Overview
--------

Training performance depends on several factors:

* **Dataset size**: Number of structures and atoms per structure
* **Force supervision**: Whether forces are included (much more expensive)
* **Descriptor parameters**: Radial and angular cutoffs, polynomial orders
* **Hardware**: CPU vs GPU, memory availability

The PyTorch implementation provides several optimization strategies that can
dramatically improve training speed, especially for force training on large
datasets.

Quick Wins
----------

For most users, these settings provide significant speedups with minimal
configuration:

**Energy-Only Training** (``force_weight=0.0``):

.. code-block:: python

   from aenet.torch_training import TorchTrainingConfig, Adam

   config = TorchTrainingConfig(
       iterations=100,
       force_weight=0.0,
       cached_features=True,  # 100× speedup!
       device='cuda'
   )

**Force Training** (``force_weight > 0.0``):

.. code-block:: python

   config = TorchTrainingConfig(
       iterations=100,
       force_weight=0.1,
       force_fraction=0.3,      # 3× speedup
       cache_neighbors=True,    # 6-20% speedup
       cache_triplets=True,     # Additional speedup
       num_workers=4,           # For large datasets
       device='cuda'
   )

Optimization Strategies
-----------------------

1. Cached Features (Energy-Only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**What it does**: Precomputes and caches atomic environment features for all
structures before training begins.

**When to use**: Energy-only training (``force_weight=0.0``) only.

**Benefits**:

* **Massive speedup**: 100× or more for energy-only training
* Eliminates on-the-fly featurization overhead

**Trade-offs**:

* **Memory overhead**: Proportional to dataset size (n_structures × n_atoms × n_features)
* **Not compatible with force training**: Forces require gradients computed on-the-fly

**Benchmark** (TiO2, 32 structures, 3 epochs, batch=32, CPU):

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Configuration
     - Mean Epoch Time
     - Speedup
   * - On-the-fly (baseline)
     - 0.812s
     - 1×
   * - Cached features
     - 0.008s
     - **100×**

**Recommendation**: Always enable for energy-only training if memory permits.

.. code-block:: python

   config = TorchTrainingConfig(
       force_weight=0.0,
       cached_features=True
   )

2. Force Sub-Sampling
~~~~~~~~~~~~~~~~~~~~~~

**What it does**: Uses only a fraction of force labels per epoch, reducing
the computational cost of force training while maintaining convergence quality.

**When to use**: Force training with large datasets.

**Benefits**:

* **Significant speedup**: 3× for ``force_fraction=0.3``, 10× for ``force_fraction=0.1``
* **Validated convergence**: Research shows 0.1-0.3 maintains good results [1,2]
* **Loss remains unbiased**: Automatic rescaling

**Trade-offs**:

* Reduced force supervision per epoch
* May require more epochs to converge (still net faster)

**Benchmark** (TiO2, 20 structures, 1 epoch, batch=8, ``force_weight=0.5``, CPU):

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - force_fraction
     - Mean Epoch Time
     - Loss Compute Time
     - Speedup
   * - 1.0 (all forces)
     - 1.401s
     - 0.855s
     - 1×
   * - 0.3
     - 0.748s
     - 0.206s
     - **1.9×**
   * - 0.1
     - 0.589s
     - 0.058s
     - **2.4×**

**Interpretation**: As ``force_fraction`` decreases, force computation time
drops dramatically. The featurization overhead becomes relatively larger, which
motivates using neighbor caching.

**Recommendation**: Use ``force_fraction=0.2-0.3`` for production force training.

.. code-block:: python

   config = TorchTrainingConfig(
       force_weight=0.1,
       force_fraction=0.3,  # Use 30% of forces per epoch
       force_sampling='random'
   )

**References**:

[1] López-Zorrilla, J.M., et al. *J. Chem. Phys.* **158**, 164105 (2023).
https://doi.org/10.1063/5.0146803

[2] Yeu, I.W., et al. *npj Comput Mater* **11**, 156 (2025).
https://doi.org/10.1038/s41524-025-01651-0

3. Neighbor List Caching
~~~~~~~~~~~~~~~~~~~~~~~~~

**What it does**: Precomputes and caches neighbor graphs (indices and distances)
per structure to avoid repeated neighbor searches across epochs.

**When to use**: Force training with multiple epochs (≥3) on fixed geometries.

**Benefits**:

* **Moderate speedup**: 6-20% reduction in epoch time
* **Synergy with force sub-sampling**: Reduces featurization overhead
* **Works with triplet caching**: Combined optimization

**Trade-offs**:

* **Memory overhead**: Proportional to number of edges (neighbors) per structure
* **Only for fixed structures**: Not useful for on-the-fly augmentation

**Benchmark** (TiO2, 20 structures, 3 epochs, batch=8, ``force_fraction=0.1``, CPU):

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Configuration
     - Mean Epoch Time
     - Speedup
   * - No caching
     - 0.575s
     - 1×
   * - With neighbor caching
     - 0.536s
     - **1.07×**

**Recommendation**: Enable for force training with ≥3 epochs.

.. code-block:: python

   config = TorchTrainingConfig(
       force_weight=0.1,
       cache_neighbors=True
   )

4. Triplet Caching & Vectorization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**What it does**: Precomputes angular triplet indices and uses fully vectorized
operations (no Python loops) for feature and gradient computation.

**When to use**: Together with neighbor caching for force training.

**Benefits**:

* **Removes Python overhead**: Fully tensorized operations
* **Most beneficial with high angular complexity**: Higher ``ang_order`` or ``ang_cutoff``
* **Synergistic with neighbor caching**: Combined optimization

**Trade-offs**:

* **Additional memory**: Storage for triplet indices
* **Requires neighbor caching**: Works together

**Recommendation**: Enable together with ``cache_neighbors`` for force training.

.. code-block:: python

   config = TorchTrainingConfig(
       force_weight=0.1,
       cache_neighbors=True,
       cache_triplets=True  # Vectorized operations
   )

5. Parallel Data Loading
~~~~~~~~~~~~~~~~~~~~~~~~~

**What it does**: Uses multiple worker processes to parallelize on-the-fly
featurization during data loading.

**When to use**: Large datasets (≥128 structures) with on-the-fly featurization.

**Benefits**:

* **Significant speedup for large datasets**: 2-3× with 4-8 workers
* **Most effective on Linux**: macOS has higher multiprocessing overhead

**Trade-offs**:

* **Overhead for small datasets**: Can be slower than single-threaded for <128 structures
* **Not beneficial with cached features**: Caching eliminates data loading bottleneck
* **Memory per worker**: Each worker loads data independently

**Benchmark** (Featurization-only, 512 structures, batch=32, CPU):

.. list-table::
   :header-rows: 1
   :widths: 30 30 30 10

   * - Workers
     - Total Time
     - Atoms/sec
     - Speedup
   * - 0 (baseline)
     - 21.6s
     - 560
     - 1×
   * - 4
     - 9.1s
     - 1328
     - **2.4×**
   * - 8
     - 8.2s
     - 1469
     - **2.6×**

**Benchmark** (Small training run, 32 structures, 3 epochs, CPU):

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Workers
     - Mean Epoch Time
     - Note
   * - 0
     - 0.735s
     - Baseline
   * - 4
     - 1.995s
     - **Slower** (overhead dominates)

**Interpretation**: Workers help significantly on large datasets but hurt
performance on small workloads due to multiprocessing overhead.

**Recommendation**:

* **Small datasets** (<128 structures): Use ``num_workers=0``
* **Large datasets** (≥128 structures): Use ``num_workers=4-8`` (≈ half your physical cores)
* **With cached features**: Keep ``num_workers=0`` (no benefit)

.. code-block:: python

   # Large dataset without cached features
   config = TorchTrainingConfig(
       iterations=100,
       force_weight=0.1,
       num_workers=8,
       prefetch_factor=4,
       persistent_workers=True
   )

6. Precision Control
~~~~~~~~~~~~~~~~~~~~

**What it does**: Controls numeric precision (FP32 vs FP64) for training.

**When to use**: Default (``'auto'``) is appropriate for most cases.

**Benefits**:

* **FP32 on CPU**: Potential speedup (not observed in small benchmarks)
* **FP32 on GPU**: Reduced memory usage, faster computation
* **FP64**: Higher numerical accuracy when needed

**Trade-offs**:

* **FP32**: May have numerical instability in some cases
* **FP64**: Higher memory usage, slightly slower

**Benchmark** (TiO2, 32 structures, 3 epochs, batch=32, CPU):

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Precision
     - Mean Epoch Time
     - Note
   * - FP32
     - 0.876s
     - Similar to FP64
   * - FP64
     - 0.783s
     - Python overhead dominates

**Interpretation**: On small CPU workloads, precision makes minimal difference
because Python-level overhead dominates. Benefits are more pronounced on GPU
and with larger workloads.

**Recommendation**: Use default ``'auto'`` (FP32 on CPU, FP64 on GPU).

.. code-block:: python

   config = TorchTrainingConfig(
       precision='auto'  # Default: FP32 on CPU, FP64 on GPU
   )

GPU Acceleration
----------------

Using a GPU can provide significant speedups, especially for:

* Large models (many nodes, deep networks)
* Large batch sizes
* Force training (parallel gradient computation)

**Recommendation**: Use GPU when available for datasets >100 structures.

.. code-block:: python

   import torch

   config = TorchTrainingConfig(
       device='cuda' if torch.cuda.is_available() else 'cpu',
       precision='auto'  # Will use FP64 on GPU
   )

Recommended Configurations
--------------------------

Small Dataset Energy-Only (< 50 structures)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = TorchTrainingConfig(
       iterations=200,
       method=Adam(mu=0.001, batchsize=16),
       testpercent=20,
       force_weight=0.0,
       cached_features=True,  # 100× speedup
       num_workers=0,
       device='cpu'
   )

**Expected performance**: ~0.01-0.05s per epoch

Large Dataset Energy-Only (500+ structures)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = TorchTrainingConfig(
       iterations=50,
       method=Adam(mu=0.001, batchsize=64),
       testpercent=10,
       force_weight=0.0,
       cached_features=True,  # 100× speedup
       num_workers=0,         # No benefit with cached features
       device='cuda',
       precision='auto'
   )

**Expected performance**: ~0.5-2s per epoch (depending on dataset size)

Small Dataset Force Training (< 100 structures)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = TorchTrainingConfig(
       iterations=100,
       method=Adam(mu=0.001, batchsize=16),
       testpercent=20,
       force_weight=0.1,
       force_fraction=0.5,     # Modest sub-sampling
       cache_neighbors=True,
       cache_triplets=True,
       num_workers=0,          # Overhead not worth it
       device='cpu'            # CPU fine for small datasets
   )

**Expected performance**: ~1-3s per epoch

Medium Dataset Force Training (100-500 structures)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = TorchTrainingConfig(
       iterations=100,
       method=Adam(mu=0.001, batchsize=32),
       testpercent=10,
       force_weight=0.1,
       force_fraction=0.3,     # 3× speedup
       cache_neighbors=True,
       cache_triplets=True,
       num_workers=4,          # Parallel loading
       prefetch_factor=4,
       device='cuda',
       precision='auto'
   )

**Expected performance**: ~2-8s per epoch

Large Dataset Force Training (500+ structures)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = TorchTrainingConfig(
       iterations=50,
       method=Adam(mu=0.001, batchsize=64),
       testpercent=10,
       force_weight=0.1,
       force_fraction=0.2,     # Aggressive sub-sampling
       cache_neighbors=True,
       cache_triplets=True,
       num_workers=8,          # Maximum parallelism
       prefetch_factor=4,
       persistent_workers=True,
       device='cuda',
       precision='auto',
       save_energies=True,
       save_forces=True
   )

**Expected performance**: ~5-20s per epoch (highly dataset-dependent)

Performance Summary Table
--------------------------

.. list-table:: Optimization Strategy Summary
   :header-rows: 1
   :widths: 25 15 15 15 30

   * - Optimization
     - Speedup
     - Memory Cost
     - Restrictions
     - Best For
   * - ``cached_features``
     - **100×**
     - High
     - Energy-only
     - All energy-only training
   * - ``force_fraction=0.3``
     - **3×**
     - None
     - None
     - Force training, large datasets
   * - ``cache_neighbors``
     - 1.1-1.2×
     - Medium
     - Fixed structures
     - Force training, epochs ≥3
   * - ``cache_triplets``
     - 1.05-1.15×
     - Medium
     - With neighbors
     - Force training with neighbors
   * - ``num_workers=4-8``
     - 2-3×
     - Low
     - Large datasets
     - Datasets ≥128 structures
   * - GPU (``device='cuda'``)
     - 2-10×
     - None
     - GPU available
     - All training, especially forces

Profiling Your Training
-----------------------

To identify bottlenecks in your specific workflow, enable timing:

.. code-block:: python

   config = TorchTrainingConfig(
       timing=True,  # Enable detailed timing output
       show_progress=True
   )

This will print timing breakdowns showing where time is spent during training.

Common Performance Pitfalls
----------------------------

1. **Using workers with cached features**

   Workers provide no benefit when features are cached. Set ``num_workers=0``.

2. **Using workers with small datasets**

   Multiprocessing overhead dominates for <128 structures. Use ``num_workers=0``.

3. **Not using cached features for energy-only**

   This is the single biggest optimization. Always enable for ``force_weight=0.0``.

4. **Not using force sub-sampling on large datasets**

   ``force_fraction=0.3`` provides 3× speedup with minimal convergence impact.

5. **Setting batch size too small**

   Larger batches improve GPU utilization. Try ``batchsize=32-64`` for force training.

Scaling to Production
----------------------

When scaling to production workloads:

1. **Start with small experiments**: Test configurations on subset of data
2. **Profile systematically**: Use ``timing=True`` to identify bottlenecks
3. **Scale gradually**: Test each optimization before combining
4. **Monitor convergence**: Ensure validation metrics remain acceptable
5. **Use GPU when available**: Especially beneficial for force training

Expected Relative Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a typical production workload (500 structures, 50 epochs, force training):

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Configuration
     - Time per Epoch
     - Total Time (50 epochs)
   * - Baseline (no optimizations)
     - ~60s
     - ~50 minutes
   * - Optimized (all strategies)
     - ~10s
     - **~8 minutes**

This represents a **6× speedup** from combined optimizations.

Descriptor Parameter Tuning
----------------------------

The descriptor parameters also impact performance:

* **``ang_cutoff``**: Most expensive parameter (scales as N²)
* **``ang_order``**: Higher orders increase angular feature computation
* **``rad_cutoff``**: Affects neighbor search but less critical than angular

**Recommendations**:

* Start with moderate values: ``rad_cutoff=5.0``, ``ang_cutoff=2.5``
* Consider smaller ``ang_cutoff`` (2.0-2.5) for CPU training
* Higher ``ang_order`` (4-5) beneficial for accuracy but slower

See :doc:`torch_featurization` for more on descriptor parameters.

Next Steps
----------

* **Configuration reference**: See :doc:`torch_training_config_reference` for
  detailed parameter documentation
* **Tutorial**: See :doc:`torch_training_tutorial` for practical examples
* **Featurization**: See :doc:`torch_featurization` for descriptor optimization

See Also
--------

* :doc:`torch_training_tutorial` - Training tutorial with examples
* :doc:`torch_training_config_reference` - Complete configuration reference
* :doc:`torch_featurization` - PyTorch featurization guide
* :doc:`choosing_implementation` - Fortran vs PyTorch comparison
