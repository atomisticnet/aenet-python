Choosing an Implementation
==========================

``aenet-python`` provides two implementations for training machine learning
interatomic potentials:

1. **Fortran-based**: Wraps the compiled ænet Fortran executables
2. **PyTorch-based**: Pure Python implementation using PyTorch

Both implementations are maintained and supported. This guide helps you choose
the right one for your needs.

Quick Recommendation
--------------------

**New users** should start with **PyTorch** for its ease of installation and
flexibility. Switch to Fortran only if you need maximum computational efficiency
or have existing Fortran-based workflows.

**Existing users** with Fortran installations can continue using it for
production work, but consider trying PyTorch for new projects or when GPU
acceleration would be beneficial.

Comparison Table
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - Feature
     - **Fortran Implementation**
     - **PyTorch Implementation**
   * - **Installation**
     - Requires compiled Fortran binaries (can be challenging)
     - ``pip install`` (Python-only, straightforward)
   * - **Dependencies**
     - Fortran compiler, BLAS/LAPACK, optionally MPI
     - PyTorch, torch-scatter, torch-cluster
   * - **CPU Performance**
     - Excellent (highly optimized)
     - Good (competitive with optimizations)
   * - **GPU Support**
     - No
     - Yes (native CUDA support)
   * - **Energy Training**
     - Very fast (~0.2s/iteration, precomputed features)
     - Very fast with caching (~0.01s/epoch)
   * - **Force Training**
     - Efficient
     - Efficient with optimizations (sub-sampling, caching)
   * - **Automatic Differentiation**
     - No (manual gradient implementation)
     - Yes (PyTorch autograd)
   * - **Flexibility**
     - Fixed to implemented algorithms
     - Easy to customize and extend
   * - **Parallelization**
     - MPI-based (CPU cores/nodes)
     - DataLoader workers + GPU parallelism
   * - **Memory Usage**
     - Low
     - Higher (but manageable with options)
   * - **Debugging**
     - Limited (compiled code)
     - Excellent (Python-level debugging)
   * - **Research/Development**
     - Difficult to modify
     - Easy to prototype new methods
   * - **Validation**
     - Battle-tested, widely published
     - Validated against Fortran (< 1e-14 error)
   * - **Documentation**
     - Extensive (Fortran manuals)
     - Growing (this documentation)
   * - **Output Compatibility**
     - ænet native format
     - Compatible + PyTorch model format
   * - **Best For**
     - Production HPC workflows, maximum CPU efficiency
     - Development, GPU acceleration, ease of use

Detailed Comparison
-------------------

Installation & Setup
~~~~~~~~~~~~~~~~~~~~

**Fortran**

Requires compilation of ænet Fortran code:

* Need Fortran compiler (gfortran, ifort)
* BLAS/LAPACK libraries
* Optionally MPI for parallelization
* Platform-specific build issues possible
* Configuration needed (``aenet config``)

**PyTorch**

Simple pip installation:

.. code-block:: bash

   pip install aenet-python[torch]

* Pure Python, no compilation
* Cross-platform (Linux, macOS, Windows)
* Works immediately after installation

**Winner**: PyTorch (significantly easier)

Performance
~~~~~~~~~~~

**Energy-Only Training**

* **Fortran**: ~0.2s per iteration (with precomputed features)
* **PyTorch**: ~0.01s per epoch (with ``cached_features=True``)

Both are very fast for energy-only training.

**Force Training**

* **Fortran**: Highly optimized, efficient implementation
* **PyTorch**: Competitive with optimizations:

  * ``force_fraction=0.3``: 3× speedup
  * ``cache_neighbors=True``: 6-20% speedup
  * ``cache_triplets=True``: Additional speedup

* **GPU acceleration**: PyTorch can be 2-10× faster on GPU

**Winner**: Fortran for CPU-only production; PyTorch with GPU for large-scale

Flexibility & Development
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fortran**

* Fixed algorithms (BFGS, LM, EKF, Adam, etc.)
* Difficult to modify or extend
* Requires recompilation for changes
* No easy access to internals

**PyTorch**

* Easy to customize loss functions
* Can implement new architectures
* Access to all intermediate values
* Pythonic API for experimentation
* Automatic differentiation for new features

**Winner**: PyTorch (much more flexible)

Parallelization
~~~~~~~~~~~~~~~

**Fortran**

* MPI-based parallelization
* Scales to HPC clusters
* Requires MPI-compiled binary
* Limited to CPU cores/nodes

**PyTorch**

* DataLoader workers for data loading
* GPU parallelism (inherent)
* Can use multiple GPUs
* Hybrid CPU+GPU strategies

**Winner**: Depends on hardware; Fortran for CPU clusters, PyTorch for GPUs

Memory Usage
~~~~~~~~~~~~

**Fortran**

* Minimal memory footprint
* Efficient data structures
* Optimized for CPU cache

**PyTorch**

* Higher baseline usage
* Caching options increase memory
* Can be managed with configuration:

  * ``cached_features=False`` for force training
  * ``precision='float32'`` to reduce memory
  * ``num_workers=0`` to avoid worker overhead

**Winner**: Fortran (lower memory usage)

Use Case Recommendations
------------------------

Choose Fortran If:
~~~~~~~~~~~~~~~~~~

* **You already have ænet installed** and working Fortran-based workflows
* **Maximum CPU efficiency is critical** for very large-scale production runs
* **You need proven, published methods** without modification
* **Running on HPC clusters** with MPI parallelization
* **Memory is extremely constrained** and you can't afford caching overhead
* **You're reproducing published results** that used Fortran ænet

Choose PyTorch If:
~~~~~~~~~~~~~~~~~~

* **You're new to aenet-python** and want the easiest setup
* **You have GPU access** and want to leverage it
* **You need to customize** training procedures or loss functions
* **You're developing new methods** or researching ML potentials
* **Installation simplicity matters** (no Fortran compiler needed)
* **You want Python-level debugging** and introspection
* **Automatic differentiation is valuable** for your workflow

Migration Between Implementations
----------------------------------

Featurization
~~~~~~~~~~~~~

Both implementations produce compatible HDF5 feature files. See
:doc:`migration_torch` for details on migrating featurization workflows.

Training
~~~~~~~~

Training workflows are **not directly compatible**:

* **Fortran**: Uses ``train.in`` input files, produces ``.nn`` files
* **PyTorch**: Uses Python API, produces ``.pt`` model files

You can convert between them for inference, but training configurations must
be rewritten when switching implementations.

Model Deployment
~~~~~~~~~~~~~~~~

Both implementations can be used for inference (energy/force prediction):

* **Fortran models** (``.nn``): Can be loaded in Python via wrappers
* **PyTorch models** (``.pt``): Native PyTorch inference

For production deployment, choose based on your runtime environment:

* **Fortran**: Better for integration with Fortran MD codes
* **PyTorch**: Better for Python-based workflows and GPU inference

Example Workflows
-----------------

Fortran Workflow
~~~~~~~~~~~~~~~~

.. code-block:: python

   from aenet.mlip import ANNPotential, TrainingConfig, Adam

   # Define architecture
   arch = {"Ti": [(30, 'tanh'), (30, 'tanh')],
           "O": [(30, 'tanh'), (30, 'tanh')]}

   # Create potential
   potential = ANNPotential(arch)

   # Configure training
   config = TrainingConfig(
       iterations=1000,
       method=Adam(mu=0.001, batchsize=32),
       testpercent=10
   )

   # Train (requires ænet installation)
   potential.train('data.train', config=config)

PyTorch Workflow
~~~~~~~~~~~~~~~~

.. code-block:: python

   from aenet.torch_training import train_model, TorchTrainingConfig, Adam
   from aenet.geometry import read_xsf
   import glob

   # Load structures
   structures = [read_xsf(f) for f in glob.glob("./xsf/*.xsf")]

   # Define architecture (same format)
   arch = {"Ti": [(30, 'tanh'), (30, 'tanh')],
           "O": [(30, 'tanh'), (30, 'tanh')]}

   # Configure training
   config = TorchTrainingConfig(
       iterations=100,
       method=Adam(mu=0.001, batchsize=32),
       testpercent=10,
       force_weight=0.1,
       device='cuda'
   )

   # Train (pure Python, no Fortran needed)
   history = train_model(
       structures=structures,
       architecture=arch,
       config=config,
       species=['Ti', 'O'],
       rad_order=10,
       rad_cutoff=5.0,
       ang_order=4,
       ang_cutoff=2.5,
       output_dir='./outputs'
   )

Performance Comparison Example
-------------------------------

For a TiO2 dataset (100 structures, 50 epochs):

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Scenario
     - Fortran
     - PyTorch
   * - **Energy-only (CPU)**
     - ~10s total
     - ~0.5s total (with caching)
   * - **Force training (CPU, baseline)**
     - ~1000s total
     - ~1500s total (no opt)
   * - **Force training (CPU, optimized)**
     - N/A
     - ~500s total
   * - **Force training (GPU)**
     - N/A
     - ~200s total

.. note::

   These are illustrative timings. Actual performance depends heavily on
   system specs, dataset characteristics, and configuration.

Common Questions
----------------

Can I use both implementations?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! They can coexist in the same environment. Use whichever is appropriate
for each task.

Which is more accurate?
~~~~~~~~~~~~~~~~~~~~~~~

Both produce equivalent results. The PyTorch implementation is validated
to match Fortran output within machine precision (< 1e-14 error).

Will Fortran be deprecated?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No. Both implementations are actively maintained. Fortran remains important
for users with existing workflows and HPC deployments.

Can I train with PyTorch and deploy with Fortran?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not directly, but model conversion may be possible in the future. Currently,
use the same implementation for training and deployment.

Should I retrain existing models with PyTorch?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not necessary. If your Fortran-trained models work well, continue using them.
PyTorch is recommended for new projects or when you need its specific features.

Getting Started
---------------

Fortran Path
~~~~~~~~~~~~

1. Install ænet Fortran binaries (see ænet documentation)
2. Configure paths: ``aenet config``
3. Follow :doc:`training` documentation

PyTorch Path
~~~~~~~~~~~~

1. Install PyTorch: ``pip install aenet-python[torch]``
2. Follow :doc:`torch_training_tutorial` documentation
3. See :doc:`torch_training_performance` for optimization

Next Steps
----------

* **PyTorch Tutorial**: :doc:`torch_training_tutorial`
* **Fortran Training**: :doc:`training`
* **Migration Guide**: :doc:`migration_torch`
* **Performance Guide**: :doc:`torch_training_performance`

See Also
--------

* :doc:`torch_training_tutorial` - PyTorch training tutorial
* :doc:`training` - Fortran-based training
* :doc:`migration_torch` - Migrating featurization workflows
* :doc:`torch_training_performance` - PyTorch performance optimization
* :doc:`torch_training_config_reference` - Complete PyTorch configuration
