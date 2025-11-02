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
   from aenet.torch_training import train_model, TorchTrainingConfig, Adam
   from aenet.geometry import read_xsf
   import glob

   # Load structures
   structures = [read_xsf(f) for f in glob.glob("./xsf/*.xsf")]

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

   # Configure training
   cfg = TorchTrainingConfig(
       iterations=100,
       method=Adam(mu=0.001, batchsize=32),
       testpercent=10,
       force_weight=0.0  # Energy-only
   )

   # Train model
   history = train_model(
       structures=structures,
       architecture=arch,
       descriptor=descr,
       config=cfg
   )

   print(f"Final train RMSE: {history['train_rmse'][-1]:.4f} eV/atom")
   print(f"Final test RMSE: {history['test_rmse'][-1]:.4f} eV/atom")

This trains a neural network potential using energies only, with 10% of
structures held out for validation.


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
       cached_features=True,  # If energy-only
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
       cached_features=True,  # 100× speedup for energy-only
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
       cache_neighbors=True,  # Cache neighbor lists
       cache_triplets=True,   # Vectorized operations
       num_workers=4,
       device='cuda'
   )

Monitoring Training Progress
-----------------------------

The training history provides diagnostic information:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Plot training curves
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

   # Energy RMSE
   ax1.plot(history['train_energy_rmse'], label='Train')
   ax1.plot(history['test_energy_rmse'], label='Test')
   ax1.set_xlabel('Epoch')
   ax1.set_ylabel('Energy RMSE (eV/atom)')
   ax1.legend()
   ax1.set_yscale('log')

   # Force RMSE (if force training)
   if 'train_force_rmse' in history:
       ax2.plot(history['train_force_rmse'], label='Train')
       ax2.plot(history['test_force_rmse'], label='Test')
       ax2.set_xlabel('Epoch')
       ax2.set_ylabel('Force RMSE (eV/Å)')
       ax2.legend()
       ax2.set_yscale('log')

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
* :doc:`torch_training_config_reference` - Complete configuration reference
* :doc:`choosing_implementation` - Fortran vs PyTorch comparison
