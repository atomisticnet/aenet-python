PyTorch Training Tutorial
==========================

This tutorial covers training machine learning interatomic potentials (MLIPs)
using the PyTorch-based implementation in ``aenet-python``. The PyTorch
implementation provides a pure Python workflow with GPU acceleration, automatic
differentiation for forces, and flexible optimization strategies.

.. note::

   For users familiar with the Fortran-based training workflow, see
   :doc:`choosing_implementation` for a comparison of the two approaches.

Overview
--------

The PyTorch training workflow consists of three main steps:

1. **Prepare structures**: Load atomic structures with energies (and optionally forces)
2. **Configure training**: Set up the model architecture and training parameters
3. **Train the model**: Run the training loop and save the trained potential

This tutorial demonstrates both **energy-only** and **force-matched** training.

Prerequisites
-------------

Before starting, ensure you have:

* PyTorch-featurized structures (see :doc:`torch_featurization`)
* Or raw atomic structure files (XSF, CIF, etc.)
* Reference energies (and optionally forces) from DFT calculations

Quick Start: Energy-Only Training
----------------------------------

Here's a minimal example for training an energy-only potential:

.. code-block:: python

   from aenet.torch_training import train_model, TorchTrainingConfig, Adam
   from aenet.geometry import read_xsf
   import glob

   # Load structures
   structures = [read_xsf(f) for f in glob.glob("./xsf/*.xsf")]

   # Define network architecture
   architecture = {
       'O': [(30, 'tanh'), (30, 'tanh')],
       'H': [(20, 'tanh'), (20, 'tanh')]
   }

   # Configure training
   config = TorchTrainingConfig(
       iterations=100,
       method=Adam(mu=0.001, batchsize=32),
       testpercent=10,
       force_weight=0.0  # Energy-only
   )

   # Train model
   history = train_model(
       structures=structures,
       architecture=architecture,
       config=config,
       output_dir='./outputs'
   )

   print(f"Final train RMSE: {history['train_rmse'][-1]:.4f} eV/atom")
   print(f"Final test RMSE: {history['test_rmse'][-1]:.4f} eV/atom")

This trains a neural network potential using energies only, with 10% of
structures held out for validation.

Understanding the Architecture
-------------------------------

The ``architecture`` dictionary defines the neural network for each atomic species:

.. code-block:: python

   architecture = {
       'O': [(30, 'tanh'), (30, 'tanh')],  # 2 hidden layers
       'H': [(20, 'tanh'), (20, 'tanh')]
   }

* **Keys**: Element symbols (must match species in your structures)
* **Values**: List of ``(nodes, activation)`` tuples for each hidden layer
* **Available activations**: ``'tanh'``, ``'relu'``, ``'gelu'``, ``'linear'``
* **Output layer**: Automatically added (1 node, linear activation)

Guidelines for architecture design:

* **Layer depth**: 2-3 hidden layers are typical
* **Layer width**: 20-50 nodes per layer for most systems
* **Species-specific**: You can use different architectures for different elements
* **Feature dimension**: First layer should accept descriptor features (typically 20-40)

Force-Matched Training
-----------------------

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

Descriptor Configuration
------------------------

You must specify the descriptor (featurization) parameters:

.. code-block:: python

   from aenet.torch_training import train_model

   history = train_model(
       structures=structures,
       architecture=architecture,
       config=config,
       # Descriptor parameters
       species=['O', 'H'],
       rad_order=10,
       rad_cutoff=4.0,
       ang_order=3,
       ang_cutoff=1.5,
       output_dir='./outputs'
   )

These parameters must match those used for featurization if you pre-computed
features. See :doc:`torch_featurization` for guidance on choosing descriptor
parameters.

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

Complete Workflow Example
--------------------------

Here's a complete example showing all steps from loading structures to
evaluating the trained model:

.. code-block:: python

   from aenet.torch_training import train_model, TorchTrainingConfig, Adam
   from aenet.geometry import read_xsf
   import glob
   import torch

   # 1. Load structures
   print("Loading structures...")
   xsf_files = sorted(glob.glob("./xsf/*.xsf"))
   structures = []

   for f in xsf_files:
       struc = read_xsf(f)
       # Structures must have energy attribute
       # For force training, also need forces attribute
       structures.append(struc)

   print(f"Loaded {len(structures)} structures")

   # 2. Define architecture
   architecture = {
       'Ti': [(30, 'tanh'), (30, 'tanh')],
       'O': [(30, 'tanh'), (30, 'tanh')]
   }

   # 3. Configure training
   config = TorchTrainingConfig(
       iterations=100,
       method=Adam(mu=0.001, batchsize=32),
       testpercent=10,
       force_weight=0.1,
       device='cuda' if torch.cuda.is_available() else 'cpu',
       save_energies=True,
       save_forces=True,
       show_progress=True
   )

   # 4. Train model
   print("Training model...")
   history = train_model(
       structures=structures,
       architecture=architecture,
       config=config,
       species=['Ti', 'O'],
       rad_order=10,
       rad_cutoff=5.0,
       ang_order=4,
       ang_cutoff=2.5,
       output_dir='./outputs'
   )

   # 5. Examine results
   print("\nTraining Summary:")
   print(f"Final train energy RMSE: {history['train_energy_rmse'][-1]:.4f} eV/atom")
   print(f"Final test energy RMSE: {history['test_energy_rmse'][-1]:.4f} eV/atom")

   if config.force_weight > 0:
       print(f"Final train force RMSE: {history['train_force_rmse'][-1]:.4f} eV/Å")
       print(f"Final test force RMSE: {history['test_force_rmse'][-1]:.4f} eV/Å")

   # 6. Trained model is saved to ./outputs/trained_model.pt
   print("\nTrained model saved to: ./outputs/trained_model.pt")

Training Output
---------------

The training process generates several output files in the specified
``output_dir``:

* **trained_model.pt**: Saved PyTorch model (can be loaded for inference)
* **history.json**: Training history (losses, RMSE values per epoch)
* **history.csv**: Same data in CSV format for plotting
* **energies.train.predicted**: Predicted energies for training set (if ``save_energies=True``)
* **energies.test.predicted**: Predicted energies for test set (if ``save_energies=True``)
* **forces.train.predicted**: Predicted forces for training set (if ``save_forces=True``)
* **forces.test.predicted**: Predicted forces for test set (if ``save_forces=True``)

Loading a Trained Model
------------------------

To use a trained model for inference:

.. code-block:: python

   from aenet.torch_training import load_model
   from aenet.geometry import read_xsf

   # Load the model
   model, descriptor = load_model('./outputs/trained_model.pt')

   # Predict energy for a new structure
   structure = read_xsf('new_structure.xsf')
   energy = model.predict_energy(structure, descriptor)
   print(f"Predicted energy: {energy:.4f} eV")

   # Predict energy and forces
   energy, forces = model.predict_energy_and_forces(structure, descriptor)
   print(f"Predicted energy: {energy:.4f} eV")
   print(f"Max force magnitude: {forces.abs().max():.4f} eV/Å")

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
       # Performance optimizations (see performance guide)
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

Troubleshooting
---------------

High Training Error
~~~~~~~~~~~~~~~~~~~

If the model doesn't achieve acceptable accuracy:

1. **Check data quality**: Ensure energies/forces are correctly assigned
2. **Increase model capacity**: Add more nodes or layers
3. **Adjust descriptor parameters**: Try larger cutoffs or higher orders
4. **Train longer**: Increase number of iterations
5. **Reduce learning rate**: Try ``mu=0.0001`` for Adam

Overfitting
~~~~~~~~~~~

If test error is much higher than train error:

1. **Increase training data**: Collect more reference structures
2. **Reduce model capacity**: Fewer nodes or layers
3. **Add regularization**: Set ``weight_decay > 0`` in optimizer
4. **Early stopping**: Monitor test error and stop when it starts increasing

Slow Training
~~~~~~~~~~~~~

If training is too slow:

1. **Use GPU**: Set ``device='cuda'`` if available
2. **Enable optimizations**: See :doc:`torch_training_performance`
3. **Increase batch size**: Try larger ``batchsize`` values
4. **Reduce force supervision**: Use ``force_fraction < 1.0``

Out of Memory
~~~~~~~~~~~~~

If you run out of GPU/CPU memory:

1. **Reduce batch size**: Smaller ``batchsize`` value
2. **Use CPU instead of GPU**: Set ``device='cpu'``
3. **Disable caching**: Set ``cached_features=False``
4. **Reduce precision**: Set ``precision='float32'``

Next Steps
----------

* **Performance optimization**: See :doc:`torch_training_performance` for
  strategies to speed up training
* **Configuration reference**: See :doc:`torch_training_config_reference` for
  detailed documentation of all training parameters
* **Advanced features**: Explore force sub-sampling, neighbor caching, and
  other optimization strategies
* **Model deployment**: Learn about exporting models for production use

See Also
--------

* :doc:`torch_featurization` - PyTorch-based structure featurization
* :doc:`torch_training_config_reference` - Complete configuration reference
* :doc:`torch_training_performance` - Performance optimization guide
* :doc:`choosing_implementation` - Fortran vs PyTorch comparison
