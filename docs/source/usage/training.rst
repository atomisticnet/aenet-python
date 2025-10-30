.. _usage-training:

Training ANN Potentials (Fortran)
==================================

.. note::

   This page documents the **Fortran-based** training implementation, which
   wraps the compiled ænet executables. For the **PyTorch-based** implementation
   (recommended for new users), see :doc:`torch_training_tutorial`.

   Both implementations are fully supported. See :doc:`choosing_implementation`
   for guidance on choosing between them.

``aenet-python`` provides tools to facilitate the training of `aenet`_
Artificial Neural Network (ANN) potentials directly from Python scripts.
This workflow is managed primarily by the :class:`~aenet.mlip.ANNPotential`
class.

.. _aenet: https://ann.atomistic.net

Defining the Network Architecture
---------------------------------

Before training, you need to define the architecture of the ANN for each
atomic species involved. This is done using a Python dictionary where
keys are the element symbols (e.g., "Si", "O") and values are lists of
tuples. Each tuple represents a layer in the network, specifying the
number of nodes and the activation function for that layer.

Supported activation functions are:
``'tanh'``, ``'linear'``, ``'relu'``, ``'gelu'``, ``'twist'``.

The final ANN layer is always a linear layer with one node, which
outputs the energy for the corresponding atomic species.  This layer
does not need to be defined.

Example architecture for a silicon potential:

.. code-block:: python

    from aenet.mlip import ANNPotential

    # Define architecture: Si with two hidden layers
    # (10 nodes, tanh activation)
    arch = {
        "Si": [(10, 'tanh'), (10, 'tanh')]
    }

    # Create the potential object
    potential = ANNPotential(arch)

Training Configuration
----------------------

Training parameters are managed through the :class:`~aenet.mlip.TrainingConfig`
class, which centralizes all configuration options with built-in validation.
This ensures type safety and prevents invalid parameter combinations.

The ``TrainingConfig`` class includes:

*   ``iterations`` (int): Maximum number of training iterations. Default: ``0``
*   ``method`` (TrainingMethod): The optimization algorithm to use. Default: ``Adam()``
*   ``testpercent`` (int): Percentage of data for test set (0-100). Default: ``0``
*   ``max_energy`` (float, optional): Exclude structures with energy above this threshold. Default: ``None``
*   ``sampling`` (str, optional): Sampling method ('sequential', 'random', 'weighted', 'energy'). Default: ``None``
*   ``timing`` (bool): Enable detailed timing output. Default: ``False``
*   ``save_energies`` (bool): Save predicted energies for training/test sets. Default: ``False``

The configuration validates parameters at creation time, raising ``ValueError``
for invalid inputs (e.g., testpercent outside 0-100 range, invalid sampling method).

Training Methods
----------------

The training process uses optimization methods to adjust the neural network
weights. Each method has specific parameters with sensible defaults.
The available training methods are provided as typed classes:

*   :class:`~aenet.mlip.Adam` - ADAM optimizer (default)
*   :class:`~aenet.mlip.BFGS` - L-BFGS-B optimizer (no parameters)
*   :class:`~aenet.mlip.EKF` - Extended Kalman filter
*   :class:`~aenet.mlip.LM` - Levenberg-Marquardt
*   :class:`~aenet.mlip.OnlineSD` - Online steepest descent

Each training method class encodes both the algorithm name and its parameters
with appropriate defaults based on the `aenet` Fortran implementation.

Training the Potential
----------------------

Once the architecture is defined, you can train the potential using
the :meth:`~aenet.mlip.ANNPotential.train` method. This method automates
several steps:

1.  Checks if the provided training set file exists and is compatible
    with the defined architecture.
2.  Creates a temporary working directory (or uses a specified one).
3.  Generates the necessary ``train.in`` file based on the architecture
    and training parameters.
4.  Calls the ``train.x`` executable from the configured `aenet` installation.
5.  Monitors the training progress with a progress bar.
6.  Collects the resulting potential files (``.nn`` files), energy files,
    and timing information into the current directory upon completion.

Basic Training Example:

.. code-block:: python

    from aenet.mlip import ANNPotential, TrainingConfig

    # Assuming 'potential' is an ANNPotential object defined as above
    # and 'data.train' is your training set file.

    # Simple training with defaults (uses Adam optimizer)
    potential.train('data.train')

    # Or customize parameters using TrainingConfig
    config = TrainingConfig(iterations=1000, testpercent=10)
    potential.train('data.train', config=config)

    # Inline configuration also works
    potential.train('data.train',
                   config=TrainingConfig(iterations=1000, testpercent=10))
    print("Training completed successfully.")

Using Different Training Methods:

.. code-block:: python

    from aenet.mlip import ANNPotential, TrainingConfig
    from aenet.mlip import BFGS, Adam, LM, EKF, OnlineSD

    # Use BFGS optimizer
    config = TrainingConfig(iterations=1000, method=BFGS())
    potential.train('data.train', config=config)

    # Customize Adam parameters
    config = TrainingConfig(
        iterations=1000,
        method=Adam(mu=0.005, batchsize=200),
        testpercent=10
    )
    potential.train('data.train', config=config)

    # Use Levenberg-Marquardt with additional options
    config = TrainingConfig(
        iterations=500,
        method=LM(batchsize=128, learnrate=0.05),
        sampling='random',
        max_energy=100.0
    )
    potential.train('data.train', config=config)

    # Use Extended Kalman filter
    config = TrainingConfig(
        iterations=500,
        method=EKF(lambda_=0.95, P=150.0),
        timing=True
    )
    potential.train('data.train', config=config)

    # Use Online steepest descent
    config = TrainingConfig(
        iterations=10000,
        method=OnlineSD(gamma=1e-6, alpha=0.3),
        save_energies=True
    )
    potential.train('data.train', config=config)


Key Parameters for ``train()``:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   ``trnset_file`` (str or Path, optional): Path to the training set file. Defaults to ``'data.train'``.
*   ``config`` (TrainingConfig, optional): Training configuration object containing all training parameters (iterations, method, testpercent, max_energy, sampling, timing, save_energies). If not provided, uses default ``TrainingConfig()`` with Adam optimizer. Defaults to ``None``.
*   ``workdir`` (str or Path, optional): A directory to store temporary files during training. If not provided, a temporary directory is created and removed afterwards. Defaults to ``None``.
*   ``output_file`` (str or Path, optional): File path to save the standard output of the ``train.x`` executable. Defaults to ``'train.out'``.

See the ``TrainingConfig`` class documentation above for all available configuration parameters.

Training Method Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Adam** (default method)

*   ``mu`` (float): Learning rate. Default: ``0.001``
*   ``b1`` (float): Exponential decay rate for first moment estimates. Default: ``0.9``
*   ``b2`` (float): Exponential decay rate for second moment estimates. Default: ``0.999``
*   ``eps`` (float): Small constant for numerical stability. Default: ``1.0e-8``
*   ``batchsize`` (int): Number of structures per batch. Default: ``16``
*   ``samplesize`` (int): Number of structures to sample per epoch. Default: ``100``

**BFGS**

*   No configurable parameters.
*   Note: Not supported on ARM-based Macs.

**EKF** (Extended Kalman Filter)

*   ``lambda`` (float): Forgetting factor. Default: ``0.99``
*   ``lambda0`` (float): Initial forgetting factor. Default: ``0.999``
*   ``P`` (float): Initial covariance. Default: ``100.0``
*   ``mnoise`` (float): Measurement noise. Default: ``0.0``
*   ``pnoise`` (float): Process noise. Default: ``1.0e-5``
*   ``wgmax`` (int): Maximum weight change. Default: ``500``

**LM** (Levenberg-Marquardt)

*   ``batchsize`` (int): Number of structures per batch. Default: ``256``
*   ``learnrate`` (float): Learning rate. Default: ``0.1``
*   ``iter`` (int): Number of iterations per epoch. Default: ``3``
*   ``conv`` (float): Convergence criterion. Default: ``1e-3``
*   ``adjust`` (int): Adjustment parameter. Default: ``5``

**OnlineSD** (Online Steepest Descent)

*   ``gamma`` (float): Learning rate. Default: ``1.0e-5``
*   ``alpha`` (float): Momentum parameter. Default: ``0.25``

This method requires a configured `aenet` installation.
Use ``aenet config`` on the command line to set the paths to the `aenet`
executables.

MPI Parallelization
-------------------

Training can be accelerated using MPI parallelization if the ``train.x``
executable is built with MPI support. This allows training to run across
multiple CPU cores or nodes on HPC systems.

Prerequisites
~~~~~~~~~~~~~

1. The ``train.x`` executable must be compiled with MPI support
2. MPI must be enabled in the aenet-python configuration:

.. code-block:: bash

    $ aenet config --enable-mpi

3. (Optional) Customize the MPI launcher for your system:

.. code-block:: bash

    # For SLURM systems
    $ aenet config --set-mpi-launcher "srun -n {num_proc} {exec}"

    # Default is "mpirun -np {num_proc} {exec}"

Using MPI in Training
~~~~~~~~~~~~~~~~~~~~~

To enable MPI parallelization, pass the ``num_processes`` parameter to the
``train()`` method:

.. code-block:: python

    from aenet.mlip import ANNPotential, TrainingConfig

    # Define architecture
    arch = {"Si": [(10, 'tanh'), (10, 'tanh')]}
    potential = ANNPotential(arch)

    # Standard training (sequential, no MPI)
    config = TrainingConfig(iterations=1000)
    potential.train('data.train', config=config)

    # MPI training with 8 processes
    config = TrainingConfig(iterations=1000)
    potential.train('data.train', config=config, num_processes=8)

    # MPI training with custom configuration
    config = TrainingConfig(
        iterations=1000,
        method=Adam(mu=0.005, batchsize=32),
        testpercent=10
    )
    potential.train('data.train', config=config, num_processes=16)

The ``num_processes`` parameter specifies how many MPI processes to use.
The actual command executed will be based on the configured MPI launcher.
For example, with the default launcher and ``num_processes=8``, the
command would be:

.. code-block:: bash

    mpirun -np 8 /path/to/train.x train.in
