``config``
==========

Alter and store settings.

MPI Configuration
"""""""""""""""""

The ``config`` tool supports configuring MPI parallelization for aenet executables.
If your aenet binaries are built with MPI support, you can enable parallel execution
for training and prediction tasks.

Enabling MPI Support
~~~~~~~~~~~~~~~~~~~~

Enable MPI parallelization:

.. code-block:: bash

    $ aenet config --enable-mpi

Disable MPI parallelization:

.. code-block:: bash

    $ aenet config --disable-mpi

Customizing the MPI Launcher
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the MPI launcher is set to ``mpirun -np {num_proc} {exec}``, where
``{num_proc}`` is replaced by the number of processes and ``{exec}`` is replaced
by the executable path.

You can customize this for different HPC systems:

.. code-block:: bash

    # For SLURM systems
    $ aenet config --set-mpi-launcher "srun -n {num_proc} {exec}"

    # For OpenMPI with specific options
    $ aenet config --set-mpi-launcher "mpirun --bind-to core -np {num_proc} {exec}"

    # For Intel MPI
    $ aenet config --set-mpi-launcher "mpiexec -n {num_proc} {exec}"

The launcher template must include both ``{num_proc}`` and ``{exec}`` placeholders.

Checking MPI Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

View the current configuration including MPI settings:

.. code-block:: bash

    $ aenet config

This displays all configuration options, including ``mpi_enabled`` and ``mpi_launcher``.

API Reference
""""""""""""""

.. autoclass:: aenet.commandline.aenet_config.Config
   :members:
   :private-members:
   :undoc-members:
