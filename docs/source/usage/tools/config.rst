``config``
==========

Alter and store settings.

.. note::

    See also the command-line help for all available
    flags and options with: ``aenet config --help``.


Inspecting the Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When `aenet config` is called without any arguments, it will print the
current configuration settings.


Configure ænet Fortran Binaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To make ``aenet-python`` aware of the ænet binaries, the paths need to
be configured.  The following command runs an interactive dialog that
works for standard installations

.. sourcecode:: console

   $ aenet config --set-aenet-path [path-to-aenet]

where ``[path-to-aenet]`` is the path pointing to the aenet root
directory.


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
