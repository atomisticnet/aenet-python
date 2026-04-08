.. _usage-inference:

Using ANN Potentials with ænet's Fortran Binaries
=================================================

.. note::

   Predictions/inference as described here make use of ænet's compiled
   ``predict.x`` tool.  Make sure to install ænet and configure the paths
   as described in :doc:`installation`.

.. note::

   **Alternative**: For a pure Python/PyTorch implementation that does not
   require Fortran, see :doc:`torch_inference`.

``aenet-python`` provides tools to to use ænet potentials for energy and
force predictions directly from Python scripts.  These features are
implemented primarily by the :class:`~aenet.mlip.ANNPotential` class.


Loading Existing Potentials
---------------------------

The easiest way to use pre-trained potentials is with the
:meth:`~aenet.mlip.ANNPotential.from_files` factory method:

.. code-block:: python

    from aenet.mlip import ANNPotential

    # Load binary format potentials
    potential_paths = {
        'Ti': 'Ti.nn',
        'O': 'O.nn'
    }
    potential = ANNPotential.from_files(potential_paths)

    # Load ASCII format potentials
    potential_paths = {
        'Ti': 'Ti.nn.ascii',
        'O': 'O.nn.ascii'
    }
    potential = ANNPotential.from_files(
        potential_paths,
        potential_format='ascii'
    )

This creates an ``ANNPotential`` instance configured for prediction-only use.

Note that architecture information cannot be extracted from the potential files,
so this instance cannot be used for re-training.

Right after Training
~~~~~~~~~~~~~~~~~~~~

If you just finished training in the same session, the potential automatically
knows where the ``.nn`` files are located:

.. code-block:: python

    # Train the potential
    arch = {'Ti': [(10, 'tanh'), (10, 'tanh')],
            'O': [(10, 'tanh'), (10, 'tanh')]}
    potential = ANNPotential(arch)
    potential.train(trnset_file='data.train', ...)

    # Predict immediately - no need to specify potential paths
    results = potential.predict(structures, eval_forces=True)


Making Predictions
------------------

The :meth:`~aenet.mlip.ANNPotential.predict` method is the main interface for
inference. It accepts either file paths to XSF structures or
:class:`~aenet.geometry.AtomicStructure` objects:

.. code-block:: python

    from aenet.geometry import AtomicStructure
    from aenet.mlip import ANNPotential, PredictionConfig

    # Load potential (format set at load time)
    potential = ANNPotential.from_files(
        {'Ti': 'Ti.nn', 'O': 'O.nn'}
    )

    # Predict from file paths - format remembered from from_files()
    results = potential.predict(['structure1.xsf', 'structure2.xsf'])

    # Or from AtomicStructure objects
    structures = [AtomicStructure.from_file(f) for f in xsf_files]
    results = potential.predict(structures)

Evaluating Forces
~~~~~~~~~~~~~~~~~

Forces can be computed by setting ``eval_forces=True``:

.. code-block:: python

    results = potential.predict(
        structures,
        eval_forces=True
    )

    # Access forces
    for i in range(results.num_structures):
        print(f"Structure {i}: forces shape = {results.forces[i].shape}")


Prediction Configuration
------------------------

The :class:`~aenet.mlip.PredictionConfig` class centralizes prediction
parameters with built-in validation:

*   ``potential_paths`` (Dict[str, str], optional): Mapping of element symbols to potential file paths. If None, uses paths stored after training. Default: ``None``
*   ``potential_format`` (str, optional): Format of potential files: ``'ascii'`` or ``None`` (binary). If ``'ascii'``, predict.x will automatically convert to binary. Default: ``None``
*   ``timing`` (bool): Enable detailed timing output. Default: ``False``
*   ``print_atomic_energies`` (bool): Print per-atom energy contributions. Default: ``False``
*   ``debug`` (bool): Enable debug output. Default: ``False``
*   ``verbosity`` (int): Verbosity level (0=low, 1=normal, 2=high). Default: ``1``

Example usage:

.. code-block:: python

    from aenet.mlip import PredictionConfig

    config = PredictionConfig(
        potential_format='ascii',       # ASCII format potentials
        verbosity=2,                    # High verbosity
        timing=True,                    # Enable timing
        print_atomic_energies=True      # Print per-atom energies
    )

    results = potential.predict(
        structures,
        eval_forces=True,
        config=config
    )

Working with Prediction Results
--------------------------------

The :class:`~aenet.io.predict.PredictOut` class provides convenient access
to prediction results:

.. code-block:: python

    # Access energies and forces
    cohesive_energies = results.cohesive_energy   # eV
    total_energies = results.total_energy         # eV
    forces = results.forces                       # eV/Å

    # Get number of structures and atoms
    n_structures = results.num_structures
    n_atoms = results.num_atoms(0)  # number of atomis in structure 0

Converting to AtomicStructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can convert prediction results back to :class:`~aenet.geometry.AtomicStructure`
objects with energies and forces attached:

.. code-block:: python

    # Convert results to structures
    structures_with_predictions = results.to_structures()

    # Access energy and forces from structure
    struc = structures_with_predictions[0]
    energy = struc.energy[-1]  # Latest energy
    forces = struc.forces[-1]  # Latest forces

Batch Predictions
-----------------

You can efficiently predict multiple structures at once:

.. code-block:: python

    import glob

    # Get all structure files
    xsf_files = glob.glob('structures/*.xsf')

    # Batch prediction
    results = potential.predict(
        xsf_files,
        eval_forces=False,  # Faster without forces
        config=PredictionConfig(verbosity=0)
    )

    # Analyze results
    import numpy as np
    energies = np.array(results.cohesive_energy)
    print(f"Mean energy: {energies.mean():.6f} eV/atom")
    print(f"Std energy: {energies.std():.6f} eV/atom")

MPI Parallelization for Predictions
------------------------------------

Similar to training, predictions can be accelerated using MPI parallelization
if the ``predict.x`` executable is built with MPI support.

Prerequisites are the same as for training:

1. The ``predict.x`` executable must be compiled with MPI support
2. MPI must be enabled in the configuration: ``aenet config --enable-mpi``
3. (Optional) Customize the MPI launcher if needed

Using MPI in Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~

To enable MPI parallelization, pass the ``num_processes`` parameter:

.. code-block:: python

    from aenet.mlip import ANNPotential, PredictionConfig

    # Load potential
    potential = ANNPotential.from_files({'Ti': 'Ti.nn', 'O': 'O.nn'})

    # Standard prediction (sequential)
    results = potential.predict(structures, eval_forces=True)

    # MPI prediction with 8 processes
    results = potential.predict(
        structures,
        eval_forces=True,
        num_processes=8
    )

Ensemble Predictions
--------------------

For uncertainty quantification, you can evaluate a committee of
independently trained potentials through the direct ``libaenet`` interfaces.
The default aggregation reports the ensemble mean energy and forces, together
with standard deviations and per-atom force uncertainties.

.. code-block:: python

    from aenet.geometry import AtomicStructure
    from aenet.mlip import AenetEnsembleInterface

    members = [
        {'Ti': 'Ti.nn.0', 'O': 'O.nn.0'},
        {'Ti': 'Ti.nn.1', 'O': 'O.nn.1'},
        {'Ti': 'Ti.nn.2', 'O': 'O.nn.2'},
    ]
    ensemble = AenetEnsembleInterface(members)

    structure = AtomicStructure.from_file('structure.xsf')
    result = ensemble.predict(structure, forces=True)

    print(f"Energy: {result.energy_mean:.6f} ± {result.energy_std:.6f} eV")
    print(f"Max force uncertainty: {result.force_uncertainty.max():.6f} eV/Å")

If you need continuity with an existing production model, you can keep one
member as the reported predictor while still computing committee statistics:

.. code-block:: python

    ensemble = AenetEnsembleInterface(
        members,
        aggregation='reference',
        reference_member=0,
    )
    result = ensemble.predict(structure, forces=True)

    # Reported values come from member 0
    print(result.energy)
    print(result.forces)

    # Committee statistics are still available
    print(result.energy_mean, result.energy_std)

ASE Ensemble Calculator
~~~~~~~~~~~~~~~~~~~~~~~

For ASE-driven simulations, ``AenetEnsembleCalculator`` reuses ASE's neighbor
list and exposes the ensemble result through the standard calculator API.

.. code-block:: python

    import ase.io
    from aenet.mlip import AenetEnsembleCalculator

    atoms = ase.io.read('structure.xsf')
    atoms.calc = AenetEnsembleCalculator(members, aggregation='mean')

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    print(atoms.calc.results['energy_std'])
    print(atoms.calc.results['force_uncertainty'].max())

The standard ASE properties ``energy`` and ``forces`` contain the aggregated
output. Additional uncertainty information is available in
``atoms.calc.results`` through the keys ``energy_mean``, ``energy_std``,
``forces_mean``, ``forces_std``, and ``force_uncertainty``.
