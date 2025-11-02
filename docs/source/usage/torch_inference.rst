.. _usage-torch-inference:

Using ANN Potentials with the PyTorch Implementation
====================================================

.. note::

   Inference as described here makes use of PyTorch.  Make sure to
   install ænet with the ``[torch]`` extra as described in :doc:`installation`.

.. note::

   **Alternative**: For inference using ænet's Fortran-based tools,
   see :doc:`inference`.

.. warning::

   The PyTorch implementation is primarily intended for training
   neural network potentials.  For production inference workflows,
   the Fortran-based implementation, described in :doc:`inference`,
   can provide 20–50x faster performance depending on the hardware
   and system size.  Only for very large model sizes and when using
   GPUs for inference, the PyTorch implementation can be competitive
   in terms of speed.

Example notebooks
-----------------

Jupyter notebooks with examples can be found in the `notebooks
<https://github.com/atomisticnet/aenet-python/tree/master/notebooks>`_
directory within the repository.


Loading a Trained Model
------------------------

To use a trained model for inference:

.. code-block:: python

   from aenet.torch_training import TorchANNPotential
   from aenet.mlip import PredictionConfig

   # Load the model
   pot = TorchANNPotential.from_file('./outputs/trained_model.pt')

   # Predict energies for structures (accepts file paths or objects)
   results = pot.predict(['structure1.xsf', 'structure2.xsf'])

   print(f"Total energies: {results.total_energy}")
   print(f"Cohesive energies: {results.cohesive_energy}")

   # Predict with forces and per-atom energies
   results = pot.predict(
       ['structure1.xsf', 'structure2.xsf'],
       eval_forces=True,
       config=PredictionConfig(
           print_atomic_energies=True,
           timing=True
       )
   )

   print(f"Total energy: {results.total_energy[0]:.4f} eV")
   print(f"Cohesive energy: {results.cohesive_energy[0]:.4f} eV")
   print(f"Max force: {results.forces[0].max():.4f} eV/Å")
   print(f"Atom 0 energy: {results.atom_energies[0][0]:.4f} eV")

   # Access timing information (if requested)
   if results.timing:
       print(f"Featurization time: {results.timing['featurization'][0]:.4f} s")
       print(f"Energy eval time: {results.timing['energy_eval'][0]:.4f} s")
       print(f"Force eval time: {results.timing['force_eval'][0]:.4f} s")

The unified API returns a :class:`~aenet.io.predict.PredictOut` object containing:

* ``total_energy``: Total energies per structure
* ``cohesive_energy``: Cohesive energies per structure
* ``forces``: Atomic forces (if ``eval_forces=True``)
* ``atom_energies``: Per-atom energies (if ``config.print_atomic_energies=True``)
* ``coords``: Atomic coordinates
* ``atom_types``: Atomic species
* ``timing``: Per-structure timing breakdown (if ``config.timing=True``)

This API is identical to the Fortran-based inference API, enabling seamless
interoperability between training in PyTorch and inference in Fortran.
