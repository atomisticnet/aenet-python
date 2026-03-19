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

For a longer workflow-oriented walkthrough, including the maintained TiO2
saved-model example, batched file-backed inference, and optional GPU
execution, see `example-06-torch-inference.ipynb
<https://github.com/atomisticnet/aenet-python/blob/master/notebooks/example-06-torch-inference.ipynb>`_.

Additional notebooks are available in the `notebooks
<https://github.com/atomisticnet/aenet-python/tree/master/notebooks>`_
directory within the repository.


Loading a Trained Model
------------------------

Use compact page-level examples for the core API, and prefer the notebook
linked above for larger file collections, performance tuning, and GPU-backed
workflows.

.. code-block:: python

   from aenet.mlip import PredictionConfig
   from aenet.torch_training import TorchANNPotential

   pot = TorchANNPotential.from_file("outputs/trained_model.pt")

   results = pot.predict(
       ["structure.xsf"],
       eval_forces=True,
       config=PredictionConfig(
           print_atomic_energies=True,
           timing=True,
       ),
   )

   print(f"Total energy: {results.total_energy[0]:.4f} eV")
   print(f"Cohesive energy: {results.cohesive_energy[0]:.4f} eV")
   print(results.forces[0].shape)
   print(results.atom_energies[0].shape)

The unified API returns a :class:`~aenet.io.predict.PredictOut` object containing:

* ``total_energy``: Total energies per structure
* ``cohesive_energy``: Cohesive energies per structure
* ``forces``: Atomic forces (if ``eval_forces=True``)
* ``atom_energies``: Per-atom energies (if ``config.print_atomic_energies=True``)
* ``coords``: Atomic coordinates
* ``atom_types``: Atomic species
* ``timing``: Per-structure timing breakdown (if ``config.timing=True``)

When ``timing=True``, the per-structure timing data is available through
``results.timing``.

This API is identical to the Fortran-based inference API, enabling seamless
interoperability between training in PyTorch and inference in Fortran.


Dataset-Backed Inference
------------------------

The PyTorch backend also provides a torch-only optimization path for
dataset-backed inference. The notebook linked above remains the maintained
home for the longer file-backed TiO2 workflow.

.. code-block:: python

   from aenet.geometry import AtomicStructure
   from aenet.mlip import PredictionConfig
   from aenet.torch_training.dataset import CachedStructureDataset

   structures = [AtomicStructure.from_file("structure.xsf")]

   ds = CachedStructureDataset(
       structures=structures,
       descriptor=pot.descriptor,
       show_progress=False,
   )

   results = pot.predict_dataset(
       ds,
       config=PredictionConfig(batch_size=32)
   )

This is useful when the dataset already stores precomputed ``features``.
For example, ``CachedStructureDataset`` can reuse cached feature tensors and
avoid featurizing the structures again during energy-only inference.

Notes:

* ``predict_dataset()`` is available only for ``TorchANNPotential``.
* The current implementation supports energy-only inference
  (``eval_forces=False``).
* The return type is still :class:`~aenet.io.predict.PredictOut`.
