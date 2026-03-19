PyTorch-Based Featurization
===========================

.. note::

   Featurization as described here makes use of PyTorch.  Make sure to
   install ænet with the ``[torch]`` extra as described in :doc:`installation`.

.. note::

   **Alternative**: For featurization using ænet's Fortran-based tools,
   see :doc:`featurization`.

Overview
--------

The ``aenet.torch_featurize`` module provides a pure Python/PyTorch
implementation of the Chebyshev descriptor (AUC method) for atomic
environments [1,2].  In contrast to the Fortran-based implementation,
this implementation exposes gradients via PyTorch's automatic
differentiation mechanism and GPU support.  On CPUs, it is typically
slower than the Fortran implementation.

The PyTorch implementation is a drop-in replacement for the traditional
Fortran-based featurization workflow and yields (numerically) identical
results.

[1] N. Artrith, A. Urban, and G. Ceder,
*Phys. Rev. B* **96**, 2017, 014112 (`link1 <https://doi.org/10.1103/PhysRevB.96.014112>`_).

[2] A. M. Miksch, T. Morawietz, J. Kästner, A. Urban, N. Artrith,
*Mach. Learn.: Sci. Technol.* **2**, 2021, 031001 (`link2 <http://doi.org/10.1088/2632-2153/abfd96>`_).


Example notebooks
-----------------

For a longer workflow-oriented walkthrough, including file-based input,
batch processing, gradient computation, and optional GPU execution, see
`example-04-torch-featurization.ipynb
<https://github.com/atomisticnet/aenet-python/blob/master/notebooks/example-04-torch-featurization.ipynb>`_.

Additional notebooks are available in the `notebooks
<https://github.com/atomisticnet/aenet-python/tree/master/notebooks>`_
directory within the repository.


Basic Usage
-----------

High-Level API with AtomicStructure Objects (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``TorchAUCFeaturizer`` class provides a high-level API that is compatible
with the Fortran-based ``AenetAUCFeaturizer``. This is the recommended approach
for most users as it works directly with ``AtomicStructure`` objects:

.. doctest::

   >>> import numpy as np
   >>> from aenet.geometry import AtomicStructure
   >>> from aenet.torch_featurize import TorchAUCFeaturizer

   >>> structure = AtomicStructure(
   ...     np.array([
   ...         [0.0, 0.0, 0.12],
   ...         [0.0, 0.76, -0.47],
   ...         [0.0, -0.76, -0.47],
   ...     ]),
   ...     ['O', 'H', 'H'],
   ... )
   >>> descriptor = TorchAUCFeaturizer(
   ...     typenames=['O', 'H'],
   ...     rad_order=10,
   ...     rad_cutoff=4.0,
   ...     ang_order=3,
   ...     ang_cutoff=1.5,
   ... )
   >>> featurized = descriptor.featurize_structure(structure)
   >>> featurized.atom_features.shape
   (3, 30)

The ``TorchAUCFeaturizer`` inherits from ``AtomicFeaturizer`` and returns
``FeaturizedAtomicStructure`` objects, providing full API compatibility with
the Fortran-based workflow. This makes it easy to switch between implementations
or integrate with existing code. For file-based input and longer multi-structure
workflows, prefer the notebook example above.

Low-Level API with PyTorch Tensors (For Advanced Users)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For advanced users who need direct access to PyTorch operations and gradients,
the ``ChebyshevDescriptor`` class provides a lower-level interface:

.. doctest::

   >>> import torch
   >>> from aenet.torch_featurize import ChebyshevDescriptor
   >>> descriptor = ChebyshevDescriptor(
   ...     species=['O', 'H'],
   ...     rad_order=10,
   ...     rad_cutoff=4.0,
   ...     ang_order=3,
   ...     ang_cutoff=1.5,
   ... )
   >>> positions = torch.tensor([
   ...     [0.0, 0.0, 0.12],
   ...     [0.0, 0.76, -0.47],
   ...     [0.0, -0.76, -0.47],
   ... ], dtype=torch.float64)
   >>> species = ['O', 'H', 'H']
   >>> features = descriptor.forward_from_positions(positions, species)
   >>> features.shape
   torch.Size([3, 30])

This low-level API is useful when you need gradient computation for
force training or other differentiable operations. The notebook example
extends this with an explicit gradient workflow.

Periodic Systems
~~~~~~~~~~~~~~~~

For crystals with periodic boundary conditions, use the low-level API:

.. doctest::

   >>> import torch
   >>> from aenet.torch_featurize import ChebyshevDescriptor
   >>> positions = torch.tensor([
   ...     [0.0, 0.0, 0.0],
   ...     [0.0, 2.0, 2.0],
   ...     [2.0, 0.0, 2.0],
   ...     [2.0, 2.0, 0.0],
   ... ], dtype=torch.float64)
   >>> species = ['Cu', 'Cu', 'Au', 'Au']
   >>> cell = torch.tensor([
   ...     [4.0, 0.0, 0.0],
   ...     [0.0, 4.0, 0.0],
   ...     [0.0, 0.0, 4.0],
   ... ], dtype=torch.float64)
   >>> pbc = torch.tensor([True, True, True], dtype=torch.bool)
   >>> descriptor = ChebyshevDescriptor(
   ...     species=['Au', 'Cu'],
   ...     rad_order=8,
   ...     rad_cutoff=3.5,
   ...     ang_order=5,
   ...     ang_cutoff=3.5,
   ... )
   >>> features = descriptor.forward_from_positions(
   ...     positions, species, cell=cell, pbc=pbc
   ... )
   >>> features.shape
   torch.Size([4, 30])

Or use the high-level API with ``TorchAUCFeaturizer`` which handles
periodic structures automatically from ``AtomicStructure`` objects.


GPU Acceleration
----------------

Enable GPU acceleration by specifying the device when creating the descriptor:

.. code-block:: python

   import torch
   from aenet.torch_featurize import ChebyshevDescriptor

   if torch.cuda.is_available():
       # Create descriptor on GPU
       descriptor = ChebyshevDescriptor(
           species=['O', 'H'],
           rad_order=10,
           rad_cutoff=4.0,
           ang_order=3,
           ang_cutoff=1.5,
           device='cuda',
       )

       # Input tensors are moved to the configured device internally
       features = descriptor.forward_from_positions(positions, species)

The complete GPU walkthrough is kept in
`example-04-torch-featurization.ipynb
<https://github.com/atomisticnet/aenet-python/blob/master/notebooks/example-04-torch-featurization.ipynb>`_
so the base docs-example job can remain CPU-only.

Both ``TorchAUCFeaturizer`` and ``ChebyshevDescriptor`` support GPU
acceleration via the ``device`` parameter.


Batch Featurization
-------------------

For efficient processing of multiple structures (e.g., during training), use the
``BatchedFeaturizer`` class which wraps a ``ChebyshevDescriptor`` and processes
structures in batch:

.. code-block:: python

   import torch
   from aenet.torch_featurize import ChebyshevDescriptor, BatchedFeaturizer

   # Create base descriptor
   descriptor = ChebyshevDescriptor(
       species=['O', 'H'],
       rad_order=10,
       rad_cutoff=4.0,
       ang_order=3,
       ang_cutoff=1.5
   )

   # Wrap in BatchedFeaturizer for efficient batch processing
   batch_featurizer = BatchedFeaturizer(descriptor)

   # Prepare batch of structures (different sizes allowed)
   batch_positions = [
       torch.tensor(
           [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
           dtype=torch.float64,
       ),
       torch.tensor(
           [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
           dtype=torch.float64,
       ),
   ]

   batch_species = [
       ['O', 'H', 'H'],
       ['O', 'H'],
   ]

   # Process entire batch at once
   features, batch_indices = batch_featurizer(batch_positions, batch_species)
   print(features.shape)            # torch.Size([5, 30])
   print(batch_indices.tolist())    # [0, 0, 0, 1, 1]

The ``BatchedFeaturizer`` returns:

* ``features``: Concatenated feature tensor of shape ``(total_atoms, n_features)``
* ``batch_indices``: Tensor indicating which structure each atom belongs to

This is particularly useful in training loops where you need to process batches
of structures efficiently. For periodic systems, you can also provide
``batch_cells`` and ``batch_pbc`` lists. The notebook example keeps the longer
batch, gradient, and GPU-oriented workflow in one place.

Performance Considerations
--------------------------

* **Angular cutoff** has the largest impact on performance (scales as N²)
* **GPU acceleration** most beneficial for systems with >100 atoms
* **Batch processing** with ``BatchedFeaturizer`` improves throughput
