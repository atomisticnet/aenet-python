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

Jupyter notebooks with examples how to use the featurization methods can
be found in the `notebooks
<https://github.com/atomisticnet/aenet-python/tree/master/notebooks>`_
directory within the repository.


Basic Usage
-----------

Featurize a water molecule:

.. code-block:: python

   import torch
   from aenet.torch_featurize import ChebyshevDescriptor

   # Create descriptor
   descriptor = ChebyshevDescriptor(
       species=['O', 'H'],
       rad_order=10,      # Radial polynomial order
       rad_cutoff=4.0,    # Radial cutoff (Angstroms)
       ang_order=3,       # Angular polynomial order
       ang_cutoff=1.5     # Angular cutoff (Angstroms)
   )

   # Featurize structure
   positions = torch.tensor([
       [0.0, 0.0, 0.12],   # O
       [0.0, 0.76, -0.47], # H
       [0.0, -0.76, -0.47] # H
   ], dtype=torch.float64)

   species = ['O', 'H', 'H']
   features = descriptor.featurize_structure(positions, species)
   # Returns: (3, 30) array - 3 atoms, 30 features each

Periodic Systems
~~~~~~~~~~~~~~~~

For crystals with periodic boundary conditions:

.. code-block:: python

   # Lattice vectors as rows
   cell = torch.tensor([
       [4.0, 0.0, 0.0],
       [0.0, 4.0, 0.0],
       [0.0, 0.0, 4.0]
   ], dtype=torch.float64)

   pbc = torch.tensor([True, True, True])

   features = descriptor.featurize_structure(
       positions, species, cell=cell, pbc=pbc
   )


GPU Acceleration
----------------

Enable GPU acceleration by specifying the device:

.. code-block:: python

   # Create descriptor on GPU
   descriptor = ChebyshevDescriptor(
       species=['O', 'H'],
       rad_order=10,
       rad_cutoff=4.0,
       ang_order=3,
       ang_cutoff=1.5,
       device='cuda'  # Use GPU
   )

   # Input tensors automatically moved to GPU
   features = descriptor.featurize_structure(positions, species)

Performance Considerations
--------------------------

* **Angular cutoff** has the largest impact on performance (scales as N²)
* **GPU acceleration** most beneficial for systems with >100 atoms
* **Batch processing** with ``BatchedFeaturizer`` improves throughput
