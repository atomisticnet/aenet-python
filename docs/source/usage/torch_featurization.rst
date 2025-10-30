PyTorch-Based Featurization
=============================

Overview
--------

The ``aenet.torch_featurize`` module provides a pure Python/PyTorch
implementation of the Chebyshev descriptor (AUC method) for atomic
environments. This implementation offers several advantages:

* **No Fortran dependency** for featurization
* **GPU acceleration** support via PyTorch
* **Automatic differentiation** for force calculations
* **Validated accuracy** matching Fortran implementation (< 1e-14 error)
* **Efficient neighbor lists** using torch_cluster

The PyTorch implementation is a drop-in replacement for the traditional
Fortran-based featurization workflow.

Quick Start
-----------

Basic Usage
~~~~~~~~~~~

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

Key Concepts
------------

Descriptor Parameters
~~~~~~~~~~~~~~~~~~~~~

The main parameters controlling featurization:

* **species**: List of unique atomic species (e.g., ``['O', 'H']``)
* **rad_order**: Maximum order for radial Chebyshev polynomials (typical: 8-12)
* **rad_cutoff**: Radial cutoff radius in Angstroms (typical: 4-6 Å)
* **ang_order**: Maximum order for angular polynomials (typical: 3-5)
* **ang_cutoff**: Angular cutoff radius (typical: 1.5-3 Å, usually < rad_cutoff)

Feature Dimensions
~~~~~~~~~~~~~~~~~~

For multi-species systems, the number of features per atom is:

.. math::

   n_{features} = 2(n_{rad} + 1) + 2(n_{ang} + 1)

Example: For 2 species with ``rad_order=10`` and ``ang_order=3``:

* Radial features: 2 × 11 = 22
* Angular features: 2 × 4 = 8
* **Total**: 30 features per atom

Gradient Computation
--------------------

The PyTorch implementation supports automatic differentiation for
computing feature gradients with respect to atomic positions.
This is essential for force calculations and force-matched training.

Computing Forces
~~~~~~~~~~~~~~~~

For MD simulations or force-matched training:

.. code-block:: python

   import torch.nn as nn

   # Define energy model
   class EnergyModel(nn.Module):
       def __init__(self, n_features):
           super().__init__()
           self.net = nn.Sequential(
               nn.Linear(n_features, 64),
               nn.Tanh(),
               nn.Linear(64, 1)
           )

       def forward(self, features):
           return self.net(features)

   # Compute forces: F = -∂E/∂r
   model = EnergyModel(descriptor.get_n_features())
   energy, forces = descriptor.compute_forces_from_energy(
       positions, species, model
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
* Use ``torch.float64`` (default) for numerical accuracy

Common Use Cases
----------------

Training ML Potentials
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from aenet.torch_featurize import ChebyshevDescriptor, BatchedFeaturizer

   descriptor = ChebyshevDescriptor(['C', 'H', 'O'], 10, 5.0, 4, 2.5)
   batch_featurizer = BatchedFeaturizer(descriptor)

   # Featurize training structures
   features, batch_indices = batch_featurizer(
       batch_positions, batch_species
   )

   # Train model on features
   model.fit(features, energies)

Replacing Fortran Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PyTorch implementation is validated to match Fortran output:

.. code-block:: python

   # Old: Fortran-based
   # from aenet.featurize import Featurizer
   # features = Featurizer.featurize(structure, setup_file)

   # New: PyTorch-based (identical output)
   from aenet.torch_featurize import ChebyshevDescriptor
   descriptor = ChebyshevDescriptor(['O', 'H'], 10, 4.0, 3, 1.5)
   features = descriptor.featurize_structure(
       structure.positions, structure.species
   )

Next Steps
----------

After featurizing your structures, you can:

* **Train a model**: See :doc:`torch_training_tutorial` for PyTorch-based training
* **Optimize performance**: See :doc:`torch_training_performance` for training speedups
* **Use cached features**: Enable ``cached_features=True`` in training config for 100× speedup on energy-only training

API Reference
-------------

For detailed API documentation, see:

* :doc:`/api/torch_featurize` - Complete API reference
* Example notebooks in the `GitHub repository <https://github.com/atomisticnet/aenet-python/tree/master/notebooks>`_

Installation
------------

The PyTorch featurization requires:

.. code-block:: bash

   pip install torch torch-scatter torch-cluster

These are included in the standard ``aenet-python`` installation.

References
----------

The Chebyshev descriptor implementation is based on:

**N. Artrith, A. Urban, and G. Ceder**,
*Phys. Rev. B* **96**, 2017, 014112
https://doi.org/10.1103/PhysRevB.96.014112

More details can be found in:

**A. M. Miksch, T. Morawietz, J. Kästner, A. Urban, N. Artrith**,
*Mach. Learn.: Sci. Technol.* **2**, 2021, 031001
http://doi.org/10.1088/2632-2153/abfd96

The reference implementation is in the ænet Fortran codebase. The ænet
reference is:

**N. Artrith and A. Urban**,
*Comput. Mater. Sci.* **114**, 2016, 135-150
http://dx.doi.org/10.1016/j.commatsci.2015.11.047
