Neighbor Lists
==============

Overview
--------

The ``aenet-python`` package provides neighbor list functionality through the ``TorchNeighborList`` class in ``aenet.torch_featurize.neighborlist``.
The neighbor list is fully integrated with ``AtomicStructure`` objects and can be used both through high-level convenience methods and low-level direct access.

Quick Start
-----------

Using with AtomicStructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to use the neighbor list is through ``AtomicStructure.get_neighbors()``:

.. code-block:: python

   from aenet.geometry import AtomicStructure
   import numpy as np

   # Create a structure
   coords = np.array([
       [0.0, 0.0, 0.0],
       [1.5, 0.0, 0.0],
       [0.0, 1.5, 0.0]
   ])
   types = ['O', 'H', 'H']
   structure = AtomicStructure(coords, types)

   # Get neighbors of atom 0 within 2.0 Angstroms
   neighbors = structure.get_neighbors(i=0, cutoff=2.0)

   # neighbors is an AtomicStructure containing the neighboring atoms
   print(f"Found {neighbors.natoms} neighbors")

Direct Usage
~~~~~~~~~~~~

For more control, use ``TorchNeighborList`` directly:

.. code-block:: python

   from aenet.torch_featurize.neighborlist import TorchNeighborList
   import numpy as np

   # Create neighbor list
   nbl = TorchNeighborList(cutoff=4.0, device='cpu')

   # Find neighbors (accepts numpy arrays)
   positions = np.array([[0.0, 0.0, 0.0],
                         [1.5, 0.0, 0.0],
                         [3.0, 0.0, 0.0]])

   # Get neighbors of atom 0
   result = nbl.get_neighbors_of_atom(0, positions)

   neighbor_indices = result['indices']    # Which atoms are neighbors
   distances = result['distances']         # Distances to neighbors
   offsets = result['offsets']            # Cell offsets (None for isolated)

Configuration Options
---------------------

Cutoff Radius
~~~~~~~~~~~~~

The primary parameter controlling neighbor search:

.. code-block:: python

   # Standard cutoff
   nbl = TorchNeighborList(cutoff=4.0)

   # Larger cutoff for long-range interactions
   nbl = TorchNeighborList(cutoff=8.0)

Maximum Number of Neighbors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the neighbor list can handle up to 256 neighbors per atom. For dense systems, increase this limit:

.. code-block:: python

   # Standard systems
   nbl = TorchNeighborList(cutoff=4.0, max_num_neighbors=256)

   # Dense systems (liquids, high-pressure solids)
   nbl = TorchNeighborList(cutoff=4.0, max_num_neighbors=512)

   # Very dense systems
   nbl = TorchNeighborList(cutoff=4.0, max_num_neighbors=1024)

If you encounter errors about exceeding neighbors, increase this parameter.

Periodic Boundary Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For periodic systems, provide lattice vectors:

.. code-block:: python

   import numpy as np

   # FCC lattice
   a = 4.05  # Angstroms
   cell = np.array([
       [0.0, 0.5*a, 0.5*a],
       [0.5*a, 0.0, 0.5*a],
       [0.5*a, 0.5*a, 0.0]
   ])

   positions = np.array([[0.0, 0.0, 0.0]])  # Single atom

   nbl = TorchNeighborList(cutoff=4.0)
   result = nbl.get_neighbors_of_atom(0, positions, cell=cell)

   # Offsets show which periodic images each neighbor belongs to
   print(result['offsets'])  # e.g., [[0,0,1], [0,1,0], ...]

Advanced Features
-----------------

Getting Neighbor Coordinates Directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of manually computing neighbor positions from indices and offsets, use ``return_coordinates=True``:

.. code-block:: python

   nbl = TorchNeighborList(cutoff=4.0)

   # For isolated systems
   result = nbl.get_neighbors_of_atom(
       0, positions, return_coordinates=True
   )
   neighbor_coords = result['coordinates']  # Actual 3D positions

   # For periodic systems - offsets are automatically applied
   result = nbl.get_neighbors_of_atom(
       0, positions, cell=cell, return_coordinates=True
   )
   neighbor_coords = result['coordinates']  # PBC offsets already applied

This is especially convenient for periodic systems, where the neighbor list automatically applies cell offsets to compute actual Cartesian coordinates.

Type-Dependent Cutoffs
~~~~~~~~~~~~~~~~~~~~~~

For multi-component systems, different atom pairs may require different cutoffs:

.. code-block:: python

   import torch

   # Water system: O and H atoms
   atom_types = torch.tensor([8, 1, 1, 8, 1, 1])  # Atomic numbers

   # Define pair-specific cutoffs
   cutoff_dict = {
       (1, 1): 2.0,   # H-H: short cutoff
       (1, 8): 2.5,   # H-O: bond length range
       (8, 8): 3.5,   # O-O: longer cutoff
   }

   # Create neighbor list with type information
   nbl = TorchNeighborList(
       cutoff=5.0,              # Max cutoff (must be >= all pair cutoffs)
       atom_types=atom_types,
       cutoff_dict=cutoff_dict
   )

   # Neighbors are automatically filtered by type-specific cutoffs
   result = nbl.get_neighbors_of_atom(0, positions, cell=cell)

Per-Atom Neighbor Access
~~~~~~~~~~~~~~~~~~~~~~~~~

Get neighbors for all atoms at once:

.. code-block:: python

   # Returns list of neighbor dicts, one per atom
   all_neighbors = nbl.get_neighbors_by_atom(positions, cell=cell)

   for i, atom_neighbors in enumerate(all_neighbors):
       print(f"Atom {i}: {len(atom_neighbors['indices'])} neighbors")

Factory Method from AtomicStructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a neighbor list pre-configured for a structure:

.. code-block:: python

   from aenet.torch_featurize.neighborlist import TorchNeighborList

   # Create from AtomicStructure
   nbl = TorchNeighborList.from_AtomicStructure(
       structure,
       cutoff=4.0,
       frame=-1  # Use last frame
   )

GPU Acceleration
~~~~~~~~~~~~~~~~

For large systems, use GPU acceleration:

.. code-block:: python

   # Create neighbor list on GPU
   nbl = TorchNeighborList(cutoff=4.0, device='cuda')

   # Input arrays are automatically moved to GPU
   result = nbl.get_neighbors_of_atom(0, positions)

   # Results are on GPU - convert back to numpy if needed
   neighbor_indices = result['indices'].cpu().numpy()

Performance Considerations
--------------------------

Caching
~~~~~~~

The neighbor list automatically caches results:

.. code-block:: python

   nbl = TorchNeighborList(cutoff=4.0)

   # First call: computes neighbor list
   result1 = nbl.get_neighbors_of_atom(0, positions)

   # Subsequent calls with same positions: uses cache
   result2 = nbl.get_neighbors_of_atom(1, positions)
   result3 = nbl.get_neighbors_of_atom(2, positions)

   # Cache invalidated when positions change
   new_positions = positions + 0.1
   result4 = nbl.get_neighbors_of_atom(0, new_positions)  # Recomputes

GPU vs CPU
~~~~~~~~~~

- **CPU**: Best for small to medium systems (< 1000 atoms)
- **GPU**: Beneficial for large systems (> 1000 atoms) or when called frequently
- Data transfer overhead can make GPU slower for small systems

System Size Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------+--------------------+------------------------+
| Atoms          | Device             | max_num_neighbors      |
+================+====================+========================+
| < 100          | CPU                | 256 (default)          |
+----------------+--------------------+------------------------+
| 100-1000       | CPU                | 256-512                |
+----------------+--------------------+------------------------+
| 1000-10000     | GPU recommended    | 512-1024               |
+----------------+--------------------+------------------------+
| > 10000        | GPU                | 1024+                  |
+----------------+--------------------+------------------------+

Migration from Legacy Neighbor List
------------------------------------

The PyTorch-based neighbor list is a drop-in replacement for the legacy implementation:

.. code-block:: python

   # Old (legacy) approach - NO LONGER USED
   # from aenet.nblist import NeighborList
   # nbl = NeighborList(coords, lattice_vectors=avec, ...)

   # New (PyTorch) approach - CURRENT
   from aenet.torch_featurize.neighborlist import TorchNeighborList
   nbl = TorchNeighborList(cutoff=4.0)
   result = nbl.get_neighbors_of_atom(i, coords, cell=avec)

The ``AtomicStructure.get_neighbors()`` method automatically uses the new implementation, so existing code using this high-level interface works without changes.

API Reference
-------------

For detailed API documentation, see:

* :py:class:`aenet.torch_featurize.neighborlist.TorchNeighborList` - Main neighbor list class
* :py:meth:`aenet.geometry.AtomicStructure.get_neighbors` - High-level interface

Key Methods
~~~~~~~~~~~

**TorchNeighborList**

* ``__init__(cutoff, atom_types=None, cutoff_dict=None, device='cpu', dtype=torch.float64, max_num_neighbors=256)``
* ``get_neighbors_of_atom(atom_idx, positions, cell=None, pbc=None, return_coordinates=False)``
* ``get_neighbors_by_atom(positions, cell=None, pbc=None)``
* ``get_neighbors(positions, cell=None, pbc=None, fractional=True)``
* ``from_AtomicStructure(structure, cutoff, frame=-1, device='cpu', max_num_neighbors=256)`` (classmethod)

**AtomicStructure**

* ``get_neighbors(i, cutoff, return_self=True, frame=-1)``

Troubleshooting
---------------

Error: "Number of neighbors exceeds max_num_neighbors"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Increase the ``max_num_neighbors`` parameter:

.. code-block:: python

   nbl = TorchNeighborList(cutoff=4.0, max_num_neighbors=512)

Slow performance for small systems on GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use CPU instead - data transfer overhead dominates for small systems:

.. code-block:: python

   nbl = TorchNeighborList(cutoff=4.0, device='cpu')

Incorrect neighbor distances in periodic systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure you're providing the cell matrix:

.. code-block:: python

   # Correct
   result = nbl.get_neighbors_of_atom(0, positions, cell=cell)

   # Wrong (treats as isolated system)
   result = nbl.get_neighbors_of_atom(0, positions)

See Also
--------

* :doc:`/usage/torch_featurization` - PyTorch-based featurization
* :doc:`/usage/structure_manipulation` - Working with AtomicStructure
* `Example notebooks <https://github.com/atomisticnet/aenet-python/tree/master/notebooks>`_ - Jupyter notebook examples
