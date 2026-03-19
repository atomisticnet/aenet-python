Neighbor Lists
==============

Overview
--------

The ``aenet-python`` package provides neighbor list functionality through
the ``TorchNeighborList`` class exported by ``aenet.torch_featurize``.
The neighbor list is fully integrated with ``AtomicStructure`` objects
and can be used both through high-level convenience methods and low-level
direct access.

For a longer workflow-oriented walkthrough, including low-level edge access
and optional GPU execution, see ``notebooks/example-07-neighbor-list.ipynb``.

.. note::

   The features described here make use of PyTorch.  Make sure to
   install core torch support plus the matching ``torch-scatter`` and
   ``torch-cluster`` wheels as described in :doc:`/usage/installation`.  For
   use without PyTorch, ``aenet-python`` also provides a (less efficient)
   pure-Python neighbor list implementation in ``aenet.geometry.nblist``,
   which can be used with ``AtomicStructure`` objects.

Quick Start
-----------

Using with AtomicStructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to use the neighbor list is through
``AtomicStructure.get_neighbors()``:

.. doctest::

   >>> import numpy as np
   >>> from aenet.geometry import AtomicStructure
   >>> structure = AtomicStructure(
   ...     np.array([
   ...         [0.0, 0.0, 0.0],
   ...         [1.5, 0.0, 0.0],
   ...         [0.0, 1.5, 0.0],
   ...     ]),
   ...     ['O', 'H', 'H'],
   ... )
   >>> neighbors = structure.get_neighbors(i=0, cutoff=2.0)
   >>> neighbors.natoms
   3
   >>> neighbors.types.tolist()
   ['O', 'H', 'H']

Direct Usage
~~~~~~~~~~~~

For more control, use ``TorchNeighborList`` directly:

.. doctest::

   >>> import numpy as np
   >>> from aenet.torch_featurize import TorchNeighborList
   >>> nbl = TorchNeighborList(cutoff=4.0, device='cpu')
   >>> positions = np.array([
   ...     [0.0, 0.0, 0.0],
   ...     [1.5, 0.0, 0.0],
   ...     [3.0, 0.0, 0.0],
   ... ])
   >>> result = nbl.get_neighbors_of_atom(0, positions)
   >>> result['indices'].cpu().tolist()
   [1, 2]
   >>> [round(float(d), 1) for d in result['distances'].cpu().tolist()]
   [1.5, 3.0]
   >>> result['offsets'] is None
   True

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

By default, the neighbor list can handle up to 256 neighbors per atom.
For dense systems, increase this limit:

.. code-block:: python

   # Standard systems
   nbl = TorchNeighborList(cutoff=4.0, max_num_neighbors=256)

   # Dense systems (liquids, high-pressure solids)
   nbl = TorchNeighborList(cutoff=4.0, max_num_neighbors=512)

   # Very dense systems
   nbl = TorchNeighborList(cutoff=4.0, max_num_neighbors=1024)

If the number of neighbors exceeds this limit during a call, the limit
is automatically increased and a warning is issued.  However, this
should be avoided for performance reasons.

Periodic Boundary Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For periodic systems, provide lattice vectors. When ``cell`` is given,
positions are interpreted as fractional coordinates by default. Pass
``fractional=False`` when your coordinates are Cartesian:

.. code-block:: python

   import numpy as np
   from aenet.torch_featurize import TorchNeighborList

   cell = np.eye(3) * 5.0
   cartesian_positions = np.array([
       [0.5, 2.5, 2.5],
       [4.5, 2.5, 2.5],
   ])

   nbl = TorchNeighborList(cutoff=2.0, device='cpu')
   result = nbl.get_neighbors_of_atom(
       0,
       cartesian_positions,
       cell=cell,
       fractional=False,
   )

   # Offsets show which periodic images each neighbor belongs to
   print(result['offsets'])  # tensor([[-1, 0, 0]])

Advanced Features
-----------------

Getting Neighbor Coordinates Directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of manually computing neighbor positions from indices
and offsets, use ``return_coordinates=True``:

.. code-block:: python

   import numpy as np
   from aenet.torch_featurize import TorchNeighborList

   cell = np.eye(3) * 5.0
   cartesian_positions = np.array([
       [0.5, 2.5, 2.5],
       [4.5, 2.5, 2.5],
   ])
   nbl = TorchNeighborList(cutoff=2.0, device='cpu')

   # For isolated systems
   isolated_positions = np.array([
       [0.0, 0.0, 0.0],
       [1.5, 0.0, 0.0],
       [3.0, 0.0, 0.0],
   ])
   result = nbl.get_neighbors_of_atom(0, isolated_positions,
                                      return_coordinates=True)
   neighbor_coords = result['coordinates']  # Actual 3D positions

   # For periodic systems - offsets are automatically applied
   result = nbl.get_neighbors_of_atom(
       0,
       cartesian_positions,
       cell=cell,
       fractional=False,
       return_coordinates=True,
   )
   neighbor_coords = result['coordinates']  # PBC offsets already applied

This is especially convenient for periodic systems, where the neighbor
list automatically applies cell offsets to compute actual Cartesian coordinates.

Type-Dependent Cutoffs
~~~~~~~~~~~~~~~~~~~~~~

For multi-component systems, different atom pairs may require different cutoffs:

.. code-block:: python

   import numpy as np
   import torch
   from aenet.torch_featurize import TorchNeighborList

   positions = np.array([
       [0.00000, 0.00000, 0.11779],
       [0.00000, 0.75545, -0.47116],
       [0.00000, -0.75545, -0.47116],
   ])
   atom_types = torch.tensor([8, 1, 1])  # O, H, H

   # Define pair-specific cutoffs
   cutoff_dict = {
       (1, 1): 1.0,
       (1, 8): 2.5,
       (8, 8): 3.0,
   }

   # Create neighbor list with type information
   nbl = TorchNeighborList(
       cutoff=5.0,  # Must be >= every pair-specific cutoff
       atom_types=atom_types,
       cutoff_dict=cutoff_dict,
       device='cpu',
   )

   oxygen_neighbors = nbl.get_neighbors_of_atom(0, positions)
   hydrogen_neighbors = nbl.get_neighbors_of_atom(1, positions)

   print(oxygen_neighbors['indices'])   # tensor([1, 2])
   print(hydrogen_neighbors['indices'])  # tensor([0])

Per-Atom Neighbor Access
~~~~~~~~~~~~~~~~~~~~~~~~~

Get neighbors for all atoms at once:

.. code-block:: python

   import torch
   from aenet.torch_featurize import TorchNeighborList

   positions = torch.tensor([
       [0.0, 0.0, 0.0],
       [1.0, 0.0, 0.0],
       [0.0, 1.0, 0.0],
   ], dtype=torch.float64)
   nbl = TorchNeighborList(cutoff=1.5, device='cpu')

   # Returns list of neighbor dicts, one per atom
   all_neighbors = nbl.get_neighbors_by_atom(positions)

   for i, atom_neighbors in enumerate(all_neighbors):
       print(f"Atom {i}: {len(atom_neighbors['indices'])} neighbors")

Factory Method from AtomicStructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a neighbor list pre-configured for a structure:

.. doctest::

   >>> import numpy as np
   >>> from aenet.geometry import AtomicStructure
   >>> from aenet.torch_featurize import TorchNeighborList
   >>> structure = AtomicStructure(
   ...     np.array([
   ...         [0.0, 0.0, 0.0],
   ...         [1.5, 0.0, 0.0],
   ...         [0.0, 1.5, 0.0],
   ...     ]),
   ...     ['O', 'H', 'H'],
   ... )
   >>> nbl = TorchNeighborList.from_AtomicStructure(
   ...     structure,
   ...     cutoff=2.0,
   ...     device='cpu',
   ... )
   >>> nbl.cutoff
   2.0
   >>> nbl.max_num_neighbors
   256

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

   import torch
   from aenet.torch_featurize import TorchNeighborList

   positions = torch.tensor([
       [0.0, 0.0, 0.0],
       [1.0, 0.0, 0.0],
       [0.0, 1.0, 0.0],
   ], dtype=torch.float64)
   nbl = TorchNeighborList(cutoff=1.5, device='cpu')

   # First call: computes neighbor list
   result1 = nbl.get_neighbors_of_atom(0, positions)

   # Subsequent calls with same positions: uses cache
   result2 = nbl.get_neighbors_of_atom(1, positions)
   result3 = nbl.get_neighbors_of_atom(2, positions)

   # Cache invalidated when positions change
   new_positions = positions + 0.1
   result4 = nbl.get_neighbors_of_atom(0, new_positions)  # Recomputes


See Also
--------

* :doc:`/usage/torch_featurization` - PyTorch-based featurization
* :doc:`/usage/structure_manipulation` - Working with AtomicStructure
* `example-07-neighbor-list.ipynb <https://github.com/atomisticnet/aenet-python/blob/master/notebooks/example-07-neighbor-list.ipynb>`_ - longer low-level, GPU, and type-dependent neighbor-list examples
