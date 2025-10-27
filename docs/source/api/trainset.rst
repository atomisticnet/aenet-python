Training Set Management
=======================

.. automodule:: aenet.trainset
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

FeaturizedAtomicStructure
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: aenet.trainset.FeaturizedAtomicStructure
   :members:
   :undoc-members:
   :show-inheritance:

   .. attribute:: neighbor_info

      Optional dictionary containing neighbor information for force training.
      If present, contains:

      - ``neighbor_counts``: (n_atoms,) numpy array of neighbor counts per atom
      - ``neighbor_lists``: List of (nnb,) numpy arrays with neighbor atom indices
      - ``neighbor_vectors``: List of (nnb, 3) numpy arrays with displacement vectors

      This information is used for computing force derivatives during training.

      :type: dict or None

   .. autoproperty:: has_neighbor_info

      Returns True if neighbor information is available for force training.

      :return: Whether the structure contains neighbor information
      :rtype: bool

TrnSet
^^^^^^

.. autoclass:: aenet.trainset.TrnSet
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: has_neighbor_info

      Check if the training set file contains neighbor information.

      Neighbor information is only available for HDF5 format files that were
      generated with the ``include_neighbor_info=True`` option. This information
      is required for force training with PyTorch autograd.

      :return: True if neighbor information is available (only for HDF5 format), False otherwise
      :rtype: bool

      **Example:**

      .. code-block:: python

         from aenet.trainset import TrnSet

         # Load training set
         trnset = TrnSet.from_file('features_with_neighbors.h5')

         # Check if neighbor info is available
         if trnset.has_neighbor_info():
             print("Can use for force training")

             # Read structure with neighbor info
             struct = trnset.read_structure(0, read_coords=True, read_forces=True)

             # Access neighbor information
             neighbor_info = struct.neighbor_info
             print(f"Atom 0 has {neighbor_info['neighbor_counts'][0]} neighbors")

Usage Examples
--------------

Reading Training Sets with Neighbor Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When training with forces, you need HDF5 files that contain neighbor information:

.. code-block:: python

   from aenet.trainset import TrnSet

   # Load HDF5 file with neighbor information
   trnset = TrnSet.from_file('features_with_neighbors.h5')

   # Verify neighbor info is available
   assert trnset.has_neighbor_info(), "Neighbor info required for force training"

   # Iterate through structures
   for struct in trnset.iter_structures(read_coords=True, read_forces=True):
       # Each structure now has neighbor information
       if struct.has_neighbor_info:
           neighbor_info = struct.neighbor_info

           # Access per-atom neighbor data
           for i in range(struct.num_atoms):
               n_neighbors = neighbor_info['neighbor_counts'][i]
               neighbor_indices = neighbor_info['neighbor_lists'][i]
               neighbor_vectors = neighbor_info['neighbor_vectors'][i]

               print(f"Atom {i}: {n_neighbors} neighbors")

Generating Training Sets with Neighbor Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the PyTorch featurizer to generate HDF5 files with neighbor information:

.. code-block:: python

   from aenet.torch_featurize import TorchAUCFeaturizer

   # Set up featurizer
   featurizer = TorchAUCFeaturizer(
       typenames=['Ti', 'O'],
       rad_order=10,
       rad_cutoff=4.0,
       ang_order=3,
       ang_cutoff=1.5,
       device='cpu'
   )

   # Generate features WITH neighbor information
   featurizer.run_aenet_generate(
       xsf_files='structures/*.xsf',
       hdf5_filename='features_with_neighbors.h5',
       atomic_energies={'Ti': -1604.6, 'O': -432.5},
       include_neighbor_info=True,  # KEY: Enable neighbor info extraction
       output_file='generate.out'
   )

Backward Compatibility
^^^^^^^^^^^^^^^^^^^^^^

The implementation maintains full backward compatibility:

.. code-block:: python

   from aenet.trainset import TrnSet

   # Old HDF5 files without neighbor info still work
   trnset_old = TrnSet.from_file('features_old.h5')
   assert not trnset_old.has_neighbor_info()

   # Structures from old files have neighbor_info=None
   struct = trnset_old.read_structure(0)
   assert not struct.has_neighbor_info
   assert struct.neighbor_info is None

   # ASCII format also works (neighbor_info not supported)
   trnset_ascii = TrnSet.from_file('trainset.train.ascii')
   assert not trnset_ascii.has_neighbor_info()

See Also
--------

* :doc:`torch_featurize` - PyTorch-based featurization with neighbor info extraction
