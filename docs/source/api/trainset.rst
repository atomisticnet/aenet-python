Training Set Management
=======================

.. automodule:: aenet.trainset

Classes
-------

FeaturizedAtomicStructure
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: aenet.trainset.FeaturizedAtomicStructure
   :members:
   :undoc-members:
   :show-inheritance:

   .. attribute:: neighbor_info
      :noindex:

      Optional dictionary containing neighbor information for force training.
      If present, contains:

      - ``neighbor_counts``: (n_atoms,) numpy array of neighbor counts per atom
      - ``neighbor_lists``: List of (nnb,) numpy arrays with neighbor atom indices
      - ``neighbor_vectors``: List of (nnb, 3) numpy arrays with displacement vectors

      This information is used for computing force derivatives during training.

      :type: dict or None

   .. autoproperty:: has_neighbor_info
      :noindex:

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
      :noindex:

      Check if the training set file contains neighbor information.

      Neighbor information is only available for HDF5 format files that were
      generated with the ``include_neighbor_info=True`` option. This information
      is required for force training with PyTorch autograd.

      :return: True if neighbor information is available (only for HDF5 format), False otherwise
      :rtype: bool

      **Example:**

      .. code-block:: python

         from aenet.trainset import TrnSet

         with TrnSet.from_file("features_with_neighbors.h5") as trnset:
             if trnset.has_neighbor_info():
                 struct = trnset.read_structure(
                     0,
                     read_coords=True,
                     read_forces=True,
                 )

                 if struct.has_neighbor_info:
                     print(struct.neighbor_info["neighbor_counts"][0])

      Use ``trnset.has_neighbor_info()`` to check whether the file stores
      neighbor-information tables at all, and ``struct.has_neighbor_info`` to
      check whether a particular returned structure exposes per-atom neighbor
      arrays.

Example Notebook
----------------

For the maintained end-to-end featurization workflows, including HDF5 export,
PyTorch-backed HDF5 compatibility, optional GPU execution, and longer
neighbor-information generation examples, see `example-01-featurization.ipynb
<https://github.com/atomisticnet/aenet-python/blob/master/notebooks/example-01-featurization.ipynb>`_.

Usage Examples
--------------

Inspecting an Existing Training Set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Keep this page focused on inspecting already-generated training sets. Prefer
the notebook linked above for file-backed featurization or generation
workflows.

.. code-block:: python

   from aenet.trainset import TrnSet

   with TrnSet.from_file("sample.h5") as trnset:
       print(trnset.num_structures)
       print(trnset.atom_types)

       struct = trnset[0]
       print(struct.num_atoms)
       print(struct.atom_features.shape)

Comparing HDF5 and ASCII Readers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both backends expose the same high-level ``TrnSet`` API for inspection:

.. code-block:: python

   from aenet.trainset import TrnSet

   with TrnSet.from_file("sample.h5") as trnset_h5, \
           TrnSet.from_file("sample.train.ascii") as trnset_ascii:
       struct_h5 = trnset_h5.read_structure(0, read_coords=True, read_forces=True)
       struct_ascii = trnset_ascii.read_structure(
           0,
           read_coords=True,
           read_forces=True,
       )

       assert trnset_h5.num_structures == trnset_ascii.num_structures
       assert trnset_h5.atom_types == trnset_ascii.atom_types
       assert struct_h5.atom_features.shape == struct_ascii.atom_features.shape
       assert struct_h5.coords.shape == struct_ascii.coords.shape

Checking Optional Neighbor Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use both the dataset-level and structure-level checks before consuming stored
neighbor arrays:

.. code-block:: python

   from aenet.trainset import TrnSet

   with TrnSet.from_file("features_with_neighbors.h5") as trnset:
       if trnset.has_neighbor_info():
           struct = trnset.read_structure(0, read_coords=True, read_forces=True)

           if struct.has_neighbor_info:
               print(struct.neighbor_info["neighbor_counts"][0])

Backward Compatibility
^^^^^^^^^^^^^^^^^^^^^^

The implementation maintains full backward compatibility:

.. code-block:: python

   from aenet.trainset import TrnSet

   with TrnSet.from_file("sample.h5") as trnset_h5:
       struct = trnset_h5.read_structure(0)
       assert not struct.has_neighbor_info
       assert struct.neighbor_info is None

   with TrnSet.from_file("sample.train.ascii") as trnset_ascii:
       assert not trnset_ascii.has_neighbor_info()

See Also
--------

* :doc:`/usage/torch_featurization` - PyTorch-based featurization APIs
