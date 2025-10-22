Migration to PyTorch Featurization
====================================

This guide shows how to migrate from the Fortran-based featurization to the
PyTorch implementation, which is a **drop-in replacement** that requires no
Fortran dependencies.

Overview
--------

The PyTorch-based featurization provides:

* **Same API** as the Fortran version
* **Same HDF5 output format** - fully compatible with TrnSet
* **No Fortran required** - pure Python/PyTorch
* **GPU acceleration** support
* **Automatic differentiation** for gradients
* **Validated accuracy** - matches Fortran to machine precision (< 1e-14)

Quick Start: Drop-in Replacement
---------------------------------

Old Code (Fortran-based)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from aenet.featurize import AenetAUCFeaturizer

   fzer = AenetAUCFeaturizer(['O', 'H'],
                             rad_cutoff=4.0, rad_order=10,
                             ang_cutoff=1.5, ang_order=3)

   fzer.run_aenet_generate(
       glob.glob("./xsf/*.xsf"),
       workdir='run',
       atomic_energies={'O': -10.0, 'H': -2.5}
   )

New Code (PyTorch-based)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from aenet.torch_featurize import TorchAUCFeaturizer

   fzer = TorchAUCFeaturizer(['O', 'H'],
                             rad_cutoff=4.0, rad_order=10,
                             ang_cutoff=1.5, ang_order=3)

   fzer.run_aenet_generate(
       glob.glob("./xsf/*.xsf"),
       workdir='run',
       atomic_energies={'O': -10.0, 'H': -2.5}
   )

**That's it!** Just change the import from ``aenet.featurize`` to
``aenet.torch_featurize`` and use ``TorchAUCFeaturizer`` instead of
``AenetAUCFeaturizer``.

Reading the Output
------------------

The output HDF5 file is **identical in format** to the Fortran version:

.. code-block:: python

   from aenet.trainset import TrnSet

   # Works with both Fortran and PyTorch-generated files
   with TrnSet.from_file('features.h5') as ts:
       print(ts)
       for struc in ts:
           print(struc.atom_features)

Complete Migration Example
---------------------------

Here's a full example showing the migration:

Before (Fortran)
~~~~~~~~~~~~~~~~

.. code-block:: python

   import glob
   from aenet.featurize import AenetAUCFeaturizer
   from aenet.trainset import TrnSet

   # Create featurizer
   fzer = AenetAUCFeaturizer(
       ['Li', 'Mo', 'Ni', 'Ti', 'O'],
       rad_cutoff=4.0, rad_order=10,
       ang_cutoff=1.5, ang_order=3
   )

   # Featurize structures (requires generate.x)
   fzer.run_aenet_generate(
       glob.glob("./xsf/*.xsf"),
       workdir='run',
       atomic_energies={
           'Li': -2.5197301758568920,
           'Mo': -0.6299325439642232,
           'Ni': -2.2047639038747695,
           'O': -10.0789207034275830,
           'Ti': -2.2047639038747695
       }
   )

   # Read features
   with TrnSet.from_file('features.h5') as ts:
       for struc in ts:
           features = struc.atom_features

After (PyTorch)
~~~~~~~~~~~~~~~

.. code-block:: python

   import glob
   from aenet.torch_featurize import TorchAUCFeaturizer
   from aenet.trainset import TrnSet

   # Create featurizer (identical API)
   fzer = TorchAUCFeaturizer(
       ['Li', 'Mo', 'Ni', 'Ti', 'O'],
       rad_cutoff=4.0, rad_order=10,
       ang_cutoff=1.5, ang_order=3
   )

   # Featurize structures (pure Python, no Fortran!)
   fzer.run_aenet_generate(
       glob.glob("./xsf/*.xsf"),
       workdir='run',
       atomic_energies={
           'Li': -2.5197301758568920,
           'Mo': -0.6299325439642232,
           'Ni': -2.2047639038747695,
           'O': -10.0789207034275830,
           'Ti': -2.2047639038747695
       }
   )

   # Read features (same API)
   with TrnSet.from_file('features.h5') as ts:
       for struc in ts:
           features = struc.atom_features

Additional Benefits
-------------------

GPU Acceleration
~~~~~~~~~~~~~~~~

Enable GPU acceleration by specifying the device:

.. code-block:: python

   fzer = TorchAUCFeaturizer(
       ['O', 'H'],
       rad_cutoff=4.0, rad_order=10,
       ang_cutoff=1.5, ang_order=3,
       device='cuda'  # Use GPU
   )

Direct Structure Featurization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also featurize structures directly without going through XSF files:

.. code-block:: python

   from aenet.geometry import AtomicStructure

   # Create or load structure
   struc = AtomicStructure(...)

   # Featurize directly
   result = fzer.featurize_structure(struc)

   # Access features
   features = [atom['fingerprint'] for atom in result['atoms']]

Batch Processing
~~~~~~~~~~~~~~~~

For multiple structures:

.. code-block:: python

   structures = [struc1, struc2, struc3, ...]
   results = fzer.featurize_structures(structures)

Convenience Function
~~~~~~~~~~~~~~~~~~~~

For quick featurization without creating a featurizer object:

.. code-block:: python

   from aenet.torch_featurize import featurize_and_write_hdf5

   featurize_and_write_hdf5(
       xsf_files=glob.glob("./xsf/*.xsf"),
       typenames=['O', 'H'],
       rad_order=10,
       rad_cutoff=4.0,
       ang_order=3,
       ang_cutoff=1.5,
       hdf5_filename='features.h5',
       atomic_energies={'O': -10.0, 'H': -2.5}
   )

Compatibility Notes
-------------------

Feature Values
~~~~~~~~~~~~~~

The PyTorch implementation produces **identical** feature values to the
Fortran version (within machine precision, < 1e-14 error). This has been
validated through comprehensive testing.

HDF5 Format
~~~~~~~~~~~

The HDF5 output format is **100% compatible** with the Fortran version:

* Same metadata structure
* Same dataset organization
* Same data types
* Readable by TrnSet without modification

Migration Checklist
-------------------

To migrate your workflow:

1. **Update imports**:

   .. code-block:: python

      # Change this:
      from aenet.featurize import AenetAUCFeaturizer

      # To this:
      from aenet.torch_featurize import TorchAUCFeaturizer

2. **Update class name**:

   .. code-block:: python

      # Change this:
      fzer = AenetAUCFeaturizer(...)

      # To this:
      fzer = TorchAUCFeaturizer(...)

3. **Keep everything else the same** - all method calls, parameters, and
   output handling remain identical

4. **Optional**: Add ``device='cuda'`` for GPU acceleration

5. **Test**: Verify your workflow produces identical results

When to Use Each Implementation
--------------------------------

Use **PyTorch** (recommended):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* New projects
* GPU acceleration needed
* Gradient computation required
* No Fortran dependencies desired
* Python-only environments

Use **Fortran**:
~~~~~~~~~~~~~~~~

* Legacy workflows
* Already have Fortran installed
* Need exact bit-for-bit reproducibility with old results

Performance Comparison
----------------------

For typical workloads:

* **CPU**: PyTorch and Fortran have similar performance
* **GPU**: PyTorch can be 5-10× faster for large systems (>100 atoms)
* **Batch processing**: PyTorch is more efficient for processing many structures

Troubleshooting
---------------

Import Error
~~~~~~~~~~~~

If you get an import error:

.. code-block:: bash

   pip install torch torch-scatter torch-cluster

Feature Value Differences
~~~~~~~~~~~~~~~~~~~~~~~~~

Feature values should match to machine precision. If you see larger differences:

1. Check that you're using the same parameters (rad_order, rad_cutoff, etc.)
2. Verify species lists are in the same order
3. Ensure structures have the same coordinates

Getting Help
------------

If you encounter issues:

* Check the :doc:`torch_featurization` documentation
* Review example notebooks in the repository
* Report issues on GitHub

References
----------

For more information:

* :doc:`torch_featurization` - Full PyTorch featurization guide
* :doc:`featurization` - Original Fortran-based guide
* :doc:`/api/torch_featurize` - API reference
