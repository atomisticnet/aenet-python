Representative Sampling
=======================

.. currentmodule:: aenet.geometry.sampling

The sampling module provides row-selection utilities for numeric structure
representations.  These functions return source-row indices for the caller to
apply to paths, in-memory structures, training sets, HDF5-backed datasets, or
other externally managed collections.

The functions do not generate atomic geometries, return
``AtomicStructure`` objects, compute descriptors, or scale features.

.. autosummary::
   :toctree: generated/

   representative_subset
   random_subset

Detailed API
------------

.. autofunction:: representative_subset
   :no-index:

.. autofunction:: random_subset
   :no-index:
