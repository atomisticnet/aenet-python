Sampling
========

.. currentmodule:: aenet.geometry.sampling

The sampling module contains backend-neutral sampling operations. Feature-row
selection returns source indices, while Taylor sampling creates local
``AtomicStructure`` configurations from force-bearing references. Neither API
requires PyTorch.

Taylor sampling
---------------

``aenet.geometry.sampling`` is the authoritative location for Taylor label
construction, validation, transformations, reproducibility, provenance, and
parent-aware splitting. See :doc:`../usage/taylor_sampling` for the neutral
workflow and PyTorch/HDF5 adapters.

.. autosummary::
   :toctree: generated/

   TaylorReference
   TaylorExpansionConfig
   TaylorSampleRecord
   TaylorSamplingResult
   taylor_energy
   generate_taylor_samples
   iter_taylor_records
   iter_taylor_structures
   split_reference_structures

Feature-row selection
---------------------

Representative sampling requires finite real-valued features. Random sampling
uses only the matrix shape. Both return source-row indices for externally
managed collections; they do not generate geometries or compute descriptors.

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

.. autoclass:: TaylorReference
   :no-index:

.. autoclass:: TaylorExpansionConfig
   :no-index:

.. autoclass:: TaylorSampleRecord
   :no-index:

.. autoclass:: TaylorSamplingResult
   :no-index:

.. autofunction:: taylor_energy
   :no-index:

.. autofunction:: generate_taylor_samples
   :no-index:

.. autofunction:: iter_taylor_records
   :no-index:

.. autofunction:: iter_taylor_structures
   :no-index:

.. autofunction:: split_reference_structures
   :no-index:
