PyTorch Taylor-Sampling Adapters
================================

.. currentmodule:: aenet.torch_training.taylor_sampling

These interfaces preserve the original :mod:`aenet.torch_training` workflow
for :class:`~aenet.torch_training.Structure` objects and HDF5 sources. The
sampling policy and label implementation are delegated to the authoritative
:mod:`aenet.geometry.sampling` API.

.. autosummary::
   :toctree: generated/

   TaylorSampleRecord
   TaylorSamplingResult
   TaylorSourceCollection
   generate_taylor_samples
   iter_taylor_records
   iter_taylor_structures
   split_reference_structures

See :doc:`../usage/taylor_sampling` for stable identity requirements,
parent-level filtering, persisted provenance, and energy-only training.

Detailed API
------------

.. autoclass:: TaylorSampleRecord
   :no-index:

.. autoclass:: TaylorSamplingResult
   :no-index:

.. autoclass:: TaylorSourceCollection
   :no-index:

.. autofunction:: generate_taylor_samples
   :no-index:

.. autofunction:: iter_taylor_records
   :no-index:

.. autofunction:: iter_taylor_structures
   :no-index:

.. autofunction:: split_reference_structures
   :no-index:
