torch_featurize API Reference
================================

The ``aenet.torch_featurize`` module provides PyTorch-based implementations
of atomic environment descriptors and neighbor lists.

Module Overview
---------------

.. currentmodule:: aenet.torch_featurize

Main Classes
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ChebyshevDescriptor
   BatchedFeaturizer
   TorchNeighborList

Basis Functions
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ChebyshevPolynomials
   RadialBasis
   AngularBasis

Detailed API
------------

Descriptors
~~~~~~~~~~~

.. autoclass:: ChebyshevDescriptor
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__

.. autoclass:: BatchedFeaturizer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__

Neighbor Lists
~~~~~~~~~~~~~~

.. autoclass:: TorchNeighborList
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Basis Functions
~~~~~~~~~~~~~~~

.. autoclass:: ChebyshevPolynomials
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__

.. autoclass:: RadialBasis
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__

.. autoclass:: AngularBasis
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__
