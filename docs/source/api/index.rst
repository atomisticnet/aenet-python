API Reference
=============

This section provides detailed API documentation for the aenet-python package.

.. toctree::
   :maxdepth: 2

   trainset
   torch_featurize

Training Set Management
-----------------------

:doc:`trainset`
   Training set file handling and data loading. Includes support for
   neighbor information required for force training with PyTorch autograd.

PyTorch Featurization
---------------------

:doc:`torch_featurize`
   PyTorch-based atomic environment descriptors and neighbor lists.
   Provides GPU-accelerated featurization with automatic differentiation
   support.
