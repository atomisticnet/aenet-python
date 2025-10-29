API Reference
=============

This section provides detailed API documentation for the aenet-python package.

.. toctree::
   :maxdepth: 2

   trainset
   torch_featurize
   torch_training_builders
   torch_training_training
   torch_training_inference

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

PyTorch Training (Modular Components)
--------------------------------------

:doc:`torch_training_builders`
   Network and optimizer builder utilities for constructing training components
   from configuration specifications.

:doc:`torch_training_training`
   Core training loop components including checkpoint management, metrics tracking,
   normalization, and epoch execution.

:doc:`torch_training_inference`
   Inference and prediction utilities for trained models.
