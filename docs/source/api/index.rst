API Reference
=============

This section provides detailed API documentation for the aenet-python package.

.. toctree::
   :maxdepth: 2

   reference_energies
   trainset
   transformations
   sampling
   torch_featurize
   torch_training_taylor
   torch_training_builders
   torch_training_committee
   torch_training_training
   torch_training_inference

Reference Energies
------------------

:doc:`reference_energies`
   Helper API for constructing atomic reference energies from regression and
   other supported workflows.

Training Set Management
-----------------------

:doc:`trainset`
   Training set file handling and data loading. Includes support for
   neighbor information required for force training with PyTorch autograd.

Structure Transformations
--------------------------

:doc:`transformations`
   Structure transformation framework for generating structural variations.
   Includes deterministic transformations (displacement, volume, strain)
   and stochastic transformations.

Sampling
--------

:doc:`sampling`
   Backend-neutral Taylor augmentation plus feature-matrix sampling utilities.

PyTorch Featurization
---------------------

:doc:`torch_featurize`
   PyTorch-based atomic environment descriptors and neighbor lists.
   Provides GPU-accelerated featurization with automatic differentiation
   support.

PyTorch Training (Modular Components)
--------------------------------------

:doc:`torch_training_taylor`
   Compatibility adapters for Taylor sampling with PyTorch structures,
   streaming sources, and HDF5 datasets.

:doc:`torch_training_builders`
   Network and optimizer builder utilities for constructing training components
   from configuration specifications.

:doc:`torch_training_committee`
   Committee-training orchestration for training multiple seeded PyTorch
   members with shared settings, reloadable metadata, PyTorch-side
   aggregation, and ASCII export interoperability.

:doc:`torch_training_training`
   Core training loop components including checkpoint management, metrics tracking,
   normalization, and epoch execution.

:doc:`torch_training_inference`
   Inference and prediction utilities for trained models.
