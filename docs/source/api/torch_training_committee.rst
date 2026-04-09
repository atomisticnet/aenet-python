.. _api_torch_training_committee:

Committee Training
==================

The committee-training API orchestrates multiple seeded
``TorchANNPotential`` runs while preserving the existing single-member
training primitive.

The main workflow is:

* train a committee with :meth:`TorchCommitteePotential.train`
* reload a saved committee run with :meth:`TorchCommitteePotential.from_directory`
  or load specific member files with :meth:`TorchCommitteePotential.from_files`
* aggregate PyTorch-side predictions with :meth:`TorchCommitteePotential.predict`
* export committee members to ``.nn.ascii`` files with
  :meth:`TorchCommitteePotential.to_aenet_ascii` for the Fortran-backed
  ensemble interfaces

.. currentmodule:: aenet.torch_training

Committee Config
----------------

.. autoclass:: TorchCommitteeConfig
   :members:
   :undoc-members:
   :show-inheritance:

Committee Potential
-------------------

.. autoclass:: TorchCommitteePotential
   :members:
   :undoc-members:
   :show-inheritance:

Committee Results
-----------------

.. autoclass:: TorchCommitteeTrainResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TorchCommitteeMemberResult
   :members:
   :undoc-members:
   :show-inheritance:
