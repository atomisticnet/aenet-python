Structure Transformations
=========================

.. currentmodule:: aenet.geometry.transformations

The transformations module provides an iterator-based framework for
generating structural variations. All transformations work with
:class:`aenet.geometry.AtomicStructure` objects.

.. seealso::

   For usage examples and detailed guides, see:

   - :doc:`../usage/transformations_basic`
   - :doc:`../usage/transformations_advanced`

Base Classes
------------

.. autosummary::
   :toctree: generated/

   TransformationABC
   TransformationChain

Deterministic Transformations
-----------------------------

These transformations produce a fixed set of output structures for a given
input structure.

.. autosummary::
   :toctree: generated/

   AtomDisplacementTransformation
   CellVolumeTransformation
   CellTransformationMatrix
   IsovolumetricStrainTransformation
   UniaxialStrainTransformation
   ShearStrainTransformation
   OrthorhombicStrainTransformation
   MonoclinicStrainTransformation

Stochastic Transformations
--------------------------

These transformations generate random structures and support reproducibility
via random seeds.

.. autosummary::
   :toctree: generated/

   RandomDisplacementTransformation
   DOptimalDisplacementTransformation

Detailed API
------------

Base Classes
^^^^^^^^^^^^

.. autoclass:: TransformationABC
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: TransformationChain
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Deterministic Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: AtomDisplacementTransformation
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: CellVolumeTransformation
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: CellTransformationMatrix
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: IsovolumetricStrainTransformation
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: UniaxialStrainTransformation
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: ShearStrainTransformation
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: OrthorhombicStrainTransformation
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: MonoclinicStrainTransformation
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Stochastic Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RandomDisplacementTransformation
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: DOptimalDisplacementTransformation
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
