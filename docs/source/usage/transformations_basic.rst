Basic Structure Transformations
================================

The :mod:`aenet.geometry.transformations` module provides tools for generating
structural variations from input structures. This is useful for:

- Creating training data for machine learning potentials
- Exploring configuration space around a reference structure
- Generating diverse structural datasets
- Testing model robustness to structural perturbations

Basic Concepts
--------------

Transformations take an input structure and yield transformed structures using
Python's iterator protocol. All transformations work with
:class:`aenet.geometry.AtomicStructure` objects.

**Key terminology:**

- **Deterministic transformation**: Always produces the same output for a given input
- **Stochastic transformation**: Uses randomness; output varies unless a seed is provided

Iterator-Based Design
---------------------

All transformations return iterators, which provides memory efficiency and flexibility:

.. code-block:: python

   from aenet.geometry import AtomicStructure
   from aenet.geometry.transformations import AtomDisplacementTransformation

   structure = AtomicStructure.from_file('structure.xsf')
   transform = AtomDisplacementTransformation(displacement=0.05)

   # Get all structures at once
   all_structures = list(transform.apply_transformation(structure))
   print(f"Generated {len(all_structures)} structures")

   # Or process lazily (memory-efficient)
   for s in transform.apply_transformation(structure):
       process(s)  # Process one structure at a time

   # Or get first N structures
   import itertools
   first_10 = list(itertools.islice(
       transform.apply_transformation(structure), 10
   ))

The iterator pattern allows you to:

- Process structures one at a time without loading all into memory
- Stop early if you find what you need
- Compose with other iterators using ``itertools``

Atom Displacement
-----------------

:class:`AtomDisplacementTransformation` displaces each atom in Cartesian
x, y, and z directions. For a structure with N atoms, this generates 3N
output structures.

**Use cases:**

- Finite difference calculation of forces
- Exploring local energy landscape
- Testing force field sensitivity

**Example:**

.. code-block:: python

   from aenet.geometry.transformations import AtomDisplacementTransformation

   # Create transformation with 0.05 Angstrom displacement
   transform = AtomDisplacementTransformation(displacement=0.05)

   # Get all 3N structures
   displaced_structures = list(transform.apply_transformation(structure))
   print(f"Generated {len(displaced_structures)} structures")
   # Output: Generated 24 structures (for 8-atom structure)

   # Or process one at a time
   for i, s in enumerate(transform.apply_transformation(structure)):
       s.write_to_file(f'displaced_{i:03d}.xsf')

The displacement magnitude is in Angstroms and should be small enough to
remain in the harmonic regime for force calculations (typically 0.01-0.1 Å).

Cell Volume Scaling
-------------------

:class:`CellVolumeTransformation` uniformly scales the unit cell volume
while preserving fractional coordinates. The scaling is controlled by
percentage changes from the original volume.

All transformations on this page that modify the unit cell preserve
fractional coordinates. Since :class:`aenet.geometry.AtomicStructure`
stores Cartesian coordinates, the Cartesian positions are recomputed
from the updated cell, and copied energy/force labels are cleared on
the generated structures because they are stale after deformation.

**Use cases:**

- Equation of state calculations
- Pressure-volume relationships
- Testing volume-dependent properties

**Example:**

.. code-block:: python

   from aenet.geometry.transformations import CellVolumeTransformation

   # Scale volume from -5% to +5% in 5 steps
   transform = CellVolumeTransformation(
       min_percent=-5.0,
       max_percent=5.0,
       steps=5
   )

   original_volume = structure.cellvolume()

   for i, s in enumerate(transform.apply_transformation(structure)):
       new_volume = s.cellvolume()
       percent_change = 100 * (new_volume - original_volume) / original_volume
       print(f"Structure {i}: V = {new_volume:.2f} Å³ ({percent_change:+.1f}%)")

**Physics:** The volume scales as :math:`V_{\text{new}} = V_{\text{old}} \times (1 + p/100)^3`
where :math:`p` is the percentage change. Lattice vectors scale uniformly by
:math:`s = (1 + p/100)`.

Isovolumetric Strain
--------------------

:class:`IsovolumetricStrainTransformation` applies strain along one lattice
direction while adjusting the other two directions to preserve volume. This
is useful for exploring shape changes at constant volume.

**Use cases:**

- Constant-volume optimization
- Studying anisotropic mechanical properties
- Shape-dependent property calculations

**Example:**

.. code-block:: python

   from aenet.geometry.transformations import IsovolumetricStrainTransformation

   # Strain direction 1 (a-axis) from 0.9× to 1.1× original length
   transform = IsovolumetricStrainTransformation(
       direction=1,  # 1=a, 2=b, 3=c
       len_min=0.9,
       len_max=1.1,
       steps=5
   )

   original_volume = structure.cellvolume()

   for i, s in enumerate(transform.apply_transformation(structure)):
       new_volume = s.cellvolume()
       volume_error = abs(new_volume - original_volume)
       print(f"Structure {i}: ΔV = {volume_error:.2e} Å³")

**Physics:** When direction :math:`i` is scaled by :math:`s`, the orthogonal
directions are scaled by :math:`s_\perp = s^{-1/2}` to maintain
:math:`\det(\mathbf{M}) = 1`, where :math:`\mathbf{M}` is the transformation
matrix. Volume is conserved within numerical tolerance (typically < 10⁻⁵ Å³).

Shear Strain
------------

:class:`ShearStrainTransformation` applies shear strain to a crystal, which
preserves volume (determinant = 1) but changes the cell shape.

**Use cases:**

- Studying elastic properties
- Calculating shear moduli
- Exploring slip systems

**Example:**

.. code-block:: python

   from aenet.geometry.transformations import ShearStrainTransformation

   # Apply shear on xy plane from -0.1 to +0.1
   transform = ShearStrainTransformation(
       direction=1,  # 1=xy, 2=xz, 3=yz
       shear_min=-0.1,
       shear_max=0.1,
       steps=5
   )

   for s in transform.apply_transformation(structure):
       process(s)

**Physics:** The shear matrix for xy shear is:

.. math::

   \mathbf{M} = \begin{pmatrix}
   1 & \gamma & 0 \\
   0 & 1 & 0 \\
   0 & 0 & 1
   \end{pmatrix}

where :math:`\gamma` is the shear strain parameter. The determinant is always 1,
ensuring volume conservation.

Practical Tips
--------------

**Choosing displacement magnitudes:**

- For forces: 0.01-0.05 Å (harmonic regime)
- For structure search: 0.1-0.3 Å (exploration)
- For large perturbations: > 0.5 Å (may need relaxation)

**Choosing strain ranges:**

- Elastic regime: ±2% strain
- Beyond elasticity: ±5-10% strain
- Phase changes: > ±10% strain

**Limiting output:**

For large systems, use ``itertools.islice()`` to limit structures:

.. code-block:: python

   import itertools

   # Get only first 100 structures
   limited = list(itertools.islice(
       transform.apply_transformation(structure), 100
   ))

See :doc:`transformations_advanced` for transformation chains and stochastic
transformations.

Common Patterns
---------------

**Save all structures:**

.. code-block:: python

   for i, s in enumerate(transform.apply_transformation(structure)):
       s.write_to_file(f'output_{i:04d}.xsf')

**Filter structures:**

.. code-block:: python

   # Only keep structures with energy below threshold
   good_structures = [
       s for s in transform.apply_transformation(structure)
       if s.energy[-1] is not None and s.energy[-1] < threshold
   ]

**Combine with list comprehension:**

.. code-block:: python

   volumes = [
       s.cellvolume()
       for s in transform.apply_transformation(structure)
   ]

Next Steps
----------

- For transformation chains and stochastic transformations,
  see :doc:`transformations_advanced`
- For complete API documentation, see :doc:`../api/transformations`
