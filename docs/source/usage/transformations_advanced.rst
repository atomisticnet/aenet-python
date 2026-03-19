Advanced Structure Transformations
===================================

This guide covers advanced features including transformation chains,
stochastic sampling, reproducibility, and iterator composition patterns.

See :doc:`transformations_basic` for an introduction to individual transformations.

Transformation Chains
---------------------

:class:`TransformationChain` allows sequential application of multiple
transformations using depth-first streaming. This is useful for generating
complex structural variations by composing simple transformations.

Basic Chain Example
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from aenet.geometry import AtomicStructure
   from aenet.geometry.transformations import (
       AtomDisplacementTransformation,
       CellVolumeTransformation,
       TransformationChain
   )

   structure = AtomicStructure.from_file('structure.xsf')

   # Create a chain of transformations
   chain = TransformationChain([
       AtomDisplacementTransformation(displacement=0.05),
       CellVolumeTransformation(min_percent=-2, max_percent=2, steps=3)
   ])

   # Process structures lazily
   for s in chain.apply_transformation(structure):
       process(s)

   # Or get all structures at once
   all_structures = list(chain.apply_transformation(structure))
   print(f"Generated {len(all_structures)} structures")
   # Output: Generated 72 structures (24 displaced × 3 volumes)

**How chains work:**

The chain applies transformations sequentially using depth-first streaming:

1. First transformation generates structures from input
2. Each generated structure is immediately passed to the second transformation
3. Results flow through the entire chain before the next structure is processed

This depth-first approach ensures correct multiplicative behavior and
memory efficiency.

Controlling Output Size
-----------------------

Since chains can produce many structures, use standard Python tools to limit output:

**Using itertools.islice():**

.. code-block:: python

   import itertools

   chain = TransformationChain([
       AtomDisplacementTransformation(displacement=0.05),
       CellVolumeTransformation(min_percent=-5, max_percent=5, steps=10)
   ])

   # Get only first 100 structures
   first_100 = list(itertools.islice(
       chain.apply_transformation(structure), 100
   ))

**Using a counter:**

.. code-block:: python

   results = []
   for i, s in enumerate(chain.apply_transformation(structure)):
       if i >= 1000:
           break
       results.append(s)

**Filtering with conditions:**

.. code-block:: python

   # Only keep structures meeting criteria
   good_structures = [
       s for s in itertools.islice(
           chain.apply_transformation(structure), 10000
       )
       if meets_criteria(s)
   ][:100]  # Keep first 100 that meet criteria

Iterator Composition
--------------------

Chains are iterators, so they compose naturally with Python's ``itertools``:

**Interleaving multiple chains:**

.. code-block:: python

   import itertools

   chain1 = TransformationChain([transform1, transform2])
   chain2 = TransformationChain([transform3, transform4])

   # Alternate between chains
   combined = itertools.chain(
       chain1.apply_transformation(structure),
       chain2.apply_transformation(structure)
   )

**Processing in batches:**

.. code-block:: python

   def batch_iter(iterable, batch_size):
       """Yield batches of structures."""
       iterator = iter(iterable)
       while True:
           batch = list(itertools.islice(iterator, batch_size))
           if not batch:
               break
           yield batch

   # Process 10 structures at a time
   for batch in batch_iter(chain.apply_transformation(structure), 10):
       process_batch(batch)

**Parallel processing:**

.. code-block:: python

   from multiprocessing import Pool

   def process_structure(s):
       # Expensive computation
       return result

   # Generate structures lazily, process in parallel
   with Pool(4) as pool:
       results = pool.map(
           process_structure,
           itertools.islice(chain.apply_transformation(structure), 1000)
       )

Stochastic Transformations
---------------------------

:class:`RandomDisplacementTransformation` generates random atomic displacement
vectors. The displacements can be optionally orthonormalized to create an
independent basis of perturbations. Unlike deterministic transformations, it
uses randomness and requires careful attention to reproducibility.

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from aenet.geometry.transformations import RandomDisplacementTransformation

   # Generate random structures with 0.1 Å RMS displacement
   # By default, generates 3N-3 orthonormal structures
   transform = RandomDisplacementTransformation(
       rms=0.1,
       random_state=42,  # Seed for reproducibility
       orthonormalize=True,  # Generate orthonormal basis (default)
       remove_translations=True  # Remove translational modes (default)
   )

   # Get all random structures
   random_structures = list(transform.apply_transformation(structure))
   print(f"Generated {len(random_structures)} random structures")

   # Generate non-orthonormalized random samples
   transform_random = RandomDisplacementTransformation(
       rms=0.1,
       max_structures=50,  # Specify number of samples
       orthonormalize=False,  # Just random perturbations
       random_state=42
   )

**Parameters:**

- ``rms``: Target root-mean-square displacement in Angstroms
- ``max_structures``: Maximum number of structures to generate. If None, defaults to 3N-3 (default: None)
- ``random_state``: Integer seed or numpy Generator for reproducibility
- ``orthonormalize``: If True, generate orthonormal basis via QR decomposition (default: True)
- ``remove_translations``: If True, remove 3 uniform translation modes (default: True)

Algorithm Details
^^^^^^^^^^^^^^^^^

**When orthonormalize=True** (default):

The transformation generates orthonormal displacement vectors in 3N-dimensional
space (N atoms, 3 coordinates per atom):

1. Generate random 3N×M matrix (M = ``max_structures``)
2. Apply QR decomposition to obtain orthonormal columns
3. Optionally project out 3 translational modes
4. Re-orthonormalize remaining vectors via second QR decomposition
5. Normalize each vector to target RMS

**When orthonormalize=False**:

Random displacement samples are generated independently:

1. Generate random 3N-dimensional vector from normal distribution
2. Optionally remove center-of-mass translation
3. Normalize to target RMS
4. Repeat for each structure (no orthogonality constraint)

**RMS displacement** is defined as:

.. math::

   \text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \|\mathbf{d}_i\|^2}

where :math:`\mathbf{d}_i` is the displacement vector for atom :math:`i`.

**Orthogonality** ensures displacement vectors are mutually independent:

.. math::

   \mathbf{v}_i \cdot \mathbf{v}_j = \delta_{ij}

where :math:`\delta_{ij}` is the Kronecker delta (1 if i=j, 0 otherwise).

Translation Mode Removal
^^^^^^^^^^^^^^^^^^^^^^^^^

When ``remove_translations=True``, the three uniform translational modes
are removed:

.. math::

   \mathbf{t}_x = (\underbrace{1, 0, 0, 1, 0, 0, \ldots}_{\text{N atoms}}), \quad
   \mathbf{t}_y = (0, 1, 0, 0, 1, 0, \ldots), \quad
   \mathbf{t}_z = (0, 0, 1, 0, 0, 1, \ldots)

After projection, 3N-3 orthonormal vectors remain. For a single atom
(3 degrees of freedom), this results in zero vectors, so the transformation
yields no structures.

Re-orthonormalization after projection is **critical** to maintain mutual
orthogonality of the remaining vectors.

Reproducibility
---------------

For scientific reproducibility, always control randomness in stochastic
transformations.

Using Integer Seeds
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Same seed → identical results
   transform1 = RandomDisplacementTransformation(rms=0.1, random_state=42)
   transform2 = RandomDisplacementTransformation(rms=0.1, random_state=42)

   results1 = list(transform1.apply_transformation(structure))
   results2 = list(transform2.apply_transformation(structure))

   # results1 and results2 are identical

Using numpy Generator
^^^^^^^^^^^^^^^^^^^^^

For more control, use a numpy Generator:

.. code-block:: python

   import numpy as np

   # Create explicit generator
   rng = np.random.default_rng(seed=12345)

   transform = RandomDisplacementTransformation(
       rms=0.1,
       max_structures=50,
       random_state=rng
   )

   # Generator state is advanced after use
   results = list(transform.apply_transformation(structure))

   # Reusing same generator produces different results (state advanced)
   more_results = list(transform.apply_transformation(structure))

Reproducibility Checklist
^^^^^^^^^^^^^^^^^^^^^^^^^^

For fully reproducible workflows:

1. **Pin package versions** in requirements.txt or environment.yml
2. **Always set random_state** for stochastic transformations
3. **Document seeds** in scripts and logs
4. **Version control** transformation parameters
5. **Record** numpy version (RNG implementation may vary)
6. **Log** all generated structures with metadata

Example:

.. code-block:: python

   import json
   import numpy as np
   from aenet.geometry.transformations import RandomDisplacementTransformation

   # Configuration
   config = {
       'rms': 0.1,
       'max_structures': 100,
       'seed': 42,
       'orthonormalize': True,
       'remove_translations': True,
       'numpy_version': np.__version__
   }

   # Save configuration
   with open('transform_config.json', 'w') as f:
       json.dump(config, f, indent=2)

   # Apply transformation
   transform = RandomDisplacementTransformation(
       rms=config['rms'],
       max_structures=config['max_structures'],
       random_state=config['seed'],
       orthonormalize=config['orthonormalize'],
       remove_translations=config['remove_translations']
   )

   structures = list(transform.apply_transformation(structure))

Complete Workflow Example
--------------------------

Generate training data for a machine learning potential:

.. code-block:: python

   from aenet.geometry import AtomicStructure
   from aenet.geometry.transformations import (
       RandomDisplacementTransformation,
       CellVolumeTransformation,
       IsovolumetricStrainTransformation,
       TransformationChain
   )
   import itertools

   # Load reference structure
   structure = AtomicStructure.from_file('reference.xsf')

   # Define workflow: random displacements + volume + strain
   chain = TransformationChain([
       RandomDisplacementTransformation(
           rms=0.08,
           max_structures=20,
           random_state=42
       ),
       CellVolumeTransformation(
           min_percent=-3,
           max_percent=3,
           steps=3
       ),
       IsovolumetricStrainTransformation(
           direction=1,
           len_min=0.95,
           len_max=1.05,
           steps=3
       )
   ])

   # Generate up to 500 structures
   print("Generating training structures...")
   training_structures = list(itertools.islice(
       chain.apply_transformation(structure), 500
   ))
   print(f"Generated {len(training_structures)} structures")

   # Save for training
   for i, s in enumerate(training_structures):
       s.to_file(f'training_data/struct_{i:04d}.xsf')

   print("Training data generation complete")

This workflow:

1. Generates 20 random displacements (orthonormal, RMS=0.08 Å)
2. Applies 3 volume scalings to each
3. Applies 3 isovolumetric strains to each
4. Limits total output to 500 structures
5. Saves all structures for training

**Expected output:** 20 × 3 × 3 = 180 structures (or 500 if limited)

Performance Considerations
--------------------------

**Computational Complexity:**

- AtomDisplacementTransformation: O(N) per structure, yields 3N structures
- CellVolumeTransformation: O(1) per structure, yields M structures
- IsovolumetricStrainTransformation: O(1) per structure, yields M structures
- ShearStrainTransformation: O(1) per structure, yields M structures
- RandomDisplacementTransformation: O(N²M) for QR decomposition (when orthonormalize=True)

where N = number of atoms, M = number of steps/structures.

**Memory Usage:**

- Iterator-based design: Only one structure in memory at a time
- Use ``list()`` only when you need all structures at once
- Process structures as they're generated for maximum efficiency

**Recommendations:**

- For small systems (< 100 atoms): No special considerations needed
- For medium systems (100-1000 atoms): Use itertools.islice() to limit output
- For large systems (> 1000 atoms): Process structures one at a time
- QR decomposition for N=100 atoms, M=100 structures: < 1 second

Memory-Efficient Patterns
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Process and discard:**

.. code-block:: python

   # Don't store all structures
   for s in chain.apply_transformation(structure):
       result = expensive_calculation(s)
       save_result(result)
       # Structure is discarded after processing

**Accumulate results, not structures:**

.. code-block:: python

   # Store computed properties, not structures
   energies = []
   for s in chain.apply_transformation(structure):
       energy = compute_energy(s)
       energies.append(energy)

**Stream to disk:**

.. code-block:: python

   # Write structures as they're generated
   for i, s in enumerate(chain.apply_transformation(structure)):
       if i >= 1000:
           break
       s.to_file(f'output_{i:04d}.xsf')

Combining Deterministic and Stochastic
---------------------------------------

Mix deterministic and stochastic transformations for comprehensive sampling:

.. code-block:: python

   import itertools

   # Deterministic baseline
   deterministic_chain = TransformationChain([
       CellVolumeTransformation(min_percent=-5, max_percent=5, steps=5),
       IsovolumetricStrainTransformation(direction=1, len_min=0.9, len_max=1.1, steps=5)
   ])

   # Stochastic perturbations
   stochastic = RandomDisplacementTransformation(rms=0.05, max_structures=10, random_state=42)

   # Combine: deterministic structures + random displacements for each
   all_structures = []
   for base_structure in deterministic_chain.apply_transformation(structure):
       # Original deterministic structure
       all_structures.append(base_structure)
       # Plus random variations
       for displaced in stochastic.apply_transformation(base_structure):
           all_structures.append(displaced)

   print(f"Total: {len(all_structures)} structures")
   # 25 deterministic + 25×10 random = 275 structures

Troubleshooting
---------------

**Problem:** Too many structures generated

**Solution:** Use ``itertools.islice()`` to limit output

**Problem:** Out of memory

**Solution:** Process structures one at a time, don't use ``list()``

**Problem:** Non-reproducible results

**Solution:** Always set ``random_state`` for stochastic transformations

**Problem:** Structures too similar

**Solution:** Increase displacement/strain magnitudes or reduce ``steps``

**Problem:** QR decomposition slow

**Solution:** Reduce ``max_structures`` for RandomDisplacementTransformation, or use ``orthonormalize=False`` for faster random sampling

**Problem:** Not enough structures for single atom

**Solution:** With ``remove_translations=True``, single atoms have 3N-3=0 vectors;
use ``remove_translations=False`` to get 3 displacement vectors

Further Reading
---------------

- :doc:`transformations_basic` - Introduction to individual transformations
- :doc:`../api/transformations` - Complete API reference
- :doc:`structure_manipulation` - Command-line structure tools
