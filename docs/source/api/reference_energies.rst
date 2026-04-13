.. _api_reference_energies:

Reference Energies
==================

The ``aenet.reference_energies`` module provides helper routines for
constructing atomic reference energies used in cohesive- and
formation-energy-based workflows.

Both `ReferenceEnergies.from_regression()` and
`ReferenceEnergies.from_reference_compounds()` consume lightweight
``(composition, energy)`` samples directly, so callers can stream data from
custom parsers without materializing full structure objects. For file-backed
workflows, the module also provides a lazy helper that yields those samples
from paths readable by ``aenet.io.structure``.

.. currentmodule:: aenet.reference_energies

.. autoclass:: ReferenceEnergies
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: iter_composition_energy_samples_from_files
