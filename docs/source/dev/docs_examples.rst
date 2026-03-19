Documentation Example Testing
=============================

Overview
--------

The repository uses a hybrid approach for runnable documentation examples:

- short, deterministic snippets stay in ``.rst`` pages as ``.. doctest::``
  blocks and run through Sphinx doctest
- longer workflows stay as ``.. code-block:: python`` examples and are backed
  by pytest smoke tests
- narrative or placeholder-heavy examples remain non-executable and should be
  covered indirectly when practical

This keeps the rendered documentation readable while still catching example
drift in the supported runnable subset.

Running the Checks
------------------

Use the ``aenet-torch`` environment for docs example testing.

Run Sphinx doctest:

.. code-block:: bash

   /Users/aurban/.local/bin/micromamba run -n aenet-torch \
       python -m sphinx -b doctest docs/source docs/build/doctest

This executes ``.. doctest::`` blocks in the rendered documentation and reports
failures by document name.

Run pytest-backed docs example tests:

.. code-block:: bash

   /Users/aurban/.local/bin/micromamba run -n aenet-torch \
       pytest -m docs_examples

Use a file path for a narrower loop while editing a single page:

.. code-block:: bash

   /Users/aurban/.local/bin/micromamba run -n aenet-torch \
       pytest -q src/aenet/geometry/tests/test_docs_transformations_basic.py

Authoring Policy
----------------

Use ``.. doctest::`` when all of the following are true:

- the example is short
- imports and setup are explicit in the snippet
- output is deterministic and stable across supported platforms

Keep a normal ``.. code-block:: python`` and add pytest coverage when the
example needs:

- temporary files or directories
- several setup steps
- helper fixtures
- assertions that would make doctest markup noisy

Keep an example narrative-only when it intentionally uses placeholders such as
``process(s)`` or project-specific external data paths.

Prefer notebooks when the example is tutorial-shaped, file-heavy, or duplicates
an existing maintained notebook workflow.

Current Coverage
----------------

The current docs-example rollout covers:

- ``docs/source/usage/transformations_basic.rst``
- ``docs/source/dev/neighbor_lists.rst``
- ``docs/source/usage/torch_featurization.rst``
- ``docs/source/usage/torch_datasets.rst``
- ``docs/source/usage/torch_training.rst``
- ``docs/source/usage/torch_inference.rst``

Reference page-level pytest coverage lives in:

- ``src/aenet/geometry/tests/test_docs_transformations_basic.py``
- ``src/aenet/torch_nblist/tests/test_docs_neighbor_lists.py``
- ``src/aenet/torch_featurize/tests/test_docs_torch_featurization.py``
- ``src/aenet/torch_training/tests/test_docs_torch_datasets.py``
- ``src/aenet/torch_training/tests/test_docs_torch_training.py``
- ``src/aenet/torch_training/tests/test_docs_torch_inference.py``

Use these pages and test modules as the reference patterns for future
docs-example work.
