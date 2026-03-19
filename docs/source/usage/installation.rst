Installation & Set-up
=====================

Installation
------------

.. note::

   For the installation of the ænet binaries and library see
   the `ænet website <http://ann.atomistic.net>`_ and the
   `GitHub repository <https://github.com/atomisticnet/aenet>`_.

.. note::

   PyTorch is only installed when requested explicitly, since it is a
   large dependency that can sometimes be difficult to install.


1. Package Install
^^^^^^^^^^^^^^^^^^

Download the source code repository from GitHub `aenet-python
<https://github.com/atomisticnet/aenet-python>`_.

Install as usual.  For example with

.. sourcecode:: console

   $ pip install .

from the repository's root directory.

Per default, PyTorch and other requirements of the PyTorch-based features
are not installed.  To request those requirements as well, select them
explicitly with

.. sourcecode:: console

   $ pip install ".[torch]"

This installs the core PyTorch dependency only.  The PyTorch-based
featurization and neighbor-list features also require the PyG extension
packages ``torch-scatter`` and ``torch-cluster`` matched to the local torch
build.

Recommended installation matrix:

.. sourcecode:: console

   # Core PyTorch support only
   $ pip install ".[torch]"

   # Full CPU PyTorch stack
   $ pip install ".[torch]"
   $ pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+cpu.html

   # Full CUDA PyTorch stack
   $ pip install ".[torch]"
   $ pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

Replace ``${TORCH}`` with the installed torch version (for example ``2.9.0``)
and ``${CUDA}`` with the matching CUDA tag (for example ``cu124``).  If you
are developing from a source checkout, the equivalent editable install is:

.. sourcecode:: console

   $ pip install -e ".[dev]"
   $ pip install ".[torch]"
   $ pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+cpu.html


2. Configure ænet Fortran Binaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To make ``aenet-python`` aware of the ænet binaries, the paths need to
be configured.  The following command runs an interactive dialog that
works for standard installations

.. sourcecode:: console

   $ aenet config --set-aenet-path [path-to-aenet]

where ``[path-to-aenet]`` is the path pointing to the aenet root
directory.
