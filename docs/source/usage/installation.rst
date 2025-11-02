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


2. Configure ænet Fortran Binaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To make ``aenet-python`` aware of the ænet binaries, the paths need to
be configured.  The following command runs an interactive dialog that
works for standard installations

.. sourcecode:: console

   $ aenet config --set-aenet-path [path-to-aenet]

where ``[path-to-aenet]`` is the path pointing to the aenet root
directory.
