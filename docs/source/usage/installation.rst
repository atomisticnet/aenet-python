Installation & Set-up
=====================

Prerequisites
--------------
* ænet library >= 2.0.3 (separate installation required)
* ænet binaries `generate.x` and `trnset2ASCII.x`
* python >= 3.6
* numpy >= 1.20.1
* scipy >= 1.6.2
* pandas >=1.2.4
* tables >=3.6.1

Installation
------------

.. note::

   For the installation of the ænet binaries and library see
   the `ænet website <http://ann.atomistic.net>`_ and the
   `GitHub repository <https://github.com/atomisticnet/aenet>`_.


1. Install ``aenet-python``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download the source code repository from GitHub `aenet-python
<https://github.com/atomisticnet/aenet-python>`_.

Install as usual.  For example with

.. sourcecode:: console

   $ pip install . --user

from the repository's root directory.


2. Configure ænet
^^^^^^^^^^^^^^^^^

To make ``aenet-python`` aware of the ænet binaries, the paths need to
be configured.  The following command runs an interactive dialog that
works for standard installations

.. sourcecode:: console

   $ aenet config --set-aenet-path [path-to-aenet]

where ``[path-to-aenet]`` is the path pointing to the aenet root
directory.
