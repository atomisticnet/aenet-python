Installation & Set-up
=====================

Prerequisites
----------------
* A Fortran compiler
* Required math libraries
* aenet >= 2.0.3
* python >=3.6
* cython
* numpy >= 1.20.1
* scipy >= 1.6.2
* pandas >=1.2.4
* tables >=3.6.1

Installation
------------------------

1. Install a Fortran compiler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``aenet`` is mostly written by Frotran programming language for
the high efficiency of numeric computation. A Fortran compiler 
is required for the installation (i.e. compilation). Two compilers 
are recommended: `gfortran <https://fortran-lang.org/en/learn/os_setup/install_gfortran/>`_
and `ifort <https://www.intel.com/content/www/us/en/developer/tools/oneapi/fortran-compiler.html#gs.vxs2id>`_.

2.. Install the required library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Three choices are available as listed below. During the compilation
of ``aenet``, select the corresponding ``Makefile.xxx`` based on the
installed libraries.

``Math Kernel Library (MKL)`` (recommended)
""""""""""""""""""""""""""""""""""""""""""""""

On the homepage of `Math Kernel Library <https://
www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.vxrgux>`_,
select the Stand-Alone Version or Toolkit. After the installation, 
system environment variables should be set by running 

.. sourcecode:: console

    $ source /opt/intel/oneapi/setvars.sh intel64   

Test the installation: ``echo $MKLROOT``, which should return the 
installation path of ``mkl``, e.g. ``/opt/intel/oneapi/mkl/2023.1.0``.

``OpenBLAS`` library
"""""""""""""""""""""""

Follow the instruction for `OpenBLAS <https://www.openblas.net/>`_.

``BLAS`` and ``LAPACK`` libraries
"""""""""""""""""""""""""""""""""""

Follow the instruction for `BLAS <https://netlib.org/blas/>`_ and 
`LAPACK <https://netlib.org/lapack/>`_.

3. Install OpenMPI (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`OpenMPI <https://www.open-mpi.org/>`_ package is optional, which
offers the parallel computing. One way of installation is:

.. sourcecode:: console

    $ conda install -c conda-forge openmpi

Other methods can be found on `Quick start: Installing Open MPI 
<https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/quickstart.html>`_.

If this package is installed, ``Makefile.xxx`` with ``mpi`` tag can be
selected, otherwise please choose among the ``serial`` versions.

4. Compile ``aenet`` package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Follow the instructurion 
on `GitHub aenet <https://github.com/atomisticnet/aenet>`_ or 
`aenet documentation <http://ann.atomistic.net/documentation/>`_. 
During the compilation, select the correct ``Makefile.xxx`` based on 
the previous steps.


5. Install ``aenet-python``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Source code is available on GitHub `aenet-python <https://github.com/atomisticnet/aenet-python>`_.
Within the root directory of the source code, run

.. sourcecode:: console

   $ pip install . --user

or

.. sourcecode:: console

   $ python setup.py install --user

