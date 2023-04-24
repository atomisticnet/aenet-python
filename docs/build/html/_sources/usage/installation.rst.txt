Installation & Set-up
=====================

``aenet`` package should be installed first. Follow the instructurion 
on `GitHub aenet <https://github.com/atomisticnet/aenet>`_ or 
`aenet documentation <http://ann.atomistic.net/documentation/>`_. 
A Fortran compiler (e.g., `gfortran <https://fortran-lang.org/en/learn/os_setup/install_gfortran/>`_
or `ifort <https://www.intel.com/content/www/us/en/developer/tools/oneapi/fortran-compiler.html#gs.vxs2id>`_
) and the ``Math Kernel Library (MKL)``
are necessary during the compiling of ``aenet`` package. The 
`OpenMPI <https://www.open-mpi.org/>`_ package is optional, which
offers the parallel computing.

Prerequisites
----------------
* aenet >= 2.0.3
* python >=3.6
* cython
* numpy

Installation of MKL
------------------------

On the homepage of `Math Kernel Library (MKL) <https://
www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.vxrgux>`_,
select the Stand-Alone Version or Toolkit. After the installation, 
system environment variables should be set by running 
    
.. sourcecode:: console

    $ source /opt/intel/oneapi/setvars.sh intel64   

Test the installation: ``echo $MKLROOT``, which should return the 
installation path of ``mkl``, e.g. ``/opt/intel/oneapi/mkl/2023.1.0``.

Installation of ``aenet-python``
--------------------------------

Install the ``aenet-python`` package from the source
directory:

.. sourcecode:: console

   $ pip install . --user

or

.. sourcecode:: console

   $ python setup.py install --user

