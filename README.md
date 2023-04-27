# Python interface for ænet

## Prerequisites
- Fortran compiler [`gfortran`](https://fortran-lang.org/en/learn/os_setup/install_gfortran/)or [`ifort`](https://www.intel.com/content/www/us/en/developer/tools/oneapi/fortran-compiler.html#gs.vxs2id)
- Required math libraries
- [aenet](https://github.com/atomisticnet/aenet)>=2.0.3
- python>=3.6
- cython
- numpy >= 1.20.1
- scipy >= 1.6.2
- pandas >=1.2.4
- tables >=3.6.1
- openmpi (optional)

## Simplified installation guide
A detailed guide can be found **xxxxx**

### Math Library

The recommended math library is  [Math Kernel Library (MKL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.vxrgux). Select the Stand-Alone Version or Toolkit. After the installation, run 
    
    $ source /opt/intel/oneapi/setvars.sh intel64   
    
Test the installation: 
    
    $ echo $MKLROOT
    
It should return the installation path of `mkl`, e.g. `/opt/intel/oneapi/mkl/2023.1.0`.

Other choices could be: [OpenBLAS library](https://www.openblas.net/); or [BLAS](https://netlib.org/blas/) together with [LAPACK](https://netlib.org/lapack/) libraries.

### Installation
(1) Follow the [instruction](https://github.com/atomisticnet/aenet) to download and compile `aenet` package first.
Select the appropriate `Makefile.XXX` based on the installed compiler, math libraries, etc. If a `mpi` version instead of `serial` version is selected, the package `OpenMPI` should be installed first:

    $ conda install -c conda-forge openmpi

(2) Build the Python interface. 

    $ cd aenet-python

    $ pip install . --user
    or
    $ python setup.py install --user




