# Python interface for ænet

## Prerequisites
- Fortran compiler [`gfortran`](https://fortran-lang.org/en/learn/os_setup/install_gfortran/)or [`ifort`](https://www.intel.com/content/www/us/en/developer/tools/oneapi/fortran-compiler.html#gs.vxs2id)
- The [Math Kernel Library (MKL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.vxrgux)
- [aenet](https://github.com/atomisticnet/aenet)>=2.0.3
- python>=3.6
- cython
- numpy
- openmpi (optional)

## Installation of MKL library

On the page of [Math Kernel Library (MKL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.vxrgux), select the Stand-Alone Version or Toolkit. After the installation, run 
    
    $ source /opt/intel/oneapi/setvars.sh intel64   
    
or add this line to `~/.bashrc` and `$ source ~/.bashrc`.
Test the installation: 
    
    $ echo $MKLROOT
    
It should return the installation path of `mkl`, e.g. `/opt/intel/oneapi/mkl/2023.1.0`.

## Installation
(1) Follow the [instruction](https://github.com/atomisticnet/aenet) to download and compile `aenet` package first.

(2) Compile the symmetry function C library. 

    $ cd aenet/src
    $ make -f ./makefiles/Makefile.XXX lib

Select the Makefile.XXX that is appropriate for your system, but *make sure to compile a serial version of `aenet`(??)*. If a `mpi` version is selected, the package OpenMPI should be installed first:

    $ conda install -c conda-forge openmpi

(3) Build the Python interface. 

    $ cd aenet-python

    $ pip install . --user
    or
    $ python setup.py install --user

~~Build Python extension module~~
~~python setup.py build_ext --inplace~~ (I didn't find the extension module in setup.py)



