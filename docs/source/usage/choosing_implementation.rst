Choosing an Implementation
==========================

``aenet-python`` provides two implementations for training and using
machine learning interatomic potentials:

1. **Fortran-based**: Wraps the compiled ænet Fortran executables
2. **PyTorch-based**: Pure Python implementation using PyTorch

Both implementations are maintained and supported, and both have advantages
and disadvantages.  The Fortran-based implementation requires that the
ænet Fortran binaries are compiled and installed, and the PyTorch-based
implementation requires that PyTorch is installed.  Both are independent
and can be installed/used separately.

Featurization
-------------

Both implementations implement the Chebyshev expansion-based featurization
method by Artrith et al. [1] (see reference [2] for a tutorial) and give
numerically equivalent results.  The Fortran code is generally more
efficient on CPUs, while the PyTorch implementation provides GPU acceleration
and ties in with PyTorch's automatic differentiation capabilities.

Training
--------

The Fortran implementation provides basic training capabilities with good
performance on CPUs but currently only supports training on energies.  The
PyTorch implementation supports training on energies and forces, provides
more advanced training algorithms, and benefits from GPU acceleration.
Models trained with either implementation can be used interchangeably for
inference.

Inference
---------

The Fortran implementation is generally more efficient for inference on
CPUs by a significant margin (20–50× faster depending on the system and
model).  However, it is parallelized only over atoms, not for the neural
network evaluation.  The PyTorch implementation can leverage GPUs for neural
network evaluation and can be more efficient for large models.  Another
advantage of the Fortran implementation is the C-compatible library that
it provides for use with third-party software, such as the LAMMPS and
Tinker molecular dynamics packages [3].

References
----------

[1] Chebyshev featurization method: N. Artrith, A. Urban, and G. Ceder,
*Phys. Rev. B* **96**, 2017, 014112
(`link1 <https://doi.org/10.1103/PhysRevB.96.014112>`_).

[2] Tutorial: A. M. Miksch, T. Morawietz, J. Kästner, A. Urban, N. Artrith,
*Mach. Learn.: Sci. Technol.* **2**, 2021, 031001
(`link2 <http://doi.org/10.1088/2632-2153/abfd96>`_).

[3] LAMMPS and Tinker integration: M. S. Chen, T. Morawietz, H. Mori,
T. E. Markland, N. Artrith, *J. Chem. Phys.* **155**, 2021, 074801
(`link3 <https://doi.org/10.1063/5.0063880>`_).
