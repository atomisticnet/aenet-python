Structure conversion and manipulation
-------------------------------------

The following examples showcase some of ``sconv``'s structure
manipulation capabilities.

* Interconversion between the various supported file formats is
  straightforward with :doc:`tools/sconv`.  If the format of the input
  and/or output file is not uniquely defined by the file name, the
  formats must be specified using the flags ``-i`` and ``-o``.  Here a
  few examples:

.. sourcecode:: console

   $ # convert VASP POSCAR to XSF format
   $ aenet sconv POSCAR structure.xsf
   $ # conversion to LAMMPS dump
   $ aenet sconv structure.xsf structure.dump
   $ # protein database format
   $ aenet sconv structure.dump structure.pdb

* :doc:`tools/sconv` can also manipulate atomic structures.  If no
  output file name is specified, the output will be printed to the
  terminal (i.e., to *standard out*).  Here just a few examples:

.. sourcecode:: console

   $ # shift unit cell such that atom 10 is on the origin
   $ aenet sconv structure.xsf -o vasp --shift origin 10
   $ # scale the atomic structure uniformly by a factor of 1.1
   $ aenet sconv structure.xsf -o vasp --scale 1.1
   $ # rotate the structure by 45 degrees around the z axis
   $ aenet sconv structure.xyz -o vasp --rotate-angle 3 45
   $ # return a spherical region with radius 10 Angstrom around atom 5
   $ aenet sconv structure.xyz -o xyz --cut 5 10.0

* Multiple instructions can be piped (if supported by the shell).  In
  that case, the input file is replaced by a hyphen (``-``), and the
  input format needs to be defined:

.. sourcecode:: bash

   $ aenet sconv structure.xsf -o vasp --shift origin 10 \
     | aenet sconv - -i vasp -o vasp --scale 1.1 \
     | aenet sconv - -i vasp -o vasp --rotate-angle 3 45
