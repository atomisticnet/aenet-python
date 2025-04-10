.. aenet-python documentation master file, created by
   sphinx-quickstart on Tue Feb  8 15:39:20 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ænet-python Documentation
===========================

The ``aenet-python`` package is a collection of utilities for preparing
input files and processing output files for the machine-learning
interatomic potential software `ænet <http://ann.atomistic.net>`_.

Common use cases of the package are

* Extraction of structures, energies, and forces from the output of
  first-principles calculations;
* Interconversion of atomic structure formats, especially conversion to
  ænet's XSF format;
* Manipulation of atomic structures, e.g., for generating reference data;
* Operations on the featurized reference data produced by ænet's
  ``generate.x``; and
* Analysis of the outputs generated by ænet's ``train.x``, i.e., from
  machine-learning potential training.

Much of the package's functionality is exposed through command-line
tools.  Specifically, the tool ``sconv`` (*structure conversion*) makes
available capabilities for atomic structure modification and
interconversion and ``sfp`` (*structure fingerprints*) can be used to
featurize atomic structures.  In addition, the ``config`` tool can be
used for :doc:`configuration </usage/installation>`.

See :doc:`/usage/commandline` for an overview of the
command-line capabilities.

Contents
--------------

.. toctree::
   :maxdepth: 2
   :glob:

   usage/installation
   usage/structure_manipulation
   usage/data_acquisition
   usage/featurization
   usage/commandline
   usage/trainset
   dev/commandline


Indices and tables
--------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
