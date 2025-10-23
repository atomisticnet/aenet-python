Structure featurization
=======================

.. note::

   Structure featurization makes use of ænet's compiled ``generate.x``
   ``trnset2ASCII.x`` tools.  Make sure to install ænet and configure
   the paths as described in :doc:`installation`.

   **Alternative**: For a pure Python/PyTorch implementation that does not
   require Fortran, see :doc:`torch_featurization`. The PyTorch implementation
   provides identical results with GPU acceleration support.

``aenet-python`` can be used to featurize atomic environments with the
expansion method by Artrith *et al.* [1,2,3].  Local atomic environment
features can, furthermore, be combined to atomic structure features with
the approach by Gharakhanyan *et al.* [4].

[1] N. Artrith and A. Urban, *Comput. Mater. Sci.* **114**, 2016, 135-150
(`link4 <http://dx.doi.org/10.1016/j.commatsci.2015.11.047>`_).

[1] N. Artrith, A. Urban, and G. Ceder,
*Phys. Rev. B* **96**, 2017, 014112 (`link1 <https://doi.org/10.1103/PhysRevB.96.014112>`_).

[2] A. M. Miksch, T. Morawietz, J. Kästner, A. Urban, N. Artrith,
*Mach. Learn.: Sci. Technol.* **2**, 2021, 031001 (`link2 <http://doi.org/10.1088/2632-2153/abfd96>`_).

[3] V. Gharakhanyan, M. S. Aalto, A. Alsoulah, N. Artrith, A. Urban,
ICLR 2023 (`link3 <https://openreview.net/forum?id=4Hl8bjobpl9>`_)

Example notebooks
-----------------

Jupyter notebooks with examples how to use the featurization methods can
be found in the `notebooks
<https://github.com/atomisticnet/aenet-python/tree/master/notebooks>`_
directory within the repository.


Featurizing local atomic environments
-------------------------------------

The class :class:`aenet.featurize.AenetAUCFeaturizer` can be used to
transparently run ænet's ``generate.x`` and create *training set*
files:

.. sourcecode:: python

   import glob
   from aenet.featurize import AenetAUCFeaturizer
   fzer = AenetAUCFeaturizer(['H', 'C', 'O'],
                             rad_cutoff=4.0, rad_order=10,
                             ang_cutoff=1.5, ang_order=3)
   fzer.run_aenet_generate(glob.glob("./xsf/*.xsf"), workdir='run')

The above will prepare input files for and run ``generate.x``, which can
take a while depending on the number of XSF files and their size.  In
addition to the usual data file in Fortran binary format that is
produced by ``generate.x``, the data is also written to a file in `HDF5
<https://www.hdfgroup.org/solutions/hdf5/>`_ format, named
``features.h5``, for better interoperability with Python.

The contents of the HDF5 data file can be interacted with using the
class :class:`aenet.trainset.TrnSet`

.. sourcecode:: python

   from aenet.trainset import TrnSet
   with TrnSet.from_file('features.h5') as ts:
      # do something

See the Jupyter `notebooks
<https://github.com/atomisticnet/aenet-python/tree/master/notebooks>`_
directory for complete examples.
