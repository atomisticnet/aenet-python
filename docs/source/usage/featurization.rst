Structure featurization
=======================

.. note::

   Structure featurization makes use of ænet's compiled ``generate.x``
   ``trnset2ASCII.x`` tools.  Make sure to install ænet and configure
   the paths as described in :doc:`installation`.

.. note::

   **Alternative**: For a pure Python/PyTorch implementation that does not
   require Fortran, see :doc:`torch_featurization`. The PyTorch implementation
   provides identical results with GPU acceleration support.

``aenet-python`` can be used to featurize atomic environments with the
expansion method by Artrith *et al.* [1,2,3].  Local atomic environment
features can, furthermore, be combined to atomic structure features with
the approach by Gharakhanyan *et al.* [4].

[1] N. Artrith and A. Urban, *Comput. Mater. Sci.* **114**, 2016, 135-150
(`link4 <http://dx.doi.org/10.1016/j.commatsci.2015.11.047>`_).

[2] N. Artrith, A. Urban, and G. Ceder,
*Phys. Rev. B* **96**, 2017, 014112 (`link1 <https://doi.org/10.1103/PhysRevB.96.014112>`_).

[3] A. M. Miksch, T. Morawietz, J. Kästner, A. Urban, N. Artrith,
*Mach. Learn.: Sci. Technol.* **2**, 2021, 031001 (`link2 <http://doi.org/10.1088/2632-2153/abfd96>`_).

[4] V. Gharakhanyan, M. S. Aalto, A. Alsoulah, N. Artrith, A. Urban,
ICLR 2023 (`link3 <https://openreview.net/forum?id=4Hl8bjobpl9>`_)


Example notebooks
-----------------

Jupyter notebooks with examples how to use the featurization methods can
be found in the `notebooks
<https://github.com/atomisticnet/aenet-python/tree/master/notebooks>`_
directory within the repository.
