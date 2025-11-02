ænet training set files
=======================

The Python class ``TrnSet`` in ``aenet.trainset`` module, can be used to
interact with data set files produced by ænet's ``generate.x`` tool.

.. note::
   The ænet executable ``trnset2ASCII.x`` needs to be configured to read
   training set files.  See also :doc:`installation`.

File formats
------------

Internally, ænet uses *unformatted* Fortran binary files to store the
featurized atomic structure data for training.  Since the
format of such files is compiler dependent, it is not straightforward to
parse them directly with Python.  Instead, the ``TrnSet`` class converts
binary data set files to plain text using the ``trnset2ASCII.x`` ænet
tool.  Text-based files can be further converted to HDF5 format to save
space and to allow for more efficient I/O.  These conversions are done
transparently:

.. sourcecode:: python

   from aenet.trainset import TrnSet
   ts = TrnSet.from_file('data.train')

This opens the training set file ``data.train`` which can be in any of
the three supported formats (Fortran binary, ASCII text, or HDF5).

API Reference
"""""""""""""""

.. autoclass:: aenet.trainset.TrnSet
   :members:
   :private-members:
   :undoc-members:
   :noindex:
