Command-line tools
======================

**ænet-python** package not only offers the python interface, but also
serves as a collection of command-line tools.  Each tool is implemented 
in a separate Python script located in ``aenet/commandline/``, but they 
are exposed through the unified shell command ``aenet``.  The ``--help`` 
flag returns a list of all available command-line tools:

.. sourcecode:: console

   $ aenet --help

or ``aenet`` for simple.

``config``
----------

The tool :doc:`tools/config` is merely used to change settings, for
example, to define the path in which ænet has been installed. Examples
can be found on :doc:`installation`.

``sconv``
----------

:doc:`tools/sconv`, short for *structure converter*, is a very versatile
tool. Some important functions are:

* extract structure/energy/force information from the output of common 
  electronic-structure codes;

* interconvert formats of atomic structures, especially to ænet's XSF format;

* perform manipulations on atomic structures, such as scaling or distortion, 
  which is useful for reference data generation.

Examples can be found on :doc:`examples`. 

``sfp``
-------------
:doc:`tools/sfp` is used to featurize atomic structures by calculating structure
*fingerprints*. For example, it can parse a training set file (binary type) produced 
by ``generate.x`` and convert it to ASCII format (a human-readable format).


See below for a list of all 
documented command-line tools.

.. note::
   Implementing new command-line tools is straighforward.  See
   also :doc:`../dev/commandline`

Tools:
"""""""""
.. toctree::
   :maxdepth: 1
   :glob:

   tools/config
   tools/sconv
   tools/sfp
