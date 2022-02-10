Command-line interface
======================

**ænet tools**, as the name implies, is primarily a collection of
command-line tools.  Each tool is implemented in a separate Python
script located in ``aenet/commandline/``, but they are exposed through
the unified shell command ``aenet``.  The ``--help`` flag returns a list
of all available command-line tools:

.. sourcecode:: console

   $ aenet --help

The tool :doc:`tools/config` is merely used to change settings, for
example, to define the path in which ænet has been installed.

A very versatile tool is :doc:`tools/sconv`, short for *structure converter*.
:doc:`tools/sconv` can (i) extract structure/energy/force information from the
output of common electronic-structure codes, (ii) interconvert atomic
structure formats, especially to ænet's XSF format, and (iii) perform
manipulations on atomis structures, such as scaling or distortion, which
is useful for reference data generation.

See below for a list of all documented command-line tools.

.. note::
   Implementing new command-line tools is straighforward.  See
   also :doc:`../dev/commandline`

.. toctree::
   :maxdepth: 2
   :caption: Tools:
   :glob:

   tools/config
   tools/sconv
   tools/sfp
