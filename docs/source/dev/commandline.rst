Defining Command-line tools
=============================

Each ænet tool is implemented in a separate file named ``aenet_*.py`` in
``aenet/commandline/``, implementing a class that inherits from the
abstract base class :class:`aenet.commandline.tools.AenetToolABC`.

If the naming convention ``aenet_*.py`` is followed, the commandline
tool will be automatically discovered and added as a subparser to the
``aenet`` comand.

API Reference
----------------

.. autoclass:: aenet.commandline.tools.AenetToolABC
   :members:
   :undoc-members:
   :private-members:
