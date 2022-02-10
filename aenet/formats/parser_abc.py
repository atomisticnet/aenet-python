"""
Abstract parser class to be inherited from.

"""

from abc import ABCMeta, abstractmethod
import numpy as np
import sys

__author__ = "Alexander Urban"
__email__ = "alexurba@mit.edu"
__date__ = "2014-10-08"
__version__ = "0.1"
__changes__ = """
2015-12-09 AU --- implemented _amend() method
"""


class ParserABC(object):
    __metaclass__ = ABCMeta

    _amend_args = ['energy', 'forces']

    @abstractmethod
    def __init__(self):
        self.name = None
        self.description = None
        self.extensions = None
        self.default_file_names = None

    @property
    def readable(self):
        return 'read' in self.__class__.__dict__

    @property
    def writable(self):
        return 'write' in self.__class__.__dict__

    def read(self, filename, **kwargs):
        raise NotImplementedError("No parser for this format available.")

    def write(self, struc, filename, **kwargs):
        raise NotImplementedError("Output not implemented for this format.")

    def __str__(self):
        out = " {:10s}  ".format(self.name)
        out += "{:30s}  ".format(self.description)
        out += "{:3s}    ".format("yes" if self.readable else "no ")
        out += "{:3s}    ".format("yes" if self.writable else "no ")
        for ext in self.extensions:
            out += "{} ".format(ext)
        return out

    def _amend(self, struc, energy=None, forces=None, **kwargs):
        """
        Add information or metadata to an AtomicStructure instance after
        reading.  To be called at he end of 'read()' with the remaining
        keyword arguments.

        Most data will be assigned to the final frame of a multi-frame
        structure only!

        Arguments:
          structure    An instance of AtomicStructure
          energy       Energy of the last frame of structure
          forces       Set all atomic force components to this value
                       (mainly useful to set all forces to 0)

        All keyword arguments supported by '_amend()' should be in the
        list '_amend_args'.

        Does not return anything; the input structure will be modified.

        """

        for k in kwargs:
            sys.stderr.write("Warning: unsupported keyword: {}\n".format(k))

        if energy is not None:
            if struc.nframes > 1:
                sys.stderr.write(
                    "Warning: Energy assigned to final frame only.\n")
            if struc.energy[-1] is not None:
                sys.stderr.write(
                    "Warning: Overwriting existing energy value.\n")
            struc.energy[-1] = float(energy)

        if forces is not None:
            if struc.nframes > 1:
                sys.stderr.write(
                    "Warning: Assigning same forces to all "
                    "{} frames.\n".format(struc.nframes))
            if (struc.forces[-1] is not None) and (len(struc.forces[-1]) > 0):
                sys.stderr.write(
                    "Warning: Overwriting existing force values.\n")
            for i in range(struc.nframes):
                struc.forces[i] = (np.zeros_like(struc.coords[-1]) +
                                   float(forces))

    def _check_amend_args(self, **kwargs):
        """
        Check if all keyword arguments are in the set of allowed arguments
        for the _amend() method.  Print warning message to stdout for
        each unsupported argument.

        Call this method when entering 'read()'.

        """
        for k in kwargs:
            if k not in self._amend_args:
                sys.stderr.write(
                    "Warning: unsupported keyword: {}\n".format(k))
