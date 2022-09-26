"""
Object classes for aenet command line tools.  Each tool is a
singleton.

"""

import abc
import argparse
import inspect

__author__ = "The aenet developers"
__email__ = "aenet@atomistic.net"
__date__ = "2020-11-30"


class AenetToolABC(object):
    """
    Attributes:
      subparsers: an instance of an argparse subparsers

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, subparsers=None):
        self.name = self.__class__.__name__.lower()
        descr = (inspect.cleandoc(self.__doc__)
                 + "\n\n{} {}\n\n".format(__date__, __author__)
                 + inspect.cleandoc(self._man()))
        if subparsers is not None:
            self.parser = subparsers.add_parser(
                self.name,
                help=self.__doc__,
                description=descr,
                formatter_class=argparse.RawDescriptionHelpFormatter)
        else:
            self.parser = argparse.ArgumentParser(
                description=descr,
                formatter_class=argparse.RawDescriptionHelpFormatter)
        self.parser.set_defaults(run=self.run)
        self._set_arguments()

    def _set_arguments(self):
        """
        Use this method to add command line argument parsers to self.parser.

        Example:

        self.parser.add_argument(
          "--path", help="Path to somewhere (default: .).", default=".")

        """
        pass

    def _man(self):
        """
        The return value of this private method is the manual entry to be
        added to the tool's help message (i.e., shown when the
        ``--help`` flag is passed).

        """
        manual = ""
        return manual

    @abc.abstractmethod
    def run(self, args):
        """
        Arguments:
          args: object returned from an 'argparse' parser
        """
        pass
