#!/usr/bin/env python

"""
Generic I/O module of the strucconv package.

This module provides generic read and write routines that interface the
corresponding routines for the various file formats.
"""

import sys
import numpy as np

from ..exceptions import ArgumentError
from ..exceptions import FormatError
from ..exceptions import FormatGuessError
from ..exceptions import ReadonlyError
from ..exceptions import WriteonlyError
from ..formats import formats

__author__ = "Alexander Urban"
__date__ = "2013-03-25"


def print_supported_formats():
    """
    Print list of supported file formats to stdout.
    """

    print("\n Supported file formats:\n")
    print(" Format      description                     "
          "read   write  extensions")
    print(" " + 68*"-")
    for frmt in np.sort([f for f in formats]):
        print(formats[frmt])
    print("")


def readable_formats():
    """
    Return list of all readable formats.

    """

    return [f for f in formats if formats[f].readable]


def writable_formats():
    """
    Return list of all writable formats.

    """

    return [f for f in formats if formats[f].writable]


def read(filename, frmt=None, **kwargs):
    """
    Read atomic structure file of the specified format.
    This function only returns the most common data.  Use the format
    specific read functions to extract further data from atomic
    structure files.

    Input:
      filename    name of the input file
      format      name of the file format
      kwargs      further keyword arguments are passed on to the backend

    Returns:
      instance of class AtomicStructure
    """

    if not frmt:
        frmt = guess_format(filename)

    if not formats[frmt].readable:
        raise WriteonlyError(frmt)

    return formats[frmt].read(filename, **kwargs)


def read_safely(filename, frmt=None, **kwargs):
    """
    Same as `read()', but handle some internal exceptions nicely.
    """

    try:
        struc = read(filename, frmt=frmt, **kwargs)
    except ArgumentError as err:
        sys.stderr.write("Error: {}\n".format(err.msg))
        sys.exit()
    except FormatGuessError as err:
        sys.stderr.write("Error: {}\n".format(err.msg))
        print_supported_formats()
        sys.exit()
    except WriteonlyError as err:
        sys.stderr.write("Error: {}\n".format(err.msg))
        print_supported_formats()
        sys.exit()
    return struc


def write(struc, filename=None, frmt=None, **kwargs):
    """
    Write atomic structure to a file in the specified format.

    Input:
      struc     instance of the AtomicStructure class
      filename  name of the output file; if None, the backend will
                write to stdout
      frmt      the name of the file format; if None, the format will
                be guessed from the file extension
      kwargs    further keyword arguments are passed on to the backend
    """

    if not frmt:
        frmt = guess_format(filename)

    if frmt not in formats.keys():
        raise FormatError(frmt)

    if not formats[frmt].writable:
        raise ReadonlyError(frmt)

    formats[frmt].write(struc, filename, **kwargs)


def write_safely(struc, filename=None, frmt=None, **kwargs):
    """
    Same as `write()', but handles some internal exceptions nicely.
    """

    try:
        write(struc, filename=filename, frmt=frmt, **kwargs)
    except ReadonlyError as err:
        sys.stderr.write("Error: {}\n".format(err.msg))
        print_supported_formats()
        sys.exit()
    except ArgumentError as err:
        sys.stderr.write("Error: {}\n".format(err.msg))
        sys.exit()


def guess_format(filename):
    """
    Guess the file format from the file extension.

    Input:
      filename   name of the file
    """

    # first, try the default file names
    for f in formats:
        for d in formats[f].default_file_names:
            if d in filename:
                return f
    # second, match against file extensions
    ext = filename.split('.')[-1]
    for f in formats:
        if ext in formats[f].extensions:
            return f

    # not successful --> raise Exception
    raise FormatGuessError(filename)
