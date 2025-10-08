"""
Automatically search the 'formats' directory for parser
implementations and collect them in a dictionary.

"""

__author__ = "Alexander Urban, Nongnuch Artrith"
__email__ = "alexurba@mit.edu, nartrith@mit.edu"
__date__ = "2014-10-08"
__version__ = "0.1"

import glob
import os
from .parser_abc import ParserABC

parser_files = glob.glob(os.path.join(os.path.dirname(__file__), '*.py'))
parser_packages = [os.path.basename(f)[:-3] for f in parser_files]
del parser_packages[parser_packages.index("__init__")]
del parser_packages[parser_packages.index("parser_abc")]

parser_module = []
for package in parser_packages:
    parser_module.append(__import__('aenet.formats.' + package))

formats = {}
for p in ParserABC.__subclasses__():
    frmt = p()
    formats[frmt.name] = frmt
