"""
Command line interface.

"""

import glob
import os
import sys
from aenet.commandline.tools import AenetToolABC

__author__ = "The aenet developers"
__email__ = "aenet@atomistic.net"
__date__ = "2020-11-30"


def discover(subparsers):
    # Import all python packages named aenet-*.py from the tool directory
    tool_files = glob.glob(
        os.path.join(os.path.dirname(__file__), 'aenet_*.py'))
    tool_packages = [os.path.basename(f)[:-3] for f in tool_files]
    for t in tool_packages:
        try:
            __import__('aenet.commandline.' + t)
        except ImportError:
            sys.stderr.write(
                "Warning: Failed to import commandline tool '{}'.".format(t))
    # The name of each tool is generated from the name of the
    # corresponding class.  This is a little redundant with the ".name"
    # attribute, but the latter is only created upon instantiation.
    tool_classes = {sg.__name__.lower(): sg for sg in
                    AenetToolABC.__subclasses__()}
    tools = {}
    for t in sorted(tool_classes):
        tool = tool_classes[t](subparsers)
        tools[tool.name] = tool
