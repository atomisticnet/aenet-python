"""
Command line interface with smart caching for fast startup.

"""

import glob
import json
import os
import sys
from pathlib import Path
from aenet.commandline.tools import AenetToolABC

__author__ = "The aenet developers"
__email__ = "aenet@atomistic.net"
__date__ = "2020-11-30"


def get_cache_path():
    """
    Get the path to the tool cache file.

    Returns:
        Path: Path to the cache file in user's cache directory
    """
    # Use platform-appropriate cache directory
    if sys.platform == 'win32':
        cache_base = Path.home() / 'AppData' / 'Local'
    else:
        # Linux/macOS
        cache_base = Path.home() / '.cache'

    cache_dir = cache_base / 'aenet'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / 'commandline_tools.json'


def load_cache():
    """
    Load the tool cache from disk.

    Returns:
        dict: Cache data or empty dict if cache doesn't exist or is invalid
    """
    cache_path = get_cache_path()
    if not cache_path.exists():
        return {}

    try:
        with open(cache_path, 'r') as f:
            cache = json.load(f)
        return cache
    except (json.JSONDecodeError, IOError):
        # Corrupted cache, return empty
        return {}


def save_cache(cache):
    """
    Save the tool cache to disk.

    Args:
        cache (dict): Cache data to save
    """
    cache_path = get_cache_path()
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        sys.stderr.write(
            f"Warning: Failed to save tool cache: {e}\n")


def is_cache_valid(cache, tool_files):
    """
    Check if the cache is up to date with current tool files.

    Args:
        cache (dict): Cached tool data
        tool_files (list): List of current tool file paths

    Returns:
        bool: True if cache is valid, False otherwise
    """
    if not cache or cache.get('version') != 1:
        return False

    cached_files = cache.get('tool_files', {})

    # Check if file list changed
    current_paths = {os.path.abspath(f) for f in tool_files}
    cached_paths = set(cached_files.keys())
    if current_paths != cached_paths:
        return False

    # Check modification times
    for filepath in tool_files:
        abs_path = os.path.abspath(filepath)
        try:
            current_mtime = os.path.getmtime(filepath)
            cached_mtime = cached_files.get(abs_path, 0)
            if current_mtime > cached_mtime:
                return False
        except OSError:
            # File no longer exists or inaccessible
            return False

    return True


def rebuild_cache(tool_files):
    """
    Rebuild the cache by importing all tool modules to extract metadata.

    Args:
        tool_files (list): List of tool file paths to process

    Returns:
        dict: New cache data
    """
    cache = {
        'version': 1,
        'tools': {},
        'tool_files': {}
    }

    for filepath in tool_files:
        abs_path = os.path.abspath(filepath)
        module_name = os.path.basename(filepath)[:-3]

        try:
            # Import the module
            mod = __import__(f'aenet.commandline.{module_name}',
                             fromlist=[''])

            # Find tool classes in module
            for name, obj in vars(mod).items():
                if (isinstance(obj, type) and
                        issubclass(obj, AenetToolABC) and
                        obj is not AenetToolABC):
                    tool_name = name.lower()

                    # Extract help text from docstring
                    help_text = ''
                    if obj.__doc__:
                        # Get first line of docstring as help
                        help_text = obj.__doc__.strip().split('\n')[0]

                    cache['tools'][tool_name] = {
                        'module': f'aenet.commandline.{module_name}',
                        'class': name,
                        'help': help_text,
                        'mtime': os.path.getmtime(filepath)
                    }

            cache['tool_files'][abs_path] = os.path.getmtime(filepath)

        except ImportError as e:
            sys.stderr.write(
                "Warning: Failed to import commandline tool"
                + f" '{module_name}': {e}\n")

    return cache


def discover(subparsers):
    """
    Discover available tools using smart caching for fast startup.

    This function:
    1. Scans for tool files (aenet_*.py)
    2. Checks if cache is valid
    3. If a specific tool is requested, imports only that tool fully
    4. Otherwise, uses cache for fast help listing

    Args:
        subparsers: argparse subparsers object to register tools with

    Returns:
        dict: Mapping of tool names to tool metadata
    """
    # Find all tool files
    tool_dir = os.path.dirname(__file__)
    tool_files = glob.glob(os.path.join(tool_dir, 'aenet_*.py'))

    # Load and validate cache
    cache = load_cache()
    cache_valid = is_cache_valid(cache, tool_files)

    if not cache_valid:
        # Rebuild cache by importing all tools
        cache = rebuild_cache(tool_files)
        save_cache(cache)

    tools_info = cache.get('tools', {})

    # Check if a specific tool was requested
    tool_requested = None
    if len(sys.argv) > 1 and sys.argv[1] in tools_info:
        tool_requested = sys.argv[1]

    if tool_requested:
        # Import only the requested tool fully for complete functionality
        tool_info = tools_info[tool_requested]
        try:
            module = __import__(tool_info['module'], fromlist=[''])
            tool_class = getattr(module, tool_info['class'])
            # Create tool with full parser (handles both --help and execution)
            tool_class(subparsers=subparsers)
        except (ImportError, AttributeError) as e:
            sys.stderr.write(
                f"Error: Failed to import tool '{tool_requested}': {e}\n")
            # Fall back to lightweight parser
            parser = subparsers.add_parser(
                tool_requested,
                help=tool_info['help']
            )
            parser.set_defaults(
                tool_module=tool_info['module'],
                tool_class=tool_info['class']
            )
    else:
        # No specific tool requested, use cache for fast help listing
        for tool_name in sorted(tools_info.keys()):
            tool_info = tools_info[tool_name]
            parser = subparsers.add_parser(
                tool_name,
                help=tool_info['help']
            )
            # Store module info for potential lazy loading
            parser.set_defaults(
                tool_module=tool_info['module'],
                tool_class=tool_info['class']
            )

    return tools_info


def lazy_load_tool(args):
    """
    Lazy load and instantiate the actual tool class when needed.

    This function is called after argument parsing to load only
    the specific tool that was requested.

    Args:
        args: Parsed arguments from argparse

    Returns:
        tuple: (tool instance, reparsed args) or (None, None) if no tool
    """
    if not hasattr(args, 'tool_module'):
        return None, None

    try:
        # Import the specific module
        module = __import__(args.tool_module, fromlist=[''])

        # Get the tool class
        tool_class = getattr(module, args.tool_class)

        # Instantiate the tool without subparsers (creates standalone parser)
        tool = tool_class(subparsers=None)

        # Re-parse arguments with the tool's full parser to get all arguments
        # We need to parse from sys.argv directly since we need the full args
        reparsed_args = tool.parser.parse_args(sys.argv[2:])

        return tool, reparsed_args
    except (ImportError, AttributeError) as e:
        sys.stderr.write(
            f"Error: Failed to load tool: {e}\n")
        return None, None
