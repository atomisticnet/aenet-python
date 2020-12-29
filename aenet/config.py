"""
Manage configurations.

"""

import json
import os
import shutil
import sys

__author__ = "The aenet developers"
__email__ = "aenet@atomistic.net"
__date__ = "2020-11-28"

DEFAULT = {
    "aenet_install_dir": None
}

PATHS = [
    '.',
    os.path.join(os.path.expanduser("~"), ".config", "aenet")
]


def config_file_path():
    """
    Search for configuration file and return path if found.

    """
    config_file = None
    for p in PATHS:
        path = os.path.join(p, "config.json")
        if os.path.exists(path):
            config_file = path
            break
    return config_file


def valid_setting(setting):
    """
    Check if a given setting actually exists.

    """
    return setting in DEFAULT


def read_config(config_file=None):
    """
    Search for configuration file and read if found.

    """
    config_dict = DEFAULT.copy()
    if config_file is None:
        config_file = config_file_path()
    if config_file is not None:
        with open(config_file) as fp:
            config_dict.update(json.load(fp))
    return config_dict


def write_config(config_dict, config_file=None, replace=False):
    """
    Write settings to a configuration file.

    Args:
      config_dict (dict): dict with settings
      config_file (str): (optional) path to a configuration file
      replace (bool): if True, replace existing configuration file

    """
    settings = list(config_dict.keys())
    for setting in settings:
        if not valid_setting(setting):
            config_dict.pop(setting)
            sys.stderr.write("Warning: unknown setting "
                             "'{}' ignored.\n".format(setting))
    if config_file is None:
        config_file = config_file_path()
    if config_file is None:
        os.makedirs(PATHS[-1])
        config_file = os.path.join(PATHS[-1], "config.json")
    else:
        shutil.copy2(config_file, config_file + ".bak")
        if not replace:
            config_dict_orig = read_config(config_file)
            config_dict_orig.update(config_dict)
            config_dict = config_dict_orig
    with open(config_file, "w") as fp:
        json.dump(config_dict, fp)
