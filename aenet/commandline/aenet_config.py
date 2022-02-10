#!/usr/bin/env python3

import sys

from aenet.commandline.tools import AenetToolABC
import aenet.config as cfg

__author__ = "The aenet developers"
__email__ = "aenet@atomistic.net"
__date__ = "2020-11-30"


class Config(AenetToolABC):
    """
    Read and write settings from/to the aenet configuration file(s).

    """

    def _set_arguments(self):
        self.parser.add_argument(
            "--write", "-w",
            help="Write a setting to the configuration file.",
            default=None,
            action='append',
            metavar=('<setting>', '<value>'),
            nargs=2)

        self.parser.add_argument(
            "--read", "-r",
            help="Read a setting from the configuration file.",
            default=None,
            action='append',
            metavar=('<setting>',),
            nargs=1)

        self.parser.add_argument(
            "--file",
            help="Path to the configuration file.  If no path is specified "
                 "the default configuration file will be used.",
            default=None)

        self.parser.add_argument(
            "--replace",
            help="Replace existing config file.",
            action="store_true")

    def _man(self):
        return """
        Read a setting from the configuration file and print to screen:

          $ aenet config --read <setting>

        Write value for a setting to the configuration file:

          $ aenet config --write <setting> <value>

        Multiple read and write instructions can be combined.  Write
        instructions will be performed before read instructions.

        If no argument is specified, the current configuration and the
        path to the configuration file will be printed out.

        """

    def run(self, args):
        if args.write is not None:
            config_dict = dict(args.write)
            cfg.write_config(config_dict, config_file=args.file)
        if args.read is not None:
            print(args.read)
            config_dict = cfg.read_config(config_file=args.file)
            for setting in args.read:
                if setting in config_dict:
                    print("'{}' = {}".format(setting, config_dict[setting]))
                else:
                    print("'' is not currently set.")
        if args.write is None and args.read is None:
            config_file = cfg.config_file_path()
            if config_file is None:
                print("No configuration file found. Using defaults.")
            else:
                print("Configuration file: {}".format(config_file))
            config_dict = cfg.read_config(config_file=args.file)
            print(config_dict)


if __name__ == "__main__":
    tool = Config()
    args = tool.parser.parse_args()
    tool.run(args)
