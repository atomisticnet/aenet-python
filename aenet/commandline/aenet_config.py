#!/usr/bin/env python3

import json
import glob
import os

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
            "--set-aenet-path", "-p",
            help="Set the root path to aenet.",
            default=None)

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

    def set_aenet_paths(self, root_path):
        """
        Configure the paths to the aenet installation and to the executables.

        """
        aenet_dict = cfg.read('aenet')
        if os.path.exists(root_path):
            aenet_dict['root_path'] = os.path.abspath(root_path)
        else:
            current = aenet_dict['root_path']
            if current is None or not os.path.exists(current):
                raise FileNotFoundError('Path not found: {}'.format(root_path))

        print('aenet root path: {}'.format(aenet_dict['root_path']))

        def get_exec_path(name, subdir, current):
            if current is not None and os.path.exists(current):
                path_try = current
            else:
                path_try = glob.glob(
                    os.path.join(aenet_dict['root_path'], 
                                subdir, '{}*'.format(name)))
                path_try = path_try[0] if len(path_try) > 0 else ''
            path = input("Path to `{}` [{}]: ".format(name, path_try))
            path = path if len(path) > 0 else path_try
            if len(path) == 0 or not os.path.exists(path):
                print('Warning: Path to {} not found: {}'.format(name, path))
                path = None
            return path

        aenet_dict['generate_x_path'] = get_exec_path(
            'generate.x', 'bin', aenet_dict['generate_x_path'])
        aenet_dict['train_x_path'] = get_exec_path(
            'train.x', 'bin', aenet_dict['train_x_path'])
        aenet_dict['predict_x_path'] = get_exec_path(
            'predict.x', 'bin', aenet_dict['predict_x_path'])
        aenet_dict['trnset2ascii_x_path'] = get_exec_path(
            'trnset2ASCII.x', 'tools', aenet_dict['trnset2ascii_x_path'])
        return {'aenet': aenet_dict}

    def run(self, args):
        config_dict = {}
        if args.set_aenet_path is not None:
            config_dict.update(self.set_aenet_paths(args.set_aenet_path))
        if args.write is not None:
            config_dict = config_dict.update(dict(args.write))
        if config_dict:
            cfg.write_config(config_dict, config_file=args.file)
        if args.read is not None:
            print(args.read)
            config_dict = cfg.read_config(config_file=args.file)
            for setting in args.read:
                if setting in config_dict:
                    print("'{}' = {}".format(setting, config_dict[setting]))
                else:
                    print("'' is not currently set.")
        if (args.write is None and args.read is None 
                and args.set_aenet_path is None):
            config_file = cfg.config_file_path()
            if config_file is None:
                print("No configuration file found. Using defaults.")
            else:
                print("Configuration file: {}".format(config_file))
            config_dict = cfg.read_config(config_file=args.file)
            out = json.dumps(config_dict, indent=2, default=str)
            for o, r in [(': null', ': None'), 
                         (': true', ': True'), 
                         (': false', ': False')]:
                out = out.replace(o, r)
            print(out)


if __name__ == "__main__":
    tool = Config()
    args = tool.parser.parse_args()
    tool.run(args)
