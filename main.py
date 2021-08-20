"""Load config & set up everything."""

import os
import shutil
import argparse
import importlib
import numpy as np
import tensorflow as tf
from util import util, EasyDict
from network import train, render

def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train network as specified in config file.')
    parser.add_argument('config', help='Path to config file.')
    args = parser.parse_args()

    # Clip away .py ending if neccessary and replace / by .
    config_path = args.config[:-3] if args.config[-3:] == '.py' else args.config
    config_module = config_path.replace('/', '.')

    # Import config file
    config = EasyDict(importlib.import_module(config_module).config)
    # Add config dict to the logger config to pass them forward to raptor
    config_copy = EasyDict(config)
    del config_copy.logger_config
    config.logger_config.update({'info': config_copy})

    # Set random seed
    if config.seed is not None:
        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)

    # Create target folder and copy config file
    os.makedirs(config.target_path, exist_ok = config.override)
    infix = 'train' if 'train' in config.module else 'render'
    config_copy_path = os.path.join(config.target_path, 'config_' + infix + '.py')
    try:
        shutil.copy(config_path + '.py', config_copy_path)
    except shutil.SameFileError:
        pass
    
    # Append git commit hash to config file
    label = util.get_git_hash()
    with open(config_copy_path, 'a') as config_file:
        config_file.write('\n# GIT COMMIT HASH: ' + label)

    # Instantiate top level module
    util.instantiate(config)

if __name__ == '__main__':
    main()