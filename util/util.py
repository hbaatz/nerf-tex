"""Shared utility classes & functions."""

from typing import Any
import importlib
import subprocess
import math

class EasyDict(dict):
    """Encapsulation of a dict to allow access via attribute syntax."""

    def __init__(self, other: dict) -> None:
        for key in other:
            value = other[key]
            if isinstance(value, dict):
                value = EasyDict(value)
            self[key] = value

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]

def get_attr_from_module(module_name: str, attr_name: str) -> Any:
    """Return attribute from a given module."""

    module = importlib.import_module(module_name)
    return getattr(module, attr_name)

def get_attr_from_path(path: str) -> Any:
    """Return attribute from a module.attribute path."""

    module_name = '.'.join(path.split('.')[:-1])
    attr_name = path.split('.')[-1]

    return get_attr_from_module(module_name, attr_name)

def instantiate(config: EasyDict) -> Any:
    """Instantiate class / evaluate function from a given module path name with given kwargs."""

    if config is None:
        return None

    args = EasyDict(config)
    module = args.module
    del args.module

    return get_attr_from_path(module)(**args)

def format_name(prefix: str, idx: int, max_idx: int, suffix: str) -> str:
    """Returns name such that idx is padded with 0 to be able to fit max_idx."""

    n_chars = math.ceil(math.log10(max_idx + 1))
    format_str = '{:0' + str(n_chars) + 'd}'

    return prefix + format_str.format(idx) + suffix

def get_git_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')