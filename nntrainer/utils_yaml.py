"""
YAML file loading and saving utilities.
"""

from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Union, Mapping
from collections import Mapping as CollectionsMapping

import yaml

from nntrainer.typext import PathType


def load_yaml_config_file(yaml_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load given yaml file. Supports loading scientific floats like 1e-8 as python floats. Preserves key order.

    Args:
        yaml_file: File to load

    Returns:
        Loaded config as nested dict.
    """
    yaml_str = Path(yaml_file).read_text(encoding="utf8")
    return convert_yaml_to_dict(yaml_str)


def convert_yaml_to_dict(yaml_str: str) -> Dict[str, Any]:
    """
    Load given yaml string. Supports loading scientific floats like 1e-8 as python floats. Preserves key order.

    Args:
        yaml_str: String to load

    Returns:
        Loaded config as nested dict.

    Returns:
        Loaded config as nested dict.
    """
    # convert yaml to ordered dict
    config: Dict[str, Any] = yaml.load(yaml_str, Loader=yaml.SafeLoader)

    # support loading scientific float values
    def post_process(d: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Recursively parse a dict and try to convert all strings to floats, fail silently.

        Args:
            d: Input dict.

        Returns:
            Dict where strings like "1e-2" have been converted to the corresponding float 0.01
        """
        new_od = {}
        for key, val in d.items():
            if isinstance(val, str):
                # try to convert strings to float and ignore errors
                try:
                    val = float(val)
                except ValueError:
                    pass
            elif isinstance(val, CollectionsMapping):
                # call this function recursively
                val = post_process(val)
            else:
                pass
            # write converted values to the return dict
            new_od[key] = val
        return new_od

    config = post_process(config)
    return config


def convert_dict_to_yaml(input_dict: Dict[str, Any], indent_spaces: int = 4, indent_level: int = 0) -> str:
    """
    The original yaml.dump needed improvements, this is a recursive re-implementation

    yaml.dump(config_dict)

    Args:
        input_dict: Dict to be converted to yaml.
        indent_spaces: How many spaces per indent level.
        indent_level: Current indent level for the recursion.

    Returns:
        YAML string.
    """
    # setup key-value collector and indent level
    ret_list = []
    indent = " " * (indent_level * indent_spaces)
    # loop input dict
    for key, value in input_dict.items():
        # setup collector for single key-value pair
        single_ret_list = [f"{indent}{key}:"]
        # check type
        if isinstance(value, bool):
            # bools as lower-case
            value_str = str(value).lower()
        elif isinstance(value, (int, float)):
            # leave float conversion to python
            value_str = str(value)
        elif isinstance(value, str):
            # put quotes around strings
            value_str = f"\"{value}\""
        elif value is None:
            # None is null in yaml
            value_str = "null"
        elif isinstance(value, dict):
            # iterate dictionaries recursively
            value_str = "\n" + convert_dict_to_yaml(value, indent_spaces=indent_spaces, indent_level=indent_level + 1)
        else:
            raise ValueError(f"dict to yaml, value type not understood: {value}")
        # concatenate the single key-value pair and add it to the key-value collector
        single_ret_list += [f" {value_str}"]
        ret_list += ["".join(single_ret_list)]
    # join the collected key-value pairs with newline
    return "\n".join(ret_list)


def dump_yaml_config_file(filename: PathType, config_dict: Dict[str, Any]) -> None:
    """
    Store dictionary as YAML file. Changes indent to 4, formats strings with quotes.

    Args:
        filename: Target filename.
        config_dict: Input dictionary.
    """

    # convert dict to yaml string
    s = convert_dict_to_yaml(config_dict)

    # write to file
    Path(filename).open("wt", encoding="utf8").write(s)

    # make sure that if it's converted back via yaml, it's still the same dict
    test_config_dict = load_yaml_config_file(filename)
    if config_dict != test_config_dict:
        # verbose error printing
        print("---------- Original config:")
        pprint(config_dict)
        print()
        print("---------- Reloaded config:")
        pprint(test_config_dict)
        print()
        raise ValueError("Config has changed during yaml saving, this is an implementation error!")
