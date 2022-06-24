"""
Test configuration file setup.
"""

from typing import Dict

import yaml

from nntrainer.typext import ConfigClass
from nntrainer.utils import check_config_dict, resolve_sameas_config_recursively


CONFIG_YAML = (
    """
description: "test config"
ref:
    mymodule:
        layers: 6
        units: 20
network1:
    module1:
        same_as: "ref.mymodule"
    module2:
        same_as: "ref.mymodule"
        units: 30
network2:
    same_as: "network1"
"""
)


class ExperimentTestConfig(ConfigClass):
    def __init__(self, config: Dict) -> None:
        self.description = config.pop("description")
        self.network1 = NetworkTestConfig(config.pop("network1"))
        self.network2 = NetworkTestConfig(config.pop("network2"))
        check_config_dict("experiment", config)


class NetworkTestConfig(ConfigClass):
    def __init__(self, config: Dict) -> None:
        self.module1 = ModuleTestConfig(config.pop("module1"))
        self.module2 = ModuleTestConfig(config.pop("module2"))
        check_config_dict("network", config)


class ModuleTestConfig(ConfigClass):
    def __init__(self, config: Dict) -> None:
        self.layers = config.pop("layers")
        self.units = config.pop("units")
        check_config_dict("module", config)


def test_config_loading():
    # convert yaml to dict
    config = yaml.load(CONFIG_YAML, Loader=yaml.SafeLoader)
    assert config == {'description': 'test config', 'ref': {'mymodule': {'layers': 6, 'units': 20}},
                      'network1': {'module1': {'same_as': 'ref.mymodule'},
                                   'module2': {'same_as': 'ref.mymodule', 'units': 30}},
                      'network2': {'same_as': 'network1'}}
    # resolve same_as fields to values
    resolve_sameas_config_recursively(config)
    assert config == {'description': 'test config', 'ref': {'mymodule': {'layers': 6, 'units': 20}},
                      'network1': {'module1': {'layers': 6, 'units': 20},
                                   'module2': {'layers': 6, 'units': 30}},
                      'network2': {'module1': {'layers': 6, 'units': 20},
                                   'module2': {'layers': 6, 'units': 30}}}
    # load it
    cfg = ExperimentTestConfig(config)
    print(cfg)


if __name__ == "__main__":
    test_config_loading()
