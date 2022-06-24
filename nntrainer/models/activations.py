"""
Activation functions.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

from torch import nn

from nntrainer import typext, utils


def make_activation_module(name: str, cfg: Optional[ActivationConfig] = None) -> nn.Module:
    """
    Get activation module instance given by name and configuration object.

    Args:
        name:
        cfg: Hyperparameter config

    Returns:
        Activation module.
    """
    err_msg_no_cfg = f"Activation Function {name} needs a configuration, but None was given."
    if name == ActivationConst.NONE:
        return nn.Identity()
    if name == ActivationConst.RELU:
        return nn.ReLU()
    if name == ActivationConst.GELU:
        return nn.GELU()
    if name == ActivationConst.LEAKYRELU:
        assert cfg is not None, err_msg_no_cfg
        return nn.LeakyReLU(negative_slope=cfg.negative_slope)
    raise ValueError(f"{name} not found in {ActivationConst.values()}")


class ActivationConst(typext.ConstantHolder):
    NONE = utils.NONE
    RELU = "relu"
    GELU = "gelu"
    LEAKYRELU = "leakyrelu"  # params: negative_slope (default 1/100)


class ActivationConfig(typext.ConfigClass):
    """
    Activation function.

    Examples:
        >>> ActivationConfig("relu")
        >>> ActivationConfig({"name": "leakyrelu", "negative_slope": 1e-2})

    Args:
        name_or_config: Either provides string name of the activation or a dict with name and hyperparameters.
    """

    def __init__(self, name_or_config: Union[str, Dict[str, Any]]):
        # Determine if configuration is given by a string name or a config dict.
        if isinstance(name_or_config, str):
            config = {}
            self.name = name_or_config
        else:
            config = name_or_config
            self.name = config.pop("name")
        # Set optional fields
        self.negative_slope = config.pop("negative_slope", 1e-2)
