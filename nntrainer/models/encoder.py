"""
Positional encoding for transformer input.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch as th
from torch import nn

from nntrainer import typext, utils


def make_encoder_module(dim: int, name: str, cfg: Optional[EncoderConfig] = None) -> Optional[nn.Module]:
    """
    Make embedding module instance given by name and config object.


    Returns:
        Normalization function class.
    """
    if cfg is None:
        # set all module hyperparameters to default values
        cfg = EncoderConfig(name)

    # create the module instance
    if name == PositionalEncodingConst.SINCOS:
        return PositionalEncodingSinCos(dim, dropout_prob=cfg.dropout_prob, max_len=cfg.max_len)
    if name == PositionalEncodingConst.NONE:
        return None
    raise ValueError(f"Embedding name unknown: {name}")


class EncoderConst(typext.ConstantHolder):
    NONE = utils.NONE
    SINCOS = "sincos"


class EncoderConfig(typext.ConfigClass):
    """
    Activation function.

    Args:
        name_or_config: Either provide string name of the activation function (e.g. "relu") or a dict with name and
            hyperparameters (e.g. {"name": "leakyrelu", "negative_slope": 1e-2})
    """

    def __init__(self, name_or_config: Union[str, Dict[str, Any]]):
        # Determine if configuration is given by a string name or a config dict.
        if isinstance(name_or_config, str):
            config: Dict[str, Any] = {}
            self.name = name_or_config
        elif isinstance(name_or_config, dict):
            config = name_or_config
            self.name = config.pop("name")
        else:
            raise ValueError(f"Type {name_or_config} not understood.")
        # Set optional fields.
        self.dropout_prob = config.pop("dropout_prob", 0)
        self.max_len = config.pop("max_len", 1000)


# ---------- Module Implementations. ----------


class PositionalEncodingSinCos(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Modified: Removed embedding, changed the calculation. Should be fine.

    Args:
        dim: embedding size
        dropout_prob: dropout parameter
        max_len: Maximum input length.
    """

    def __init__(self, dim: int, dropout_prob: float = 0., max_len: int = 1000):
        super().__init__()
        pe = th.zeros(max_len, dim).float()
        position = th.arange(0, max_len).unsqueeze(1).float()
        dimension = th.arange(0, dim).float()
        div_term = 10000 ** (2 * dimension / dim)
        # print(div_term.shape)
        pe[:, 0::2] = th.sin(position / div_term[0::2])
        pe[:, 1::2] = th.cos(position / div_term[1::2])
        # print(f"Positional Encoding max_len x dim: {pe.shape}\n", pe, "\n",
        #       sep="")
        # div_term = torch.exp((torch.arange(0, dim, 2) *
        #                       -(math.log(10000.0) / dim)).float())
        # pe[:, 0::2] = torch.sin(position.float() * div_term)
        # pe[:, 1::2] = torch.cos(position.float() * div_term)
        # pe = pe.unsqueeze(0)

        # put it into state dict even though it is not learnable
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim

    def forward(self, x, step=None):
        # x *= math.sqrt(self.dim) # not sure
        # print(x.size(1))
        assert step is None, "Never used step"
        x = x + self.pe[:x.shape[1], :]
        # else:
        #     # x = x + self.pe[:, step]
        x = self.dropout(x)
        return x


class PositionalEncodingConst(typext.ConstantHolder):
    """
    Positional encoding modules.
    """
    NONE = utils.NONE
    SINCOS = "sincos"
