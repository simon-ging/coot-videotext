"""
Pooling modules.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn

from nntrainer import typext
from nntrainer.models.activations import ActivationConfig, make_activation_module
from nntrainer.typext import INF


def make_pooler_module(normalized_shape: Union[int, List[int]], name: str, cfg: Optional[PoolerConfig] = None
                       ) -> nn.Module:
    """
    Get pooler module instance given by name and config object.


    Args:
        normalized_shape:
        name:
        cfg:

    Returns:
        Pooler module instance.
    """
    if cfg is None:
        # set all module hyperparameters to default values
        cfg = PoolerConfig(name)

    # create module instance
    if name == PoolerConst.ATN:
        assert isinstance(normalized_shape, int), f"MultiGenPool only understands int shape but got {normalized_shape}"
        return MultiGenPool(cfg.num_layers, normalized_shape, cfg.hidden_dim, cfg.num_heads, cfg.dropout,
                            cfg.activation.name, activation_cfg=cfg.activation)
    if cfg.name == PoolerConst.AVG_SPECIAL:
        return TemporalAvgPool()
    raise NotImplementedError(f"Pooler {name} unknown.")


class PoolerConst(typext.ConstantHolder):
    """
    Pooler types for coot.

    Notes:
        ATN: Attention-aware feature aggregation
        AVG_SPECIAL: Average-pooling as described in the appendix
    """
    ATN = "atn"
    AVG_SPECIAL = "avg_special"


class PoolerConfig(typext.ConfigClass):
    """
    Pooling Submodule

    Args:
        name_or_config: Either provide string name of or a dict with name and hyperparameters.
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
        # fields required for attention-based pooling
        self.hidden_dim: int = config.pop("hidden_dim", 0)
        self.num_heads: int = config.pop("num_heads", 1)
        self.num_layers: int = config.pop("num_layers", 1)
        self.dropout: float = config.pop("dropout", 0)
        self.activation = ActivationConfig(config.pop("activation", "relu"))


# ---------- Module implementations. ----------

class MultiGenPool(nn.Module):
    """
    Apply multiple pooling layers and concatenate the output.

    Idea is that the model will learn to pool different things and generate
    better embeddings out of the sequence.
    """

    def __init__(
            self, n_pools: int, d_input: int, d_attn: int, n_heads: int, dropout_prob: float, activation_name: str,
            activation_cfg: Optional[ActivationConfig] = None):
        super().__init__()
        pools = []
        for _n in range(n_pools):
            pools.append(
                GenPool(d_input, d_attn, n_heads, dropout_prob, activation_name, activation_cfg=activation_cfg))
        self.pools = nn.ModuleList(pools)

    def forward(self, features, mask, lengths):
        feature_stack = []
        for pool in self.pools:
            features = pool(features, mask, lengths)
            feature_stack.append(features)
        pooled = torch.cat(feature_stack, dim=-1)
        return pooled


class GenPool(nn.Module):
    """
    Generalized pooling from 'Enhancing Sentence Embedding with Generalized Pooling.'
    """

    def __init__(
            self, d_input: int, d_attn: int, n_heads: int, dropout_prob: float, activation_name: str,
            activation_cfg: Optional[ActivationConfig] = None):
        super().__init__()

        if d_attn == 0:
            d_attn = d_input
        # print(f"atn pooler dim {d_attn} in dim {d_input}")
        assert d_attn % n_heads == 0,\
            f"attention pooling dim {d_attn} not divisible by {n_heads} heads"
        self.d_head = d_attn // n_heads
        self.d_head_output = d_input // n_heads
        self.num_heads = n_heads

        w1_head = torch.zeros(n_heads, d_input, self.d_head)
        b1_head = torch.zeros(n_heads, self.d_head)
        w2_head = torch.zeros(n_heads, self.d_head, self.d_head_output)
        b2_head = torch.zeros(n_heads, self.d_head_output)

        self.genpool_w1_head = nn.Parameter(w1_head, requires_grad=True)
        self.genpool_b1_head = nn.Parameter(b1_head, requires_grad=True)
        self.genpool_w2_head = nn.Parameter(w2_head, requires_grad=True)
        self.genpool_b2_head = nn.Parameter(b2_head, requires_grad=True)

        self.activation = make_activation_module(activation_name, activation_cfg)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=2)
        self.softmax_temp = 1

        self.genpool_one = nn.Parameter(torch.ones(1), requires_grad=False)

    def extra_repr(self) -> str:
        strs = []
        for p in [self.genpool_w1_head, self.genpool_b1_head,
                  self.genpool_w2_head, self.genpool_b2_head]:
            strs.append(f"pool linear {p.shape}")
        return "\n".join(strs)

    def forward(self, features: torch.FloatTensor, mask: torch.BoolTensor, _lengths: torch.LongTensor):
        """
        Args:
            features: Input features shape (batch_size, seq_len, feat_dim=
            mask: Input mask shape (batch_size, seq_len)
            _lengths: Input lengths, unused, shape (batch_size)

        Returns:
        """
        # print(f"genpool input {features.shape}")
        _batch_size, seq_len, input_dim = features.shape
        # apply first FCs, one for each head

        # features (batch, seq_len, d_input)
        # weight1 (num_heads, d_input, d_head)
        b1 = torch.matmul(features.unsqueeze(1), self.genpool_w1_head.unsqueeze(0))
        b1 += self.genpool_b1_head.unsqueeze(1).unsqueeze(0)
        # output (batch, num_heads, seq_len, d_head)

        # dropout + activation
        # apply nonlinear activation
        b1 = self.activation(self.dropout1(b1))

        # apply second FCs, one for each head
        # weight2 (num_heads, d_head, d_head_output)
        b1 = torch.matmul(b1, self.genpool_w2_head.unsqueeze(0))
        b1 += self.genpool_b2_head.unsqueeze(1).unsqueeze(0)
        # output (batch, num_heads, seq_len, d_head_output)

        # dropout
        b1 = self.dropout2(b1)

        # set pre-softmax activations for masked sequence elements to -inf
        # mask shape (batch, seq_len)
        b1.masked_fill_(mask.unsqueeze(1).unsqueeze(-1), -INF)

        # now softmax individually per head over the sequence
        smweights = self.softmax(b1 / self.softmax_temp)
        # shape (batch, num_heads, seq_len, d_head_output)

        # drop attentions
        smweights = self.dropout3(smweights)

        # multiply input features with softmax weights for all heads
        smweights = smweights.transpose(1, 2).reshape(
            -1, seq_len, input_dim)
        # shape (batch, seq_len, input_dim)

        # use the attention weights to pool over the sequence and done
        pooled = (features * smweights).sum(dim=1)

        # return
        return pooled


class TemporalMaxPool(nn.Module):
    def forward(self, features, mask, _lengths):
        # features (batch, seq_len, feat_dim)
        # mask (batch, seq_len) - 1 for values, 0 for padding
        # lengths (batch)
        mask_expanded = mask.unsqueeze(-1)
        value = -INF
        try:
            feat_fill = features.masked_fill(mask_expanded, value)
        except RuntimeError as e:
            print(
                f"input features {features.shape} mask {mask.shape} "
                f"inverted mask {mask_expanded.shape} unpacked h_last "
                f"{features.shape}")
            raise e
        result2, _ = torch.max(feat_fill, dim=1)
        # (batch, feat_dim)

        return result2


class TemporalAvgPool(nn.Module):
    def forward(self, features, _mask, lengths):
        # features (batch, seq_len, feat_dim)
        # mask (batch, seq_len) - 1 for values, 0 for padding
        # lengths (batch)
        len_div = lengths.unsqueeze(-1).float()
        result2 = torch.sum(features, dim=1) / len_div
        # output shape (batch, feat_dim * num_dirs)

        return result2


class TemporalAvgPoolFixed(nn.Module):
    def forward(self, features, mask, lengths):
        # features (batch, seq_len, feat_dim)
        # mask (batch, seq_len) - 1 for values, 0 for padding
        # lengths (batch)

        # MASK features
        f2 = features.masked_fill(mask.unsqueeze(-1), 0)
        len_div = lengths.unsqueeze(-1).float()
        result2 = torch.sum(f2, dim=1) / len_div
        # output shape (batch, feat_dim * num_dirs)
        return result2


class TemporalLastPool(nn.Module):
    """
    Use last hidden state (end of sequence) as output
    Bugfix: last unmasked sequence element, not last in the tensor...
    """

    def forward(self, features, _mask, lengths):
        # get end of sequence by checking the sequence lengths
        idx_last = (lengths - 1).unsqueeze(-1).unsqueeze(-1).repeat(
            1, 1, features.shape[-1])
        result2 = torch.gather(features, 1, idx_last)
        return result2


class TemporalFirstPool(nn.Module):
    """
    Use first hidden state (start of sequence) as output
    e.g. CLS token
    """

    def __init__(self, half_pool=False):
        super().__init__()
        self.half_pool = half_pool

    def forward(self, features, _, __):
        # get start of sequence
        result2 = features[:, 0, :]
        if self.half_pool:
            _, feat_dim = result2.shape
            result2 = result2.reshape(-1, 2, feat_dim // 2).mean(dim=1)
        return result2
