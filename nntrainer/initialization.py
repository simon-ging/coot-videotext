"""
Network initialization.
"""
import torch as th
from torch import nn

from nntrainer import utils_torch, typext, utils


def init_weight_(w: th.Tensor, init_type="uniform", init_std=1) -> None:
    """
    Initialize given tensor.

    Args:
        w: Tensor to initialize in-place.
        init_type: Distribution type.
        init_std: Distribution standard deviation.
    """
    if init_type == InitTypesConst.UNIFORM:
        # uniform distribution
        nn.init.xavier_uniform_(w, gain=init_std)
    elif init_type == InitTypesConst.NORM:
        # normal distribution
        nn.init.xavier_normal_(w, gain=init_std)
    elif init_type == InitTypesConst.TRUNCNORM:
        # truncated normal distribution
        utils_torch.fill_tensor_with_truncnorm(w, std=init_std)
    elif init_type == InitTypesConst.NONE:
        # do nothing, keep pytorch default init
        pass
    else:
        raise RuntimeError(f"unknown init method {init_type}")


class InitTypesConst(typext.ConstantHolder):
    """
    Weight Initialization

    Notes:
        NONE: Keep PyTorch default init.
        NORM: Random Normal Distribution (Xavier).
        UNIFORM: Random Uniform Distribution (Xavier).
        TRUNCNORM: Truncated Normal Distribution (Resample values IFF abs(value - mean) > 2 * std_dev)
    """
    NONE = utils.NONE  # leave it to pytorch
    NORM = "normal"  # random normal dist
    UNIFORM = "uniform"  # random uniform dist
    TRUNCNORM = "truncnorm"  # truncated normal dist


def init_network(net: nn.Module, init_type: str, init_std: float, verbose: bool = False) -> None:
    """
    Initialize network.

    Args:
        net: Network.
        init_type: Distribution type.
        init_std: Distribution standard deviation.
        verbose: Enable verbosity for debugging.
    """

    def debug_print(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    if init_type == "none":
        # trust pytorch to have decent initialization by default
        return
    # get all parameters
    for key, val in net.named_parameters():
        if verbose:
            debug_print(key, end=" ")
        if "layer_normalization" in key or "input_norm." in key or "norm_input." in key or "encoder.norm." in key:
            # old layer norm is already initialized with gain=1, bias=0
            # new layer norm i have no idea
            debug_print("skip layernorm")
            continue
        if "cls_token" in key:
            # init CLS vector
            debug_print("init cls")
            init_weight_(val.data, init_type, init_std)
            continue
        if "genpool_" in key:
            if "genpool_one" in key:
                # this parameter should stay at 1
                debug_print("skip genpool one")
                continue
            # init generalized pooling # this?
            debug_print("init genpool")
            init_weight_(val.data, init_type, init_std)
            continue
        if "input_rnn." in key:
            # rnn weights are initialized on creation
            debug_print("skip rnn")
            continue
        if "weight" in key:  # or "bias" in key:
            # FC input / output layers, attention layers, pointwise FF
            debug_print("init weight")
            init_weight_(val.data, init_type, init_std)
            continue
        if "bias" in key:
            # make bias slightly positive to allow flow through activations # is it this?
            # th.fill_(val.data, init_std) # TODO put in config / test
            debug_print("init bias")
            init_weight_(val.data, init_type, init_std)
            continue
        if "input_indices" in key or "fixed_ones" in key:
            # this is a fixed parameter
            debug_print("skip fixed param")
            continue
        raise ValueError(f"no init method for key {key} defined.")
