"""
Utilities for randomness.
"""
import ctypes
import multiprocessing
import os
import random
from typing import Any, Dict, Tuple, Mapping, Sequence, List

import GPUtil
import numpy as np
import psutil
import torch as th
import torch.backends.cudnn as cudnn
from torch import cuda


# ---------- Multiprocessing ----------

MAP_TYPES: Dict[str, Any] = {
        'int': ctypes.c_int,
        'long': ctypes.c_long,
        'float': ctypes.c_float,
        'double': ctypes.c_double
}


def create_shared_array(arr: np.ndarray, dtype: str = "float") -> np.array:
    """
    Converts an existing numpy array into a shared numpy array, such that
    this array can be used by multiple CPUs. Used e.g. for preloading the
    entire feature dataset into memory and then making it available to multiple
    dataloaders.

    Args:
        arr (np.ndarray): Array to be converted to shared array
        dtype (np.dtype): Datatype of shared array

    Returns:
        shared_array (multiprocessing.Array): shared array
    """
    shape = arr.shape
    flat_shape = int(np.prod(np.array(shape)))
    c_type = MAP_TYPES[dtype]
    shared_array_base = multiprocessing.Array(c_type, flat_shape)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(shape)
    shared_array[:] = arr[:]
    return shared_array


# ---------- Random ----------

def set_seed(seed: int, cudnn_deterministic: bool = False, cudnn_benchmark: bool = True):
    """
    Set all relevant seeds for torch, numpy and python

    Args:
        seed: int seed
        cudnn_deterministic: set True for deterministic training..
        cudnn_benchmark: set False for deterministic training.
    """
    th.manual_seed(seed)
    cuda.manual_seed(seed)
    cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cudnn_deterministic:
        cudnn.deterministic = True
    cudnn.benchmark = cudnn_benchmark


def get_truncnorm_tensor(shape: Tuple[int], *, mean: float = 0, std: float = 1,
                         limit: float = 2) -> th.Tensor:
    """
    Create and return normally distributed tensor, except values with too much deviation are discarded.

    Args:
        shape: tensor shape
        mean: normal mean
        std: normal std
        limit: which values to discard

    Returns:
        Filled tensor with shape (*shape)
    """
    assert isinstance(shape, (tuple, list)), f"shape {shape} is not a tuple or list of ints"
    num_examples = 8
    tmp = th.empty(shape + (num_examples,)).normal_()
    valid = (tmp < limit) & (tmp > -limit)
    _, ind = valid.max(-1, keepdim=True)
    return tmp.gather(-1, ind).squeeze(-1).mul_(std).add_(mean)


def fill_tensor_with_truncnorm(input_tensor: th.Tensor, *, mean: float = 0, std: float = 1,
                               limit: float = 2) -> None:
    """
    Fill given input tensor with a truncated normal dist.

    Args:
        input_tensor: tensor to be filled
        mean: normal mean
        std: normal std
        limit: which values to discard
    """
    # get truncnorm values
    tmp = get_truncnorm_tensor(input_tensor.shape, mean=mean, std=std, limit=limit)
    # fill input tensor
    input_tensor[...] = tmp[...]


# ---------- Profiling ----------

def profile_gpu_and_ram(
) -> Tuple[List[str], List[float], List[float], List[float], float, float, float]:
    """
    Profile GPU and RAM.

    Returns:
        GPU names, total / used memory per GPU, load per GPU, total / used / available RAM.
    """

    # get info from gputil
    _str, dct_ = _get_gputil_info()
    dev_num = os.getenv("CUDA_VISIBLE_DEVICES")
    if dev_num is not None:
        # single GPU set with OS flag
        gpu_info = [dct_[int(dev_num)]]
    else:
        # possibly multiple gpus, aggregate values
        gpu_info = []
        for dev_dict in dct_:
            gpu_info.append(dev_dict)

    # convert to GPU info and MB to GB
    gpu_names: List[str] = [gpu["name"] for gpu in gpu_info]
    total_memory_per: List[float] = [gpu["memoryTotal"] / 1024 for gpu in gpu_info]
    used_memory_per: List[float] = [gpu["memoryUsed"] / 1024 for gpu in gpu_info]
    load_per: List[float] = [gpu["load"] / 100 for gpu in gpu_info]

    # get RAM info and convert to GB
    mem = psutil.virtual_memory()
    ram_total: float = mem.total / 1024 ** 3
    ram_used: float = mem.used / 1024 ** 3
    ram_avail: float = mem.available / 1024 ** 3

    return gpu_names, total_memory_per, used_memory_per, load_per, ram_total, ram_used, ram_avail


def _get_gputil_info():
    """
    Returns info string for printing and list with gpu infos. Better formatting than the original GPUtil.

    Returns:
        gpu info string, List[Dict()] of values. dict example:
            ('id', 1),
            ('name', 'GeForce GTX TITAN X'),
            ('temperature', 41.0),
            ('load', 0.0),
            ('memoryUtil', 0.10645266950540452),
            ('memoryTotal', 12212.0)])]
    """

    gpus = GPUtil.getGPUs()
    attr_list = [
            {'attr': 'id', 'name': 'ID'}, {'attr': 'name', 'name': 'Name'},
            {'attr': 'temperature', 'name': 'Temp', 'suffix': 'C', 'transform': lambda
                x: x, 'precision': 0},
            {'attr': 'load', 'name': 'GPU util.', 'suffix': '% GPU', 'transform': lambda x: x * 100,
             'precision': 1},
            {'attr': 'memoryUtil', 'name': 'Memory util.', 'suffix': '% MEM', 'transform': lambda
                x: x * 100,
             'precision': 1},
            {'attr': 'memoryTotal', 'name': 'Memory total', 'suffix': 'MB', 'precision': 0},
            {'attr': 'memoryUsed', 'name': 'Memory used', 'suffix': 'MB', 'precision': 0}
    ]
    gpu_strings = [''] * len(gpus)
    gpu_info = []
    for _ in range(len(gpus)):
        gpu_info.append({})

    for attrDict in attr_list:
        attr_precision = '.' + str(attrDict['precision']) if (
                'precision' in attrDict.keys()) else ''
        attr_suffix = str(attrDict['suffix']) if (
                'suffix' in attrDict.keys()) else ''
        attr_transform = attrDict['transform'] if (
                'transform' in attrDict.keys()) else lambda x: x
        for gpu in gpus:
            attr = getattr(gpu, attrDict['attr'])

            attr = attr_transform(attr)

            if isinstance(attr, float):
                attr_str = ('{0:' + attr_precision + 'f}').format(attr)
            elif isinstance(attr, int):
                attr_str = '{0:d}'.format(attr)
            elif isinstance(attr, str):
                attr_str = attr
            else:
                raise TypeError('Unhandled object type (' + str(
                        type(attr)) + ') for attribute \'' + attrDict[
                                    'name'] + '\'')

            attr_str += attr_suffix

        for gpuIdx, gpu in enumerate(gpus):
            attr_name = attrDict['attr']
            attr = getattr(gpu, attr_name)

            attr = attr_transform(attr)

            if isinstance(attr, float):
                attr_str = ('{0:' + attr_precision + 'f}').format(attr)
            elif isinstance(attr, int):
                attr_str = ('{0:' + 'd}').format(attr)
            elif isinstance(attr, str):
                attr_str = ('{0:' + 's}').format(attr)
            else:
                raise TypeError(
                        'Unhandled object type (' + str(
                                type(attr)) + ') for attribute \'' + attrDict[
                            'name'] + '\'')
            attr_str += attr_suffix
            gpu_info[gpuIdx][attr_name] = attr
            gpu_strings[gpuIdx] += '| ' + attr_str + ' '

    return "\n".join(gpu_strings), gpu_info


# ---------- Modeling ----------

def count_parameters(model, verbose=True):
    """
    Count number of parameters in PyTorch model,
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if verbose:
        print(f"Parameters total: {n_all}, frozen: {n_frozen}")
    return n_all, n_frozen


def edit_moduledot_in_state_keys(state, remove: bool = True):
    """
    When using a model wrapped with nn.DataParallel or nn.DistributedDataParallel,
    the state dict keys must additionally start with "module.", otherwise they
    must not start with "module.". This function takes care to correct the keys,
    such that a model saved with DataParallel can be loaded with a normal model,
    and vice versa.

    Works with nested states e.g. lists or dicts of several model state dicts.

    Args:
        state: state dict or nested state dict
        remove: if True, remove module.dot from state dict, else add it.

    Returns:
        state: state dict without module.dot
    """
    if isinstance(state, Sequence):
        # handle lists
        return [edit_moduledot_in_state_keys(s, remove=remove) for s in state]
    elif isinstance(state, Mapping):
        # handle dicts
        new_state = {}
        for key, value in state.items():
            if key.startswith("module.") and remove:
                key = key[7:]
            elif not key.startswith("module.") and not remove:
                key = f"module.{key}"
            if isinstance(value, Mapping) or isinstance(value, Sequence):
                # handle nesting
                value = edit_moduledot_in_state_keys(value, remove=remove)
            new_state[key] = value
        return new_state
    else:
        raise ValueError(f"Unexpected type {type(state)}")
