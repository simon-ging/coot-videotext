"""
Tests PyTorch specific utilites.
"""
import random
from pprint import pprint

import numpy as np
import torch as th
from torch import nn

from nntrainer.utils_torch import (
    create_shared_array, fill_tensor_with_truncnorm, get_truncnorm_tensor, profile_gpu_and_ram, set_seed)


def test_create_shared_array() -> None:
    """
    Create shared array and make sure it's consistent.
    """
    arr = np.random.rand(5, 20, 7).astype(np.float32)
    shared_arr = create_shared_array(arr)
    assert shared_arr.shape == arr.shape
    assert np.array_equal(arr, shared_arr)


def test_profile_gpu_and_ram():
    pprint(profile_gpu_and_ram())


def test_set_seed() -> None:
    """
    Set seed and create some random ints in torch, numpy, python. Check if
    the random functions are deterministic.
    """
    set_seed(420)
    rand_int1 = th.randint(42, (1,)).squeeze()
    assert rand_int1 == 29, rand_int1

    set_seed(69)
    rand_int2 = np.random.randint(1337, size=(1,))
    assert rand_int2.squeeze() == 1078, rand_int2
    del rand_int2

    set_seed(300)
    rand_int3 = random.randint(0, 1001)
    assert rand_int3 == 612, rand_int3
    del rand_int3


def test_truncated_normal_fill():
    """
    Random truncated normal fill a tensor and make sure values are not out of bound.
    """
    t = get_truncnorm_tensor((12, 6, 512), mean=0, std=1, limit=2)
    th.testing.assert_allclose(t.abs() <= 2, np.ones_like(t), msg="Tensor out of bounds")


def test_get_truncnorm():
    std = 0.01
    limit = 2

    # create a large truncated normal tensor
    tn = get_truncnorm_tensor((10000,), mean=0, std=std, limit=limit)
    assert th.all(th.abs(tn) <= limit * std)

    # create a network and init it's weight
    net = nn.Linear(2, 2)
    for param in net.parameters():
        fill_tensor_with_truncnorm(param.data, std=std, limit=limit)
    for param in net.parameters():
        assert th.all(param.data < std * limit)


if __name__ == "__main__":
    test_profile_gpu_and_ram()
    test_create_shared_array()

    test_set_seed()
    test_truncated_normal_fill()
    test_get_truncnorm()
