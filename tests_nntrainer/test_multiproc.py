"""
Test multiprocessing.
"""

import gc

import numpy as np

from nntrainer.utils_torch import create_shared_array


# pylint: disable=unused-variable
def print_garbage_info() -> None:
    print(f"Garbage collector: {gc.get_count()}")


def test_many_shared_arrays() -> None:
    """
    Create shared array and make sure it's consistent.
    """
    runs = 10
    num = 1000
    print_garbage_info()
    for k in range(runs):
        shared_arrs = []
        for n in range(num):
            arr = np.random.rand(5, 20, 7).astype(np.float32)
            shared_arr = create_shared_array(arr)
            assert shared_arr.shape == arr.shape
            assert np.array_equal(arr, shared_arr)
            shared_arrs.append(arr)
        del shared_arrs
        gc.collect()
        print_garbage_info()


if __name__ == "__main__":
    test_many_shared_arrays()
