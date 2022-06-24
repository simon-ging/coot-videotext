"""
Test math utilities.
"""
import numpy as np
import pytest

from nntrainer.maths import ceil, compute_indices, floor, np_round_half_down, np_str_len, rnd


def test_maths() -> None:
    """
    Test maths.
    """
    # test numpy string length
    input_list = [["Lorem", "Ipsum"], ["Dolor", "Sit Amet"]]
    input_arr = np.array(input_list)
    assert np.all(np_str_len(input_list) == np.array([[5, 5], [5, 8]]))
    assert np.all(np_str_len(input_arr) == np.array([[5, 5], [5, 8]]))
    with pytest.raises(TypeError):
        np_str_len(77)

    # test rounding for bankers round
    assert rnd(.7) == 1
    assert rnd(1.5) == 2
    assert rnd(2.5) == 2

    # test floor and ceil
    assert floor(.5) == 0
    assert ceil(.5) == 1

    # test rounding halfs down
    assert np.all(np_round_half_down([0, 0.7, 0.5, 1.5]) == [0, 1, 0, 1])

    # test compute_indices
    assert np.all(compute_indices(5, 10, is_train=False) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    assert np.all(compute_indices(8, 6, is_train=False) == [0, 2, 3, 4, 6, 7])
    np.random.seed(0)
    assert np.all(compute_indices(80, 6, is_train=True) == [7, 20, 32, 49, 59, 78])


if __name__ == "__main__":
    test_maths()
