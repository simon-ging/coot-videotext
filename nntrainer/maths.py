"""
Math utilities.
"""

from typing import Iterable, Union

import numpy as np


# ---------- Math for sequence data. ----------

def compute_indices(
        num_frames_orig: int, num_frames_target: int, is_train: bool) -> np.ndarray:
    """
    Given two sequence lengths n_orig and n_target, sample n_target indices from the range [0, n_orig-1].

    Random sample approximately from intervals during training:
    with factor f = n_orig / n_target, sample in the range [i*f, (i+1)*f].
    Center sample in the same range during validation.

    Args:
        num_frames_orig: Original sequence length n_orig.
        num_frames_target: Target sequence length n_target.
        is_train:

    Returns:
        Indices with shape (n_target)
    """
    # random sampling during training
    if is_train:
        # create rounded start points
        start_points = np.linspace(0, num_frames_orig, num_frames_target, endpoint=False)
        start_points = np_round_half_down(start_points).astype(int)

        # compute random offsets s.t. the sum of offsets equals num_frames_orig
        offsets = start_points[1:] - start_points[:-1]
        np.random.shuffle(offsets)
        last_offset = num_frames_orig - np.sum(offsets)
        offsets = np.concatenate([offsets, np.array([last_offset])])

        # compute new start points as cumulative sum of offsets
        new_start_points = np.cumsum(offsets) - offsets[0]

        # move offsets to the left so they fit the new start points
        offsets = np.roll(offsets, -1)

        # now randomly sample in the uniform intervals given by the offsets
        random_offsets = offsets * np.random.rand(num_frames_target)

        # calculate indices and floor them to get ints
        indices = new_start_points + random_offsets
        indices = np.floor(indices).astype(int)
        return indices
    # center sampling during validation
    # compute the linspace and offset it so its centered
    start_points = np.linspace(0, num_frames_orig, num_frames_target, endpoint=False)
    offset = num_frames_orig / num_frames_target / 2
    indices = start_points + offset
    # floor the result to get ints
    indices = np.floor(indices).astype(int)
    return indices


def expand_video_segment(num_frames_video: int, min_frames_seg: int, start_frame_seg: int, stop_frame_seg: int):
    """
    Expand a given video segment defined by start and stop frame to have at least a minimum number of frames.

    Args:
        num_frames_video: Total number of frames in the video.
        min_frames_seg: Target minimum number of frames in the segment.
        start_frame_seg: Current start frame of the segment.
        stop_frame_seg: Current stop frame of the segment.

    Returns:
        Tuple of start frame, stop frame, flag whether the segment was changed.
    """
    num_frames_seg = stop_frame_seg - start_frame_seg
    changes = False
    if min_frames_seg > num_frames_video:
        min_frames_seg = num_frames_video
    if num_frames_seg < min_frames_seg:
        while True:
            if start_frame_seg > 0:
                start_frame_seg -= 1
                num_frames_seg += 1
                changes = True
            if num_frames_seg == min_frames_seg:
                break
            if stop_frame_seg < num_frames_video:
                stop_frame_seg += 1
                num_frames_seg += 1
                changes = True
            if num_frames_seg == min_frames_seg:
                break
    return start_frame_seg, stop_frame_seg, changes


# ---------- Convenience functions for numpy. ----------

def rnd(x: Union[int, float]) -> int:
    """
    Convenience function to round a number and get an int back.
    Bankers rounding is used, i.e. round half numbers to the next even number.

    Args:
        x: Input number.

    Returns:
        Rounded number.
    """
    return int(np.round(x).astype(int))


def floor(x: Union[int, float]) -> int:
    """
    Convenience function to floor a number and get an int back.

        Args:
            x: Input number.

        Returns:
            Floored number.
        """
    return int(np.floor(x).astype(int))


def ceil(x):
    """
    Convenience function to ceil a number and get an int back.

        Args:
            x: Input number.

        Returns:
            Floored number.
        """
    return int(np.ceil(x).astype(int))


def np_round_half_down(array: Union[np.ndarray, Iterable[Union[int, float]]]):
    """
    Numpy round function that rounds half numbers down.

    Args:
        array: Input number array with arbitrary shape.

    Returns:
        Rounded array with same shape as input.

    Notes:
        Default np.round rounds half numbers to the next even number, so called "bankers rounding"
        i.e. (0.5, 1.5, 2.5, 3.5, ...) to (0, 2, 2, 4, 4, ...).
        This function rounds half numbers always down instead which is better for sampling frames
        i.e. (0.5, 1.5, 2.5, 3.5, ...) to (0, 1, 2, 3, 4, ...).
    """
    if not isinstance(array, np.ndarray):
        # also support iterables
        array = np.array(array)
    return np.ceil(array - 0.5)


def np_str_len(str_arr: Union[np.ndarray, Iterable[str]]) -> np.ndarray:
    """
    Fast way to get string length in a numpy array with datatype string.

    Args:
        str_arr: Numpy array of strings with arbitrary shape.

    Returns:
        Numpy array of string lengths, same shape as input.

    Notes:
        Source: https://stackoverflow.com/questions/44587746/length-of-each-string-in-a-numpy-array
        The latest improved answers don't really work. This code should work for all except strange special characters.
    """
    if not isinstance(str_arr, np.ndarray):
        # also support iterables of strings
        str_arr = np.array(str_arr)
    # check input type
    if str(str_arr.dtype)[:2] != "<U":
        raise TypeError(
            f"Computing string length of dtype {str_arr.dtype} will not work correctly. Cast array to string first.")

    # see the link in the docstring as an explanation of what exactly is happening here
    try:
        v = str_arr.view(np.uint32).reshape(str_arr.size, -1)
    except TypeError as e:
        print(f"Input {str_arr} shape {str_arr.shape} dtype {str_arr.dtype}")
        raise e
    len_arr = np.argmin(v, 1)
    len_arr[v[np.arange(len(v)), len_arr] > 0] = v.shape[-1]
    len_arr = np.reshape(len_arr, str_arr.shape)
    return len_arr
