"""
Utility code for doing retrieval.
"""

from timeit import default_timer as timer
from typing import Callable, Dict, Tuple

import numpy as np
import torch as th


VALKEYS = ["r1", "r5", "r10", "r50", "medr", "meanr", "sum"]
VALHEADER = "Retriev | R@1   | R@5   | R@10  | R@50  | MeanR |  MedR |    Sum"


def retrieval_results_to_str(results: Dict[str, float], name: str) -> str:
    """
    Convert single dictionary of retrieval results to string.

    Args:
        results: Results dictionary.
        name: Type of retrieval to print.

    Returns:
        String results.
    """
    return ("{:7s} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:5.1f} | "
            "{:5.1f} | {:6.3f}").format(name, *[results[key] for key in VALKEYS])


def compute_retrieval(data_collector: Dict[str, th.Tensor], key1: str, key2: str, print_fn: Callable = print) -> (
        Tuple[Dict[str, float], Dict[str, float], float, str]):
    """
    Get embeddings from data collector given by keys, compute retrieval and return results.

    Args:
        data_collector: Collected validation data (output embeddings of the model).
        key1: Name of source embedding.
        key2: Name of target embedding.
        print_fn: Function to print the results with.

    Returns:
        Tuple of:
            Metrics for retrieval from key1 to key2.
            Metrics for retrieval from key2 to key1.
            Sum of R@1 metrics.
            Additional info string to print later (number of datapoints, time performance).
    """
    start_time = timer()
    emb1 = data_collector[key1]
    emb2 = data_collector[key2]
    if isinstance(emb1, th.Tensor):
        emb1 = emb1.numpy()
    if isinstance(emb2, th.Tensor):
        emb2 = emb2.numpy()

    d = np.dot(emb1, emb2.T)
    num_points = len(d)
    res1, _, _ = compute_retrieval_cosine(d)
    res2, _, _ = compute_retrieval_cosine(d.T)
    sum_at_1 = (res1["r1"] + res2["r1"]) / 2
    print_fn(retrieval_results_to_str(res1, key1[:3]))
    print_fn(retrieval_results_to_str(res2, key2[:3]))
    result_str = f"{key1[:3]}{key2[:3]} ({num_points}) in {timer() - start_time:.3f}s, "
    return res1, res2, sum_at_1, result_str


def compute_retrieval_cosine(dot_product: np.ndarray) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Args:
        dot_product: Result of computing cosine similarity between two sets of embeddings (emb1 @ emb2.T)
            with shape (num_datapoints, num_datapoints).

    Returns:
        Retrieval metrics for that similarity.
    """
    len_dot_product = len(dot_product)
    ranks = np.empty(len_dot_product)
    top1 = np.empty(len_dot_product)
    # loop source embedding indices
    for index in range(len_dot_product):
        # get order of similarities to target embeddings
        inds = np.argsort(dot_product[index])[::-1]
        # find where the correct embedding is ranked
        where = np.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank
        # save the top1 result as well
        top1[index] = inds[0]
    # compute retrieval metrics
    r1 = len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    report_dict = {"r1": r1, "r5": r5, "r10": r10, "r50": r50, "medr": medr, "meanr": meanr, "sum": r1 + r5 + r50}
    return report_dict, top1, ranks
