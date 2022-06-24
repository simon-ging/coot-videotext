"""
Arguments for training MART.
"""
import argparse
from typing import Any, Dict


def add_mart_args(parser: argparse.ArgumentParser) -> None:
    """
    Add some additional arguments that are required for mart.

    Args:
        parser: Command line argument parser.
    """
    # paths
    parser.add_argument("--cache_dir", type=str, default="cache_caption", help="Cached vocabulary dir.")
    parser.add_argument("--coot_feat_dir", type=str, default="provided_embeddings", help="COOT Embeddings dir.")
    parser.add_argument("--annotations_dir", type=str, default="annotations", help="Annotations dir.")
    parser.add_argument("--video_feature_dir", type=str, default="data/mart_video_feature",
                        help="Dir containing the video features")

    # Technical
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_preload", action="store_true")
    parser.add_argument("--dataset_max", type=int, default=None, help="Reduce dataset size for testing.")


def update_mart_config_from_args(
        config: Dict, args: argparse.Namespace, *, verbose: bool = True) -> Dict[str, Any]:
    """
    Modify config given script arguments.

    Args:
        config: Config dictionary.
        args: Arguments.
        verbose: Print message when updating the config.

    Returns:
        Updated config dict.
    """
    if args.debug:
        config["debug"] = True
        if verbose:
            print(f"    Change config: Set debug to True")
    if args.dataset_max is not None:
        assert args.dataset_max > 0, "--dataset_max must be positive int."
        config["dataset_train"]["max_datapoints"] = args.dataset_max
        config["dataset_val"]["max_datapoints"] = args.dataset_max
        if verbose:
            print(f"    Change config: Set dataset_(train|val).max_datapoints to {args.dataset_max}")
    if args.preload:
        config["dataset_train"]["preload"] = True
        config["dataset_val"]["preload"] = True
        if verbose:
            print(f"    Change config: Set dataset_(train|val).preload to True")
    if args.no_preload or args.validate:
        config["dataset_train"]["preload"] = False
        config["dataset_val"]["preload"] = False
        if verbose:
            print(f"    Change config: Set dataset_(train|val).preload to False (--no_preload or --validate)")
    return config
