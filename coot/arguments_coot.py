"""
Modular arguments specific to COOT training.
"""
import argparse
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple


def add_dataloader_args(parser: ArgumentParser) -> None:
    """
    Add flags for the dataloader (preloading).

    Args:
        parser: Command line argument parser.
    """
    parser.add_argument("--preload", action="store_true", help="Preload everything.")
    parser.add_argument("--preload_vid", action="store_true", help="Preload visual features.")
    parser.add_argument("--preload_text", action="store_true", help="Preload text features.")
    parser.add_argument("--no_preload", action="store_true", help="Don't preload anything.")
    parser.add_argument("--no_preload_vid", action="store_true", help="Don't preload visual features.")
    parser.add_argument("--no_preload_text", action="store_true", help="Don't preload text features.")


def update_coot_config_from_args(
        config: Dict, args: argparse.Namespace, *, verbose: bool = True) -> Tuple[Dict, Path]:
    """
    Modify config and paths given script arguments.

    Here, independent of the value in the config it's easy to set preloading depending on the user needs.

    Args:
        config: Config dictionary.
        args: Arguments.
        verbose: Print message when updating the config.

    Returns:
        Tuple of:
            Updated config dict.
            Dataset base path.
    """
    if args.preload_vid or args.preload:
        if verbose:
            print(f"    Change config: Preload video features")
        config["dataset_train"]["preload_vid_feat"] = True
        config["dataset_val"]["preload_vid_feat"] = True
    if args.no_preload_vid or args.no_preload:
        if verbose:
            print(f"    Change config: Don't preload video features")
        config["dataset_train"]["preload_vid_feat"] = False
        config["dataset_val"]["preload_vid_feat"] = False
    if args.preload_text or args.preload:
        if verbose:
            print(f"    Change config: Preload text features")
        config["dataset_train"]["preload_text_feat"] = True
        config["dataset_val"]["preload_text_feat"] = True
    if args.no_preload_text or args.no_preload:
        if verbose:
            print(f"    Change config: Don't preload text features")
        config["dataset_train"]["preload_text_feat"] = False
        config["dataset_val"]["preload_text_feat"] = False
    return config
