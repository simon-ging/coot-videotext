"""
Modular arguments for scripts and utilities to parse some of those arguments.
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import repo_config
from nntrainer.utils import TrainerPathConst


GITLIKE_SUPPORT = "Supports .gitignore-like patterns, separated by comma."
GITLIKE_SUPPORT_FILE = "Supports .gitignore-like patterns, one per line."


def add_exp_group_arg(parser: argparse.ArgumentParser) -> None:
    """
    Add experiment directory argument.

    Args:
        parser: Command line argument parser.
    """
    parser.add_argument("-c", "--config_file", type=str, default=None,
                        help="Specify either config file location or experiment group and name.")
    parser.add_argument("-g", "--exp_group", type=str, default="default",
                        help="Experiment group. Path to config: config/$TYPE/$GROUP/$NAME.yaml")


def add_exp_identifier_args(parser: argparse.ArgumentParser) -> None:
    """
    Add full single experiment run identification arguments (dir, name, run name).

    Args:
        parser: Command line argument parser.
    """
    add_exp_group_arg(parser)
    parser.add_argument("-e", "--exp_name", type=str, default="default",
                        help="Experiment name. Path to config: config/$TYPE/$GROUP/$NAME.yaml")
    _add_run_args(parser)


def add_trainer_args(parser: argparse.ArgumentParser, *, dataset_path: bool = True,
                     profiling_path: bool = False) -> None:
    """
    Add various arguments for experiment running.

    Args:
        parser: Command line argument parser.
        dataset_path: Whether to add dataset path argument.
        profiling_path: Whether to add profiling path argument.
    """
    # configuration loading
    parser.add_argument("-o", "--config", type=str, default=None,
                        help="Modify the loaded YAML config. E.g. to change the number of dataloader workers "
                             "and the batchsize, use '-c dataloader.num_workers=20;train.batch_size=32'")
    parser.add_argument("--print_config", action="store_true", help="Print the experiment config.")
    # num workers
    parser.add_argument("--workers", type=int, default=None, help="Shortcut for setting dataloader workers.")
    # dataset path
    add_path_args(parser, dataset_path=dataset_path, profiling_path=profiling_path)
    # checkpoint loading
    parser.add_argument("--load_epoch", type=int, default=None, help="Load epoch number.")
    parser.add_argument("--load_best", action="store_true", help="Load best epoch.")
    # validation
    parser.add_argument("--validate", action="store_true", help="Validation only.")
    parser.add_argument("--ignore_untrained", action="store_true", help="Validate even if no checkpoint was loaded.")
    # reset (delete everything)
    parser.add_argument("--reset", action="store_true", help="Delete experiment.")
    parser.add_argument("--print_graph", action="store_true", help="Print model and forward pass, then exit.")
    parser.add_argument("--seed", type=str, default=None,
                        help="Set seed. integer or none/null for auto-generated seed.")
    _add_gpu_args(parser)


def add_dataset_path_arg(parser: argparse.ArgumentParser) -> None:
    """
    Add dataset path argument.

    Args:
        parser: Command line argument parser.
    """
    parser.add_argument("--data_path", type=str, default=None, help="Change the data path.")


def add_default_args(parser) -> None:
    """
    Add some common script args: logging options -q, -v and test flag -t

    Args:
        parser: Command line argument parser.
    """
    group = parser.add_mutually_exclusive_group()
    group.set_defaults(log_level=logging.INFO)
    group.add_argument(
        "-v", "--verbose", help="Verbose (debug) logging",
        action="store_const", const=logging.DEBUG, dest="log_level")
    group.add_argument(
        "-q", "--quiet", help="Silent mode, only log warnings",
        action="store_const", const=logging.WARN, dest="log_level")
    group.add_argument(
        "--log", help="Set log level manually", type=str, dest="log_level")


def add_test_arg(parser) -> None:
    """
    Test argument -t

    Args:
        parser: Command line argument parser.
    """
    parser.add_argument("-t", "--test", action="store_true", help="test only (no-op)")


def add_path_args(parser: argparse.ArgumentParser, *, dataset_path: bool = True, profiling_path: bool = False) -> None:
    """
    Add all arguments for customizing the paths.

    Args:
        parser: Command line argument parser.
        dataset_path: Whether to add dataset path argument.
        profiling_path: Whether to add profiling path argument.
    """
    parser.add_argument("--config_dir", type=str, default=TrainerPathConst.DIR_CONFIG, help="Folder with config files.")
    parser.add_argument("--log_dir", type=str, default=TrainerPathConst.DIR_EXPERIMENTS,
                        help="Folder with experiment results.")
    if dataset_path:
        add_dataset_path_arg(parser)
    if profiling_path:
        parser.add_argument("--profiling_dir", type=str, default=TrainerPathConst.DIR_PROFILING,
                            help="Profiling output.")


def add_dataset_test_arg(parser: argparse.ArgumentParser) -> None:
    """
    Add flag for testing the dataset.

    Args:
        parser: Command line argument parser.
    """
    parser.add_argument("--test_dataset", action="store_true", help="Test dataset and exit.")


def add_multi_experiment_args(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for working on multiple experiment groups.

    Args:
        parser: Command line argument parser.
    """
    parser.add_argument("-g", "--exp_group", type=str, default=None,
                        help=f"Search experiment group and name. {GITLIKE_SUPPORT}")
    parser.add_argument("-s", "--search", type=str, default=None,
                        help=f"Search experiments name only. {GITLIKE_SUPPORT} ")
    parser.add_argument("-l", "--exp_list", type=str, default=None,
                        help=f"Search experiment group and name given by the list in the file. {GITLIKE_SUPPORT_FILE}")


def add_show_args(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for printing results.

    Args:
        parser: Command line argument parser.
    """
    parser.add_argument("--last", action="store_true", help="View last epoch results instead of best.")
    parser.add_argument("--mean", action="store_true", help="Average over multiple runs with same run name.")
    parser.add_argument("--add_group", action="store_true", help="Add experiment group to identifier.")
    parser.add_argument("--mean_all", action="store_true",
                        help="Average over all runs of a single experiment, regardless of run names.")
    parser.add_argument("--sort_asc", action="store_true", help="Sort ascending instead of descending.")
    parser.add_argument("--sort", type=str, default="score", help="Define sorting field, alpha for alphabetic.")
    parser.add_argument("--compact", action="store_true", help="Compact the printed table.")
    parser.add_argument("-m", "--metrics", type=str, default="",
                        help="Which fields (columns) to print. 'all' for all metrics or a comma "
                             "separated list of groups.")
    parser.add_argument("--less_metrics", action="store_true", help="Hide the basic metrics like loss, score, ...")
    parser.add_argument("-f", "--fields", type=str, default="",
                        help="Define field or comma separated list of fields to print.")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """
    Add run number argument.

    Args:
        parser: Command line argument parser.
    """
    parser.add_argument("-n", "--num_runs", type=int, default=1, help="How many runs to do.")
    parser.add_argument("-a", "--start_run", type=int, default=1, help="Start at which run number.")
    parser.add_argument("-r", "--run_name", type=str, default="run",
                        help="Run name to save the model. Must not contain underscores.")


def _add_gpu_args(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for gpu settings.

    Args:
        parser: Command line argument parser.
    """
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA.")
    parser.add_argument("--single_gpu", action="store_true", help="Disable multi GPU with nn.DataParallel.")


# ---------- Utilities ----------

def determine_multi_runs(exp_type: str, exp_group: str = "", exp_list: Optional[List[str]] = None, *,
                         config_dir=TrainerPathConst.DIR_CONFIG):
    """
    Given the experiment group and experiment list file, determine the experiments to run.


    Notes:
        If the list is empty, read all experiments from the folder determined by exp_group.
        If the list is not empty, each line can contain either:
            Only the experiment name, then the experiment: exp_group/exp_name will be loaded.
            Both experiment group and name then the list determines both.

    Args:
        exp_type: Experiment type.
        exp_group: Experiment group.
        exp_list: Experiment list.
        config_dir: Experiment configurations base directory.

    Returns:
        List of tuples with experiment group, experiment name.
    """
    if exp_list is None:
        # no list given, read experiments from the group
        assert exp_group != "", ("Script argument error. Either give a group of experiments to run with --exp_group "
                                 "or a list with --list_file")

        # get list of all names in the group
        exp_names = sorted(os.listdir(Path(config_dir) / exp_type / exp_group))
        # parse files
        output_tuple = []
        for exp_name in exp_names:
            # search for yaml files and remove the file type
            if not exp_name.endswith(".yaml"):
                continue
            exp_name = exp_name[:-5]
            output_tuple.append((exp_group, exp_name))
        return output_tuple
    # read the list
    output_tuple = []
    for line in exp_list:
        slash_split = line.split("/")
        if len(slash_split) == 1:
            # only the experiment name is given
            exp_name = slash_split[0]
            assert exp_group != "", ("Script argument error. No experiment group with --exp_group was given, "
                                     "so all lines in the --list_file must look like this: exp_group/exp_name. "
                                     f"This line is {line}")
            output_tuple.append((exp_group, exp_name))
        elif len(slash_split) == 2:
            # experiment group and name are given
            exp_group, exp_name = slash_split
            output_tuple.append((exp_group, exp_name))
        else:
            raise ValueError(f"Can't understand line {line} in the list given by --list_file, too many slashes.")
    return output_tuple


def update_config_from_args(config: Dict, args: argparse.Namespace, *, verbose: bool = True) -> Dict[str, Any]:
    """
    Modify config and paths given script arguments.

    Args:
        config: Config dictionary.
        args: Arguments.
        verbose: Print message when updating the config.

    Returns:
        Updated config dict.
    """
    # parse the --config inline modifier
    if args.config is not None:
        # get all fields to update from the argument and loop them
        update_fields: List[str] = args.config.split(",")
        for field_value in update_fields:
            # get field and value
            fields_str, value = field_value.strip().split("=")
            # convert value if necessary
            try:
                value = float(value)
                if round(value) == value:
                    value = int(value)
            except ValueError:
                pass
            if str(value).lower() == "true":
                value = True
            elif str(value).lower() == "false":
                value = False
            # update the correct nested dictionary field
            fields = fields_str.split(".")
            current_dict = config
            for i, field in enumerate(fields):
                if i == len(fields) - 1:
                    # update field
                    if field not in current_dict:
                        assert "same_as" in current_dict, (
                            f"Field {fields_str} not found in config {list(current_dict.keys())}. "
                            f"Typo or field missing in config.")
                    current_dict[field] = value
                    if verbose:
                        print(f"    Change config: Set {fields_str} = {value}")
                    break
                # go one nesting level deeper
                current_dict = current_dict[field]

    if args.workers is not None:
        config["dataset_train"]["num_workers"] = int(args.workers)
        config["dataset_val"]["num_workers"] = int(args.workers)
        if verbose:
            print(f"    Change config: Set dataloader workers to {args.workers} for train and val.")

    if args.seed is not None:
        if str(args.seed).lower() in ["none", "null"]:
            config["random_seed"] = None
        else:
            config["random_seed"] = int(args.seed)
        if verbose:
            print(f"    Change config: Set seed to {args.seed}. Deterministic")

    if args.no_cuda:
        config["use_cuda"] = False
        if verbose:
            print(f"    Change config: Set use_cuda to False.")

    if args.single_gpu:
        config["use_multi_gpu"] = False
        if verbose:
            print(f"    Change config: Set use_multi_gpu to False.")

    return config


def update_path_from_args(args: argparse.Namespace) -> Path:
    """
    Either set path from args or from default.

    Args:
        args:

    Returns:
        Root path to data.
    """
    path_data = args.data_path if args.data_path is not None else repo_config.DATA_PATH
    return Path(path_data)


def setup_experiment_identifier_from_args(args: argparse.Namespace, exp_type: str) -> Tuple[str, str, str]:
    """
    Determine the experiment identifier (Group, name, config file) either from group and name or from config file path.

    Args:
        args: Arguments.
        exp_type: Experiment type.

    Returns:
        Tuple of group, name, config file.
    """
    if args.config_file is None:
        # no path to config file given, determine from experiment identifier
        exp_group = args.exp_group
        exp_name = args.exp_name
        config_file = setup_config_file_from_experiment_identifier(
            exp_type, exp_group, exp_name, config_dir=args.config_dir)
    else:
        # path to config file given, determine experiment name from config file name (without file type)
        exp_group = args.exp_group
        exp_name = ".".join(str(Path(args.config_file).parts[-1]).split(".")[:-1])
        config_file = args.config_file
    print(f"Source config: {config_file}")
    print(f"Results path:  {args.log_dir}/{exp_type}/{exp_group}/{exp_name}")
    return exp_group, exp_name, config_file


def setup_config_file_from_experiment_identifier(
        exp_type: str, exp_group: str, exp_name: str, *, config_dir: str = TrainerPathConst.DIR_CONFIG) -> Path:
    """
    Given the identifier parts, return Path to config yaml file.

    Args:
        exp_type: Experiment type
        exp_group: Experiment group
        exp_name: Experiment name
        config_dir: Config directory

    Returns:
        Path to config yaml.
    """
    return Path(config_dir) / exp_type / exp_group / f"{exp_name}.yaml"
