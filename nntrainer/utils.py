"""
General utilities: Logging, ArgumentParser with better formatting, Time / File utilities
"""
import argparse
import datetime
import logging
import os
import sys
from collections import defaultdict
from copy import deepcopy
from json import JSONEncoder
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pathspec

from nntrainer import typext
from nntrainer.typext import ConstantHolder


DEFAULT = "default"
REF = "ref"
NONE = "none"
LOGGER_NAME = "trainlog"
LOGGING_FORMATTER = logging.Formatter("%(levelname)5s %(message)s", datefmt="%m%d %H%M%S")


class LogLevelsConst(ConstantHolder):
    """
    Loglevels, same as logging module.
    """
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0


def create_logger_without_file(name: str, log_level: int = LogLevelsConst.INFO, no_parent: bool = False,
                               no_print: bool = False) -> logging.Logger:
    """
    Create a stdout only logger.

    Args:
        name: Name of the logger.
        log_level: Verbosity level.
        no_parent: Disable parents, can be used to avoid duplicate log entries.
        no_print: Do not print a message on creation.
    Returns:
        Created logger.
    """
    return create_logger(name, log_dir="", log_level=log_level, no_parent=no_parent, no_print=no_print)


def create_logger(
        name: str, *, filename: str = "run", log_dir: Union[str, Path] = "", log_level: int = LogLevelsConst.INFO,
        no_parent: bool = False, no_print: bool = False) -> logging.Logger:
    """
    Create a new logger.

    Notes:
        This created stdlib logger can later be retrieved with logging.getLogger(name) with the same name.
        There is no need to pass the logger instance between objects.

    Args:
        name: Name of the logger.
        log_dir: Target logging directory. Empty string will not create files.
        filename: Target filename.
        log_level: Verbosity level.
        no_parent: Disable parents, can be used to avoid duplicate log entries.
        no_print: Do not print a message on creation.

    Returns:
    """
    # create logger, set level
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # remove old handlers to avoid duplicate messages
    remove_handlers_from_logger(logger)

    # file handler
    file_path = None
    if log_dir != "":
        ts = get_timestamp_for_filename()
        file_path = Path(log_dir) / "{}_{}.log".format(filename, ts)
        file_hdlr = logging.FileHandler(str(file_path))
        file_hdlr.setFormatter(LOGGING_FORMATTER)
        logger.addHandler(file_hdlr)

    # stdout handler
    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(LOGGING_FORMATTER)
    logger.addHandler(strm_hdlr)

    # disable propagating to parent to avoid double logs
    if no_parent:
        logger.propagate = False

    if not no_print:
        print(f"Logger: '{name}' to {file_path}")
    return logger


def remove_handlers_from_logger(logger: logging.Logger) -> None:
    """
    Remove handlers from the logger.

    Args:
        logger: Logger.
    """
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.flush()
        handler.close()


def print_logger_info(logger: logging.Logger) -> None:
    """
    Print infos describing the logger: The name and handlers.

    Args:
        logger: Logger.
    """
    print(logger.name)
    x = list(logger.handlers)
    for i in x:
        handler_str = f"Handler {i.name} Type {type(i)}"
        print(handler_str)


# ---------- Argparser ----------


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter,
                      argparse.MetavarTypeHelpFormatter):
    """
    Custom formatter
    - raw descriptions (no removing newlines)
    - show default values
    - show metavars (str, int, ...) instead of names
    - fit to console width
    """

    def __init__(self, prog: Any):
        try:
            term_size = os.get_terminal_size().columns
            max_help_pos = term_size // 2
        except OSError:
            term_size = None
            max_help_pos = 24
        super().__init__(
            prog, max_help_position=max_help_pos, width=term_size)


class ArgParser(argparse.ArgumentParser):
    """
    ArgumentParser with Custom Formatter and some convenience functions.

    For best results, write a docstring at the top of the file and call it
    with ArgParser(description=__doc__)

    Args:
        description: Help text for Argparser. Set description=__doc__ and write help text into module header.
    """

    def __init__(self, description: str = "none"):
        super().__init__(description=description, formatter_class=CustomFormatter)


# ---------- Time utilities ----------

def get_timestamp_for_filename(dtime: Optional[datetime.datetime] = None):
    """
    Convert datetime to timestamp for filenames.

    Args:
        dtime: Optional datetime object, will use now() if not given.

    Returns:
    """
    if dtime is None:
        dtime = datetime.datetime.now()
    ts = str(dtime).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    return ts


# ---------- Files ----------
def parse_file_to_list(file: Union[Path, str]) -> List[str]:
    """
    Given a text file, read contents to list of lines. Strip lines, ignore empty and comment lines

    Args:
        file: Input file.

    Returns:
        List of lines.
    """
    # loop lines
    output = []
    for line in Path(file).read_text(encoding="utf8").splitlines(keepends=False):
        # strip line
        line = line.strip()
        if line == "":
            # skip empty line
            continue
        if line[0] == "#":
            # skip comment line
            continue
        # collect
        output.append(line)
    return output


# ---------- Config / dict ----------

def resolve_sameas_config_recursively(config: Dict, *, root_config: Optional[Dict] = None):
    """
    Recursively resolve config fields described with same_as.

    If any container in the config has the field "same_as" set, find the source identifier and copy all data
    from there to the target container. The source identifier can nest with dots e.g.
    same_as: "net_video_local.input_fc_config" will copy the values from container input_fc_config located inside
    the net_video_local container.

    Args:
        config: Config to modify.
        root_config: Config to get the values from, usually the same as config.

    Returns:
    """
    if root_config is None:
        root_config = config
    # loop the current config and check
    loop_keys = list(config.keys())
    for key in loop_keys:
        value = config[key]
        if not isinstance(value, dict):
            continue
        same_as = value.get("same_as")
        if same_as is not None:
            # current container should be filled with the values from the source container. loop source container
            source_container = get_dict_value_recursively(root_config, same_as)
            for key_source, val_source in source_container.items():
                # only write fields that don't exist yet, don't overwrite everything
                if key_source not in config[key]:
                    # at this point we want a deepcopy to make sure everything is it's own object
                    config[key][key_source] = deepcopy(val_source)
            # at this point, remove the same_as field.
            del value["same_as"]

        # check recursively
        resolve_sameas_config_recursively(config[key], root_config=root_config)


def get_dict_value_recursively(dct: Dict, key: str) -> Any:
    """
    Nest into the dict given a key like root.container.subcontainer

    Args:
        dct: Dict to get the value from.
        key: Key that can describe several nesting steps at one.

    Returns:
        Value.
    """
    key_parts = key.split(".")
    if len(key_parts) == 1:
        # we arrived at the leaf of the dict tree and can return the value
        return dct[key_parts[0]]
    # nest one level deeper
    return get_dict_value_recursively(dct[key_parts[0]], ".".join(key_parts[1:]))


def check_config_dict(name: str, config: Dict[str, Any], strict: bool = True) -> None:
    """
    Make sure config has been read correctly with .pop(), and no fields are left over.

    Args:
        name: config name
        config: config dict
        strict: Throw errors
    """
    remaining_keys, remaining_values = [], []
    for key, value in config.items():
        if key == REF:
            # ignore the reference configurations, they can later be used for copying things with same_as
            continue
        remaining_keys.append(key)
        remaining_values.append(value)
    # check if something is left over
    if len(remaining_keys) > 0:
        if not all(value is None for value in remaining_values):
            err_msg = (
                f"keys and values remaining in config {name}: {remaining_keys}, {remaining_values}. "
                f"Possible sources of this error: Typo in the field name in the yaml config file. "
                f"Incorrect fields given with --config flag. "
                f"Field should be added to the config class so it can be parsed. "
                f"Using 'same_as' and forgot to set these fields to null.")

            if strict:
                print(f"Print config for debugging: {config}")
                raise ValueError(err_msg)
            logging.getLogger(LOGGER_NAME).warning(err_msg)


def create_string_matcher(pattern: Union[str, List[str]]) -> pathspec.PathSpec:
    """
    Given one or several patterns with the syntax of a .gitignore file, create a matcher object that can
    be used to match strings against the pattern.

    Args:
        pattern: One or several patterns.

    Returns:
        PathSpec matcher object.
    """
    if isinstance(pattern, str):
        pattern = [pattern]
    matcher = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, pattern)
    return matcher


def match_folder(folder: Union[str, Path], exp_type: str, exp_group: str = None,
                 exp_list: Optional[Union[Path, str]] = None, search: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Match experiments in a folder.

    Args:
        folder: Folder of experiments to match, should be setup like FOLDER/EXP_TYPE/EXP_GROUP/EXP_NAME
        exp_type:
        exp_group:
        exp_list:
        search:

    Returns:
        Dictionary of experiment groups with a list of experiment names each.
    """
    logger = logging.getLogger(LOGGER_NAME)
    assert not (exp_list is not None and exp_group is not None), (
        "Cannot provide --exp_list and --exp_group at the same time.")

    # determine experiment group/name combinations to search
    exp_matcher_raw = []
    if exp_list is not None:
        # get experiment groups to search in from list
        exp_list_lines = Path(exp_list).read_text(encoding="utf8").splitlines(keepends=False)
        for line in exp_list_lines:
            line = line.strip()
            if line == "" or line[0] == "#":
                continue
            exp_matcher_raw.append(line)
    elif exp_group is not None:
        # get experiment groups from the argument
        for group in exp_group.split(","):
            exp_matcher_raw.append(group.strip())
    else:
        # include all groups and experiments
        exp_matcher_raw.append("*")
    matcher = create_string_matcher(exp_matcher_raw)

    # determine experiment name to search
    search_names = []
    if search is None:
        search_names.append("*")
    else:
        for name in search.split(","):
            search_names.append(name.strip())
    name_matcher = create_string_matcher(search_names)

    # determine root path and print infos
    root_path = Path(folder) / exp_type

    logger.info(f"Matching in {root_path} for --exp_group {exp_matcher_raw}, names --search {search_names}")

    # get all experiments and groups
    found = defaultdict(list)
    for new_exp_group in sorted(os.listdir(root_path)):
        for new_exp_name in sorted(os.listdir(root_path / new_exp_group)):
            # when searching configs, remove the .yaml ending
            if new_exp_name.endswith(".yaml"):
                new_exp_name = new_exp_name[:-5]
            # match group and name
            match_str = f"{new_exp_group}/{new_exp_name}"
            if matcher.match_file(match_str) and name_matcher.match_file(new_exp_name):
                found[new_exp_group].append(new_exp_name)

    logger.debug(f"Found: {found}")

    return found


class BetterJSONEncoder(JSONEncoder):
    """
    Enable the JSON encoder to handle Path objects.

    It would be nice to also handle numpy arrays, tensors etc. but that is not required currently.
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


# ---------- Constants ----------

class ConfigNamesConst(typext.ConstantHolder):
    """
    Stores configuration group names.
    """
    TRAIN = "train"
    VAL = "val"
    DATASET_TRAIN = "dataset_train"
    DATASET_VAL = "dataset_val"
    LOGGING = "logging"
    SAVING = "saving"
    OPTIMIZER = "optimizer"
    LR_SCHEDULER = "lr_scheduler"


class TrainerPathConst(typext.ConstantHolder):
    """
    S
    tores directory and file names for training.
    """
    DIR_CONFIG = "config"
    DIR_EXPERIMENTS = "experiments"
    DIR_LOGS = "logs"
    DIR_MODELS = "models"
    DIR_METRICS = "metrics"
    DIR_EMBEDDINGS = "embeddings"
    DIR_TB = "tb"
    DIR_PROFILING = "profiling"
    DIR_CAPTION = "caption"
    DIR_ANNOTATIONS = "annotations"
    FILE_PREFIX_TRAINERSTATE = "trainerstate"
    FILE_PREFIX_MODEL = "model"
    FILE_PREFIX_MODELEMA = "modelema"
    FILE_PREFIX_OPTIMIZER = "optimizer"
    FILE_PREFIX_DATA = "data"
    FILE_PREFIX_METRICS_STEP = "metrics_step"
    FILE_PREFIX_METRICS_EPOCH = "metrics_epoch"
    FILE_PREFIX_TRANSL_RAW = "translations"
    FILE_PREFIX_TRANSL_LANG = "results_lang"
    FILE_PREFIX_TRANSL_STAT = "results_stat"
    FILE_PREFIX_TRANSL_REP = "results_rep"
    FILE_PREFIX_TRANSL_METRICS = "text_metrics"


class MetricComparisonConst(typext.ConstantHolder):
    """
    Fields for the early stopper.
    """
    # metric comparison
    VAL_DET_BEST_MODE_MIN = "min"
    VAL_DET_BEST_MODE_MAX = "max"
    VAL_DET_BEST_TH_MODE_REL = "rel"
    VAL_DET_BEST_TH_MODE_ABS = "abs"
