"""
Utilities for showing training results.
"""
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from nntrainer import maths
from nntrainer.experiment_organization import ExperimentFilesHandler
from nntrainer.metric import DEFAULT_METRICS, DefaultMetricsConst, PrintGroupConst, PrintMetric
from nntrainer.utils import LOGGER_NAME, TrainerPathConst


RESULTS_TYPE = Dict[str, Dict[str, float]]
RE_SPLIT_RUN = re.compile(r"(.*?)([0-9]+)")  # extract run number (group 2) and lazy match everything else (group 1)


def collect_results_data(exp_type: str, exp_groups_names: Dict[str, List[str]], *,
                         log_dir: str = TrainerPathConst.DIR_EXPERIMENTS,
                         read_last_epoch: bool = False, add_group: bool = False) -> RESULTS_TYPE:
    """
    Collect results from a folder of experiments.

    Args:
        exp_groups_names: Dictionary of experiment groups with a list of experiment names each.
        exp_type: Experiment type (ret for Retrieval, cap for Captioning, ...)
        log_dir: Save directory for experiments.
        read_last_epoch: Read last epoch instead of best epoch for experiments.
        add_group: Add experiment group to the model identifier.

    Returns:
        Dictionary of experiment runs, with a dictionary of metric name and value for each run. E.g.:
            {"experiment1": {"val_base/loss" : 0.5, "val_base/score" : 0.8}}
    """
    logger = logging.getLogger(LOGGER_NAME)
    log_dir = Path(log_dir)

    # collect experiments data
    collector: Dict[str, Dict[str, float]] = defaultdict(dict)
    not_found_list = []

    # which groups to search
    for exp_group, exp_names in exp_groups_names.items():
        # get experiment folder
        root_path = Path(log_dir) / exp_type / exp_group
        if not root_path.is_dir():
            raise FileNotFoundError(f"Path {root_path} given by -g/--exp_group not known.")

        for exp_ident in exp_names:
            # extract experiment and run name
            splits = exp_ident.split("_")
            exp_name, run_name = "_".join(splits[:-1]), splits[-1]

            if add_group:
                # add group to the identifier
                exp_ident = f"{exp_group}/{exp_ident}"

            # create the experiment helper and get relevant epochs
            handler = ExperimentFilesHandler(exp_type, exp_group, exp_name, run_name, log_dir=log_dir)
            last_epoch = handler.find_last_epoch()
            best_epoch = handler.find_best_epoch()

            # determine search epoch for validation results (default best epoch)
            search_epoch = last_epoch if read_last_epoch else best_epoch

            if search_epoch == -1:
                # if there was no training yet, check if there are at least some metrics
                metrics_epochs = handler.get_existing_metrics()
                if len(metrics_epochs) == 0:
                    not_found_list.append(exp_ident)
                    continue
                search_epoch = metrics_epochs[-1]

            # load epoch-based validation results
            epoch_file = handler.get_metrics_epoch_file(search_epoch)
            epoch_data = json.load(epoch_file.open("rt", encoding="utf8"))

            # save experiment group and name
            collector[exp_ident][DefaultMetricsConst.EXP_GROUP] = exp_group
            collector[exp_ident][DefaultMetricsConst.EXP_NAME] = exp_name
            collector[exp_ident][DefaultMetricsConst.RUN_NAME] = run_name

            # loop result metrics
            for key, metrics in epoch_data.items():
                # loop epoch, value tuples in the result
                found = 0
                result = 0
                for ep, value in metrics:
                    # look for the epoch we are searching
                    if ep != search_epoch:
                        continue
                    found += 1
                    result = value
                # there must be exactly 1 epoch logged for this metric
                assert found == 1, f"File {epoch_file} metric {key} found {found} results for "\
                                   f"epoch {search_epoch} in:\n{metrics}"
                collector[exp_ident][key] = result

            # load step-based metrics (LR, step times, GPU / RAM profiles)
            step_file = handler.get_metrics_step_file(search_epoch)
            if not step_file.is_file():
                logger.debug("Skipping step metrics (not found).")
                continue
            step_data = json.load(step_file.open("rt", encoding="utf8"))

            # average step timings and GPU load over all steps
            for key in (DefaultMetricsConst.TIME_STEP_FORWARD, DefaultMetricsConst.TIME_STEP_BACKWARD,
                        DefaultMetricsConst.TIME_STEP_OTHER,
                        DefaultMetricsConst.TIME_STEP_TOTAL, DefaultMetricsConst.PROFILE_GPU_LOAD):
                avg_val = np.mean([val for _, val in step_data[f"{key}-avg"]])
                collector[exp_ident][key] = avg_val

            # take the max of GPU and RAM memory load over all steps
            for key in (DefaultMetricsConst.PROFILE_GPU_MEM_USED, DefaultMetricsConst.PROFILE_RAM_USED):
                max_val = np.max([val for _, val in step_data[key]])
                collector[exp_ident][key] = max_val
    if len(not_found_list) > 0:
        # print warning about not having found some results
        logger.info(f"No results found for {not_found_list}")

    return collector


def update_performance_profile(collector: RESULTS_TYPE, profiling_dir=TrainerPathConst.DIR_PROFILING):
    """
    Args:
        collector: Dictionary of experiment runs, with a dictionary of metric name and value for each run.
        profiling_dir: Directory with stored performance results.

    Returns:
        Same collector dictionary with updated performance values.
    """
    for _exp_ident, metrics in collector.items():
        exp_group = metrics[DefaultMetricsConst.EXP_GROUP]
        exp_name = metrics[DefaultMetricsConst.EXP_NAME]
        performance_file = Path(profiling_dir) / f"{exp_group}_{exp_name}.json"
        if not performance_file.is_file():
            continue
        performance_data = json.load(performance_file.open("rt", encoding="utf8"))
        metrics[DefaultMetricsConst.PERF_PARAMS] = float(performance_data["params_total"])
        metrics[DefaultMetricsConst.PERF_SPEED] = float(performance_data["forward_time_per"])
        metrics[DefaultMetricsConst.PERF_GFLOPS] = float(performance_data["total_gflops"])
    return collector


def average_results_data(collector: RESULTS_TYPE, group_by_names: bool = False
                         ) -> Tuple[RESULTS_TYPE, RESULTS_TYPE, Dict[str, int]]:
    """
    Average all metrics in all models and store mean and sample stddev with Bessel's correction.

    Args:
        collector: Dictionary of experiment runs, with a dictionary of metric name and value for each run.
        group_by_names: Only group runs in an experiment if the run names match.

    Returns:
        Tuple of:
            Dictionary of experiment runs, with a dictionary of metric name and mean for each run.
            Same dictionary but with stddev for each run.
            Number of models averaged over
    """
    # collect list of values for all models and metrics
    collector_multi: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for exp_ident, metrics in collector.items():
        # extract experiment and run name
        splits = exp_ident.split("_")
        exp_name, run_name_full = "_".join(splits[:-1]), splits[-1]
        collector_name = exp_name
        if group_by_names:
            # only group if runs have the same name
            run_name, _run_number = RE_SPLIT_RUN.findall(run_name_full)[0]
            collector_name = f"{exp_name}_{run_name}"
        for metric_name, metric_value in metrics.items():
            collector_multi[collector_name][metric_name].append(metric_value)

    # calculate mean and stddev for all models and metrics
    collector_mean: RESULTS_TYPE = defaultdict(dict)
    collector_stddev: RESULTS_TYPE = defaultdict(dict)
    num_models: Dict[str, int] = {}
    for exp_name, metrics in collector_multi.items():
        for metric_name, metric_value_list in metrics.items():
            values = np.array(metric_value_list)
            if len(values) == 1 or isinstance(metric_value_list[0], str):
                mean = values[0]
                stddev = 0
            else:
                mean = np.mean(metric_value_list)
                stddev = np.sqrt(1 / (len(values) - 1) * np.sum((values - mean) ** 2))
            collector_mean[exp_name][metric_name] = mean
            collector_stddev[exp_name][metric_name] = stddev
            num_models[exp_name] = len(values)
    return collector_mean, collector_stddev, num_models


def output_results(
        collector: RESULTS_TYPE, custom_metrics: Optional[Dict[str, PrintMetric]] = None,
        metrics: str = PrintGroupConst.BASE, default_metrics: Optional[List[str]] = None, fields: str = "",
        default_fields: Optional[List[str]] = None, mean: bool = False,
        mean_all: bool = False, sort: str = "score", sort_asc: bool = False, compact: bool = False,
        print_fn: Callable = print) -> None:
    """
    Given a collection of result metrics and various options, output the results as a text table.

    Examples:
        >>> output_results({"exp1": {"loss": 0.3}})

    Args:
        collector: Dictionary of experiment runs, with a dictionary of metric name and value for each run.
        custom_metrics: Dictionary of custom metrics to print.
        metrics: Additional groups of metrics to print (default only basic metrics).
        default_metrics: Default groups of metrics to print.
        fields: Additional single metrics to print.
        default_fields: Default single metrics to print.
        mean: Average over multiple runs of the same experiment with the same run name.
        mean_all: Average over all runs of the same experiment, regardless of run name.
        sort: Sort by this field (default score).
        sort_asc: Sort ascending instead of descending.
        compact: Make the output table smaller.
        print_fn: This function is called to print the output. Override to save output to string instead of console.
    """
    logger = logging.getLogger(LOGGER_NAME)

    # ---------- Determine metrics to show ----------

    # create dictionary of all metrics that can possibly be printed
    if custom_metrics is None:
        custom_metrics = {}
    all_metrics: Dict[str, PrintMetric] = {**DEFAULT_METRICS, **custom_metrics}

    # determine which additional groups of metrics to show
    groups_available = list(set(metric.print_group for metric in all_metrics.values()))
    groups_to_print = default_metrics if default_metrics is not None else []
    if metrics == "all":
        groups_to_print = groups_available  # print all possible metrics
    elif metrics != "":
        # convert possible comma-separated list and add it to the groups to print
        groups_to_print += [group.strip() for group in metrics.split(",")]
    for group in groups_to_print:
        assert group in groups_available, (
            f"Metric group {group} requested for printing but it doesn't exist in {groups_available}")

    # determine which additional metric fields to show
    fields_available = list(set(all_metrics.keys()))
    fields_to_print = default_fields if default_fields is not None else []
    if fields != "":
        # add single fields for output if requested
        fields_to_print += [field.strip() for field in fields.split(",")]
    for field in fields_to_print:
        assert field in fields_available, (
            f"Metric field {field} requested for printing but it doesn't exist in {fields_available}")

    # ---------- Average, sort and output the results ----------

    assert not (mean and mean_all), "--mean and --mean_all cannot be true at the same time."
    if mean_all:
        # average over all runs for each experiment
        collector_mean, collector_stddev, num_models = average_results_data(collector)
    elif mean:
        # average over all runs for each experiment only if they have the same name
        collector_mean, collector_stddev, num_models = average_results_data(collector, group_by_names=True)
    else:
        # don't average
        collector_mean, collector_stddev, num_models = collector, None, None

    # sort by given sort key
    if sort == "alpha":
        # sort alphabetically
        sorted_model_names = sorted(collector_mean.keys())
    else:
        # sort models by given metric, fail-safe if the field doesn't exist
        sort_key = all_metrics[sort].long_name
        sort_values = []
        for metrics_names in collector_mean.values():
            if sort_key in metrics_names:
                sort_values.append(metrics_names[sort_key])
            else:
                sort_values.append(0)
        sort_idx = np.argsort(sort_values)
        sorted_model_names = np.array(list(collector_mean.keys()))[sort_idx]

    if not sort_asc:
        # sort descending if not requested otherwise
        sorted_model_names = reversed(sorted_model_names)

    # print which groups are available and which will be printed
    logger.info(f"Metrics (-m) to print: {set(groups_to_print)}, available groups: 'all' or {groups_available}")

    # define which keys to print
    print_keys_all = fields_to_print + [key for key, metr in all_metrics.items() if metr.print_group in groups_to_print]

    # skip keys if there is no data
    print_keys, skipped = [], []
    for key in print_keys_all:
        metr = all_metrics[key]
        if not any(metr.long_name in model for model in collector_mean.values()):
            skipped.append(key)
        else:
            print_keys.append(key)
    if len(skipped) > 0:
        logger.info(f"Skipped {skipped} because there is no data for them.")
    logger.info(f"Printing {print_keys}")

    # create table header
    header: List[str] = ["experiment" + " (num)" if num_models is not None else "experiment"] + print_keys

    # create table body
    body: List[List[str]] = []
    correct_spaces = []
    for model in sorted_model_names:
        correct_spaces_line = []
        # create model name string (first table column)
        out_str = f"{model}"
        if num_models is not None:
            # print how many models we averaged over
            out_str += f" ({num_models[model]})"
        body_line = [out_str]

        # loop metrics (all other columns)
        for key in print_keys:
            # get metric properties (long name, formatting etc.)
            metr = all_metrics[key]
            formatter = "{:." + str(metr.decimals) + metr.formatting + "}"
            format_lambda = metr.format_lambda
            # get (mean) value for printing
            value = 0
            if metr.long_name in collector_mean[model]:
                value = collector_mean[model][metr.long_name]
            if format_lambda is not None:
                value = format_lambda(value)
            out_str = formatter.format(value)

            if collector_stddev is not None:
                # get stddev for printing
                value_std = 0
                if metr.long_name in collector_stddev[model]:
                    value_std = collector_stddev[model][metr.long_name]
                if format_lambda is not None:
                    value_std = format_lambda(value_std)
                if value_std != 0:
                    std_str = formatter.format(value_std)
                    out_str = f"{out_str} ±{std_str}"
                    correct_spaces_line.append(len(std_str))
                else:
                    correct_spaces_line.append(0)
            body_line.append(out_str)
        body.append(body_line)
        correct_spaces.append(correct_spaces_line)

    # in table body, realign plusminus sign for standard deviation
    correct_spaces = np.array(correct_spaces)
    if 0 not in correct_spaces.shape:
        max_per_col = correct_spaces.max(axis=0)
        for row in range(correct_spaces.shape[0]):
            for col in range(correct_spaces.shape[1]):
                dist = max_per_col[col] - correct_spaces[row, col]
                if dist > 0:
                    body[row][col + 1] = (" ±" + " " * dist).join(cell.replace("±", "")
                                                                  for cell in body[row][col + 1].split(" "))

    # display
    print_fn()
    if compact:
        display_table_compact(body, header, print_fn=print_fn)
    else:
        display_table(body, header, print_fn=print_fn)


# ---------- Console table printing ----------

COLOR_DEFAULT = "[39m"
COLOR_WHITE = "[96m"  # "[97m"
# 0 default gray
# 1-9 formatting
# 30-39 dark
# 40-47 bg dark
# 90-97 light
# 100-107 bg light
COLOR_CODE = "\033"


# # view all
# CSI = "\x1B["
# for n in range(108): print(n, CSI+f"31;{n}m" + u"\u2588" + CSI + "0m")


def get_color(num: int) -> str:
    """
    Alternating colors for console output.
    """
    if num % 2 == 0:
        return COLOR_CODE + COLOR_DEFAULT
    return COLOR_CODE + COLOR_WHITE


def get_color_reset() -> str:
    return COLOR_CODE + COLOR_DEFAULT


def display_table(lines: List[List[str]], header: List[str] = None, sep_line="---", use_colors=True,
                  merger="|", merge_spaces=1, merge_edges=True, sep_line_repeat=False, print_fn: Callable = print
                  ) -> None:
    """
    Display content as a nicely formatted table. Accepts either lists or numpy arrays.

    Args:
        lines: list of lines, one line is a list of str with one entry per
            field.
        header: optional header names of the fields
        sep_line: "---" default for markdown compatibility, "-" for compact
            view, "" for no splitter
        use_colors: format alternating lines with different colors, default
            True
        merger: "|" default for markdown compatibility, " " for compact view
        merge_spaces: 1 default for markdown, 0 for compact view
        merge_edges: default True for markdown, False for compact view
        sep_line_repeat: asserts sep_line is single character and repeats
            that character in the sep line
        print_fn: This function is called to print the output. Override to save output to string instead of console.

    Returns:
    """

    # make sure input is valid
    if len(lines) == 0:
        print_fn("nothing to display (no lines)")
        return
    base_len = len(lines[0])
    if header is not None:
        base_len = len(header)
    for i, line in enumerate(lines):
        assert base_len == len(line),\
            "line {} has length {} but base length is {}".format(
                i, len(line), base_len)

    # build numpy array for header
    top_arr = None
    if header is not None:
        lines_arr = [header]
        if sep_line != "":
            lines_arr.append([sep_line] * len(header))
        top_arr = np.array(lines_arr)

    # build array for content and concatenate
    lines_arr = np.array(lines)
    if top_arr is not None:
        lines_arr = np.concatenate([top_arr, lines_arr])

    # output data, all columns same width
    lines_len = maths.np_str_len(lines_arr)
    max_lens = np.max(lines_len, axis=0)
    for i, line in enumerate(lines_arr):
        line_content = []
        if use_colors and (i >= 2 and sep_line != "") or (i >= 1 and sep_line == ""):
            print_fn(get_color(i), end="")
        if i == 1 and sep_line_repeat:
            sep_line = sep_line[:1]
            # special behaviour for repeating sep lines
            for m in max_lens:
                line_content.append(sep_line * m)
        else:
            # collect values
            for j, (val, max_len) in enumerate(zip(line, max_lens)):
                align = ">"
                if j == 0:
                    align = "<"
                line_content.append(("{:" + str(align) + str(max_len) + "s}").format(val))
        # output given the parameters and values
        spaces = " " * merge_spaces
        if merge_edges:
            print_fn(merger + spaces, end="")
        merger_total = spaces + merger + spaces
        print_fn(merger_total.join(line_content), end="")
        if merge_edges:
            print_fn(spaces + merger, end="")
        print_fn()
    if use_colors:
        print_fn(get_color_reset())


def display_table_compact(lines: List[List[str]], header: List[str] = None, use_colors: bool = True,
                          print_fn: Callable = print) -> None:
    """
    Convenience function to display a table that is as narrow as possible.

    Args:
        lines: Data lines.
        header: Data header.
        use_colors: Alternate row colors for readability.
        print_fn: This function is called to print the output. Override to save output to string instead of console.
    """
    return display_table(lines, header, sep_line="-", use_colors=use_colors, merger="|", merge_spaces=0,
                         merge_edges=False, sep_line_repeat=True, print_fn=print_fn)
