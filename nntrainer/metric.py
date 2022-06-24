"""
Metric writing and reading utilities.

This automates the following:
    - Logging the same metrics both to text files and to tensorboard.
    - Reload old metrics when resuming training.
    - Only save metrics where something was logged to.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch as th
from torch.utils.tensorboard import SummaryWriter

from nntrainer import typext
from nntrainer.experiment_organization import ExperimentFilesHandler
from nntrainer.typext import ConstantHolder
from nntrainer.utils import LOGGER_NAME


class PrintGroupConst(typext.ConstantHolder):
    """
    Define metric groups. This is used for creating experiment tables.
    """
    BASE = "base"
    PROFILE = "profile"
    PERFORMANCE = "performance"


class PrintMetric(typext.TypedNamedTuple):
    """
    Define named tuple for string formatting of metrics.
    """
    long_name: str
    formatting: str
    decimals: int
    print_group: str
    format_lambda: Optional[Callable[[float], float]]


class DefaultMetricsConst(ConstantHolder):
    """
    Define metric names. Forward slash groups them in tensorboard.
    """
    TRAIN_EPOCH = "train_base/epoch"
    TIME_TOTAL = "ztime/time_total"
    TIME_VAL = "ztime/time_val"
    VAL_LOSS = "val_base/loss"
    VAL_BEST_FIELD = "val_base/best_field"
    TRAIN_LR = "train_base/lr"
    PROFILE_GPU_MEM_PERCENT = "zgpu/mem_percent"
    PROFILE_GPU_MEM_USED = "zgpu/mem_used"
    TIME_STEP_FORWARD = "ztime/step_forward"
    TIME_STEP_BACKWARD = "ztime/step_backward"
    TIME_STEP_TOTAL = "ztime/step_total"
    TIME_STEP_OTHER = "ztime/step_other"
    TRAIN_GRAD_CLIP = "train_base/grad_clip_total_norm"
    TRAIN_LOSS = "train_base/loss"
    PROFILE_GPU_LOAD = "zgpu/load"
    # not logged
    PROFILE_GPU_MEM_TOTAL = "zgpu/mem_total"
    PROFILE_RAM_TOTAL = "zram/total"
    PROFILE_RAM_USED = "zram/used"
    PROFILE_RAM_AVAILABLE = "zram/avail"
    # performance
    PERF_GFLOPS = "perf/gflops"
    PERF_PARAMS = "perf/params"
    PERF_SPEED = "perf/speed"
    # IDENTIFIER
    EXP_GROUP = "exp_group"
    EXP_NAME = "exp_name"
    RUN_NAME = "run_name"


DEFAULT_METRICS = {
    "ep": PrintMetric(DefaultMetricsConst.TRAIN_EPOCH, "f", 0, PrintGroupConst.BASE, None),
    "loss": PrintMetric(DefaultMetricsConst.VAL_LOSS, "f", 3, PrintGroupConst.BASE, None),
    "score": PrintMetric(DefaultMetricsConst.VAL_BEST_FIELD, "f", 3, PrintGroupConst.BASE, None),
    "GPU mem": PrintMetric(DefaultMetricsConst.PROFILE_GPU_MEM_USED, "f", 0, PrintGroupConst.PROFILE, None),
    "GPU load": PrintMetric(DefaultMetricsConst.PROFILE_GPU_LOAD, "f", 1, PrintGroupConst.PROFILE, None),
    "RAM": PrintMetric(DefaultMetricsConst.PROFILE_RAM_USED, "f", 1, PrintGroupConst.PROFILE, None),
    "Time": PrintMetric(DefaultMetricsConst.TIME_TOTAL, "f", 2, PrintGroupConst.PROFILE, lambda x: x / 3600),
    "GFlop": PrintMetric(DefaultMetricsConst.PERF_GFLOPS, "f", 3, PrintGroupConst.PERFORMANCE, None),
    "MPar": PrintMetric(DefaultMetricsConst.PERF_PARAMS, "f", 2, PrintGroupConst.PERFORMANCE, lambda x: x / 1e6),
    "InfMS": PrintMetric(DefaultMetricsConst.PERF_SPEED, "f", 2, PrintGroupConst.PERFORMANCE, None),
}


# ---------- Text metrics ----------

class MartPrintGroupConst(PrintGroupConst):
    TEXT = "text"


class TextMetricsConst(ConstantHolder):
    """
    Text metrics names.
    """
    BLEU_1 = "cap/b1"
    BLEU_2 = "cap/b2"
    BLEU_3 = "cap/b3"
    BLEU_4 = "cap/b4"
    METEOR = "cap/met"
    ROUGE_L = "cap/rol"
    CIDER = "cap/cid"
    RE1 = "cap/re1"
    RE2 = "cap/re2"
    RE3 = "cap/re3"
    RE4 = "cap/re4"
    SUBMISSION_VOCAB_SIZE = "cap/voc"
    SUBMISSION_AVG_SEN_LEN = "cap/slen"
    SUBMISSION_NUM_SEN = "cap/snum"
    GT_STAT_VOCAB_SIZE = "capgt/voc"
    GT_STAT_AVG_SEN_LEN = "capgt/slen"
    GT_STAT_NUM_SEN = "capgt/snum"


class TextMetricsConstEvalCap(ConstantHolder):
    """
    Text metrics names as provided by PyCocoEvalCap
    """
    BLEU_1 = "Bleu_1"
    BLEU_2 = "Bleu_2"
    BLEU_3 = "Bleu_3"
    BLEU_4 = "Bleu_4"
    METEOR = "METEOR"
    ROUGE_L = "ROUGE_L"
    CIDER = "CIDEr"
    RE1 = "re1"
    RE2 = "re2"
    RE3 = "re3"
    RE4 = "re4"
    SUBMISSION_VOCAB_SIZE = "submission_vocab_size"
    SUBMISSION_AVG_SEN_LEN = "submission_avg_sen_len"
    SUBMISSION_NUM_SEN = "submission_num_sen"
    GT_STAT_VOCAB_SIZE = "gt_stat_vocab_size"
    GT_STAT_AVG_SEN_LEN = "gt_stat_avg_sen_len"
    GT_STAT_NUM_SEN = "gt_stat_num_sen"


# create mapper from pycocoevalcap results to tensorboard names
keys1, keys2 = list(TextMetricsConst.keys()), list(TextMetricsConstEvalCap.keys())
assert keys1 == keys2, (
    f"Mismatch in text metrics definition, the constant holder classes must match.\n{keys1}\n-----\n{keys2}")
TRANSLATION_METRICS = {TextMetricsConstEvalCap.get(key): name for key, name in TextMetricsConst.items()}

# TRANSLATION_METRICS = {
#     "Bleu_1": TextMetricsConst.BLEU_1,
#     "Bleu_2": TextMetricsConst.BLEU_2,
#     "Bleu_3": TextMetricsConst.BLEU_3,
#     "Bleu_4": TextMetricsConst.BLEU_4,
#     "METEOR": TextMetricsConst.METEOR,
#     "ROUGE_L": TextMetricsConst.ROUGE_L,
#     "CIDEr": TextMetricsConst.CIDER,
#     "re1": TextMetricsConst.RE1,
#     "re2": TextMetricsConst.RE2,
#     "re3": TextMetricsConst.RE3,
#     "re4": TextMetricsConst.RE4,
#     "submission_vocab_size": TextMetricsConst.SUBMISSION_VOCAB_SIZE,
#     "submission_avg_sen_len": TextMetricsConst.SUBMISSION_AVG_SEN_LEN,
#     "submission_num_sen": TextMetricsConst.SUBMISSION_NUM_SEN,
#     "gt_stat_vocab_size": TextMetricsConst.GT_STAT_VOCAB_SIZE,
#     "gt_stat_avg_sen_len": TextMetricsConst.GT_STAT_AVG_SEN_LEN,
#     "gt_stat_num_sen": TextMetricsConst.GT_STAT_NUM_SEN
# }

TEXT_METRICS = {
    "bleu1": PrintMetric(TextMetricsConst.BLEU_1, "%", 2, MartPrintGroupConst.TEXT, None),
    "bleu2": PrintMetric(TextMetricsConst.BLEU_2, "%", 2, MartPrintGroupConst.TEXT, None),
    "bleu3": PrintMetric(TextMetricsConst.BLEU_3, "%", 2, MartPrintGroupConst.TEXT, None),
    "bleu4": PrintMetric(TextMetricsConst.BLEU_4, "%", 2, MartPrintGroupConst.TEXT, None),
    "meteo": PrintMetric(TextMetricsConst.METEOR, "%", 2, MartPrintGroupConst.TEXT, None),
    "rougl": PrintMetric(TextMetricsConst.ROUGE_L, "%", 2, MartPrintGroupConst.TEXT, None),
    "cider": PrintMetric(TextMetricsConst.CIDER, "%", 2, MartPrintGroupConst.TEXT, None),
    "re1": PrintMetric(TextMetricsConst.RE1, "%", 2, MartPrintGroupConst.TEXT, None),
    "re2": PrintMetric(TextMetricsConst.RE2, "%", 2, MartPrintGroupConst.TEXT, None),
    "re3": PrintMetric(TextMetricsConst.RE3, "%", 2, MartPrintGroupConst.TEXT, None),
    "re4": PrintMetric(TextMetricsConst.RE4, "%", 2, MartPrintGroupConst.TEXT, None),
    "c/voc": PrintMetric(TextMetricsConst.SUBMISSION_VOCAB_SIZE, "f", 0, MartPrintGroupConst.TEXT, None),
    "c/slen": PrintMetric(TextMetricsConst.SUBMISSION_AVG_SEN_LEN, "f", 2, MartPrintGroupConst.TEXT, None),
    "c/snum": PrintMetric(TextMetricsConst.SUBMISSION_NUM_SEN, "f", 0, MartPrintGroupConst.TEXT, None),
    "t/voc": PrintMetric(TextMetricsConst.GT_STAT_VOCAB_SIZE, "f", 0, MartPrintGroupConst.TEXT, None),
    "t/slen": PrintMetric(TextMetricsConst.GT_STAT_AVG_SEN_LEN, "f", 2, MartPrintGroupConst.TEXT, None),
    "t/snum": PrintMetric(TextMetricsConst.GT_STAT_NUM_SEN, "f", 0, MartPrintGroupConst.TEXT, None),
}


# ---------- Metric handlers ----------

class MetricsWriter:
    """
    Manager to store training and validation metrics.

    Args:
        exp: Helper for getting experiment file names.

    Attributes:
        meters: Dictionary of meters.
        meter_settings: Dictionary of settings for the meters.
        storage_step: For each metric, save a list of (step, value) tuples.
        storage_epoch: For each metric, save a list of (epoch, value) tuples.
        tensorb_writer: SummaryWriter for adding scalars to tensorboard.
    """

    def __init__(self, exp: ExperimentFilesHandler) -> None:
        self.exp = exp

        # meters for collecting individual values during training
        self.meters: Dict[str, AverageMeter] = {}
        self.meter_settings: Dict[str, MeterSettings] = {}

        # storage for metrics to be saved to files
        self.storage_step: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.storage_epoch: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

        # tensorboard writer
        self.tensorb_writer = SummaryWriter(log_dir=self.exp.path_tensorb)

    def add_meter(self, meter_name: str, *, per_step: bool = False, use_value: bool = True, use_avg: bool = True,
                  reset_avg_each_epoch: bool = False, no_tensorboard: bool = False) -> None:
        """
        Create meter with name and settings.

        Args:
            meter_name: Name of the meter.
            per_step: Metric will be provided per step (False: per epoch).
            use_value: Metric values will be logged.
            use_avg: Metric averages will be logged.
            reset_avg_each_epoch: Metric averages will reset at the beginning of each epoch.
            no_tensorboard: Metric will not be logged to tensorboard.
        """
        # make sure meter doesnt exist already
        assert meter_name not in self.meters, f"Meter {meter_name} already exists in {self.meters.keys()}"
        # create averagemeter
        self.meters[meter_name] = AverageMeter()
        self.meter_settings[meter_name] = MeterSettings(per_step, use_value, use_avg, reset_avg_each_epoch,
                                                        no_tensorboard)

    def update_meter(self, meter_name: str, value: float) -> None:
        """
        Update given meter with given value.

        Args:
            meter_name: Meter to update.
            value: Value to update.
        """
        if isinstance(value, th.Tensor):
            value = value.item()
        assert isinstance(value, (int, float)), (
            f"Got type {type(value).__name__} for metric {meter_name}.")
        if meter_name not in self.meters:
            raise KeyError(f"Meter {meter_name} does not exist . It was not created in the trainer __init__ method.")
        self.meters[meter_name].update(value)

    def hook_epoch_start(self) -> None:
        """
        Called at the start of each new training epoch, resets the relevant meters.
        """
        for meter_name, meter in self.meters.items():
            settings = self.meter_settings[meter_name]
            if settings.reset_avg_each_epoch:
                meter.reset()

    def feed_metrics_step(self, global_step: int, current_epoch: int) -> None:
        """
        Feed the collected step-based metrics to tensorboard and files.

        Args:
            global_step: Current global step.
            current_epoch: Current epoch.
        """
        self.feed_metrics(True, global_step, current_epoch)

    def feed_metrics_epoch(self, global_step: int, current_epoch: int):
        """
        Feed the collected epoch-based metrics to tensorboard and files.

        Args:
            global_step: Current global step.
            current_epoch: Current epoch.
        """
        self.feed_metrics(False, global_step, current_epoch)
        # write to file...

    def feed_metrics(self, per_step: bool, total_step: int, current_epoch: int) -> None:
        """
        Called at end of step or epoch, feeds tensorboard with metrics.

        Args:
            per_step: Whether to feed the "per step" or the "per epoch" meters.
            total_step: Current step since epoch 0.
            current_epoch: Current epoch.
        """
        for meter_name, meter in self.meters.items():
            settings = self.meter_settings[meter_name]
            if settings.per_step != per_step:
                # per_step variable must match meter
                continue
            if meter.count == 0:
                # skip empty meters. alot of meters will only be logged to sometimes.
                continue
            if settings.use_value:
                # log last value
                self.feed_single_metric(per_step, meter_name, meter.value, total_step, current_epoch,
                                        no_tensorboard=settings.no_tensorboard)
            if settings.use_avg:
                # log current average
                self.feed_single_metric(per_step, meter_name + "-avg", meter.avg, total_step, current_epoch,
                                        no_tensorboard=settings.no_tensorboard)

    def load_epoch(self, current_epoch: int) -> None:
        """
        Reload metric storage from file.

        Args:
            current_epoch: Training epoch.
        """
        logger = logging.getLogger(LOGGER_NAME)
        step_file = self.exp.get_metrics_step_file(current_epoch)
        if not step_file.is_file():
            logger.warning(f"Metrics in {step_file} not found, training metrics will be incomplete.")
        else:
            self.storage_step = defaultdict(list, json.load(step_file.open("rt")))

        epoch_file = self.exp.get_metrics_epoch_file(current_epoch)
        if not step_file.is_file():
            logger.warning(f"Metrics in {epoch_file} not found, training metrics will be incomplete.")
        else:
            self.storage_epoch = defaultdict(list, json.load(epoch_file.open("rt")))

    def save_epoch(self, current_epoch: int) -> None:
        """
        Save current metric storage to file.

        Args:
            current_epoch: Training epoch.
        """
        json.dump(self.storage_step, self.exp.get_metrics_step_file(current_epoch).open("wt"))
        json.dump(self.storage_epoch, self.exp.get_metrics_epoch_file(current_epoch).open("wt"))

    def save_epoch_to_file(self, file: Union[Path, str]) -> None:
        """
        Save current metric storage to file.

        Args:
            file: Target file.
        """
        json.dump(self.storage_epoch, Path(file).open("wt"))

    def feed_single_metric(
            self, per_step: bool, metric_name: str, metric_value: float, global_step: int, current_epoch: int, *,
            no_tensorboard: bool = False):
        """
        Args:
            per_step:
            metric_name:
            metric_value:
            global_step:
            current_epoch:
            no_tensorboard:

        Returns:
        """
        if not no_tensorboard:
            # write to tensorboard
            self.tensorb_writer.add_scalar(metric_name, metric_value, global_step=global_step)

        # write to metrics logger
        if per_step:
            self.storage_step[metric_name].append((global_step, metric_value))
        else:
            self.storage_epoch[metric_name].append((current_epoch, metric_value))

    def close(self) -> None:
        """
        Close all meters.
        """
        self.tensorb_writer.close()


class MeterSettings:
    """
    Storage class for meter settings for each individual meter.

    Args:
        per_step: Metric will be provided per step (False: per epoch)
        use_value: Metric values will be logged.
        use_avg: Metric averages will be logged.
        reset_avg_each_epoch: Metric averages will reset at the beginning of each epoch.
        no_tensorboard: Metric will not be logged to tensorboard.
    """

    def __init__(self, per_step: bool, use_value: bool, use_avg: bool, reset_avg_each_epoch: bool,
                 no_tensorboard: bool) -> None:
        self.per_step: bool = per_step
        self.use_value: bool = use_value
        self.use_avg: bool = use_avg
        self.reset_avg_each_epoch: bool = reset_avg_each_epoch
        self.no_tensorboard: bool = no_tensorboard


class AverageMeter:
    """
    AverageMeter that can be used to log values easily.
    """

    def __init__(self) -> None:
        self.value: float = 0
        self.sum: float = 0
        self.count: int = 0
        self.avg: float = 0

    def reset(self) -> None:
        """
        Set all values to zero.
        """
        self.value, self.sum, self.count, self.avg = 0, 0, 0, 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update meter with value.

        Args:
            val: Value to log.
            n: How many times to log that value.
        """
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
