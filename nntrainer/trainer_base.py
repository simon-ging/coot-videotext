"""
Generic Deep Learning trainer that automates tasks required for all kinds of training.
"""
import datetime
import logging
import os
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

import torch as th
from torch import nn
from torch.backends import cudnn
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader

from nntrainer import lr_scheduler, metric, models, trainer_configs, utils, utils_torch, utils_yaml
from nntrainer.experiment_organization import ExperimentFilesHandler
from nntrainer.metric import DefaultMetricsConst as Metrics
from nntrainer.utils import MetricComparisonConst


class BaseTrainer:
    """
    Base Trainer class. Inherited trainer instances must call hooks

    BaseTrainer takes care of Path setup, logging, device setup, checkpoints, metrics.


    determining device and moving models to cuda, setting up checkpoint loading and metrics.

    Args:
        cfg: Loaded configuration instance.
        model_mgr: Model manager.
        exp_group: Experiment group.
        exp_name: Experiment name.
        run_name: Experiment run.
        train_loader_length: Length of the train loader, required for some LR schedulers.
        model_type:
        log_dir: Directory to put results.
        log_level: Log level. None will default to INFO = 20 if a new logger is created.
        logger: Logger. With the default None, it will be created by the trainer.
        print_graph: Print graph and forward pass of the model.
        reset: Delete entire experiment and restart from scratch.
        load_best: Whether to load the best epoch (default loads last epoch to continue training).
        load_epoch: Whether to load a specific epoch.
        load_model: Load model given by file path.
        is_test: Removes some parts that are not needed during inference for speedup.
        exp_files_handler: Optionally provide instance to overwrite standard ExperimentFilesHandler
    """

    def __init__(
            self, cfg: trainer_configs.DefaultExperimentConfig, model_mgr: models.BaseModelManager,
            exp_group: str,
            exp_name: str, run_name: str, train_loader_length: int, model_type: str, *,
            log_dir: str = "experiments", log_level: Optional[int] = None,
            logger: Optional[logging.Logger] = None,
            print_graph: bool = False, reset: bool = False, load_best: bool = False,
            load_epoch: Optional[int] = None,
            load_model: Optional[str] = None, is_test: bool = False,
            exp_files_handler: ExperimentFilesHandler = None):
        assert "_" not in run_name, f"Run name {run_name} must not contain underscores."
        self.is_test: bool = is_test

        # save model manager
        self.model_mgr: models.BaseModelManager = model_mgr

        # create empty trainer state
        self.state = trainer_configs.BaseTrainerState()

        # save config
        self.cfg: trainer_configs.DefaultExperimentConfig = cfg

        # create experiment helper for directories, if it wasn't overwritten by the base trainer
        self.exp = exp_files_handler
        if self.exp is None:
            self.exp = ExperimentFilesHandler(model_type, exp_group, exp_name, run_name,
                                              log_dir=log_dir)
            self.exp.setup_dirs(reset=reset)

        # setup logging
        assert logger is None or log_level is None, "Cannot specify loglevel and logger together."
        if logger is None:
            if log_level is None:
                self.log_level = utils.LogLevelsConst.INFO
            else:
                self.log_level = log_level
            self.logger = utils.create_logger(utils.LOGGER_NAME, log_dir=self.exp.path_logs,
                                              log_level=self.log_level)
        else:
            self.logger = logger
            self.log_level = self.logger.level

        # print graph, check performance
        if print_graph:
            raise NotImplementedError

        # setup devices
        if not self.cfg.use_cuda:
            # force disable nn DataParallel and fp16 on CPUs
            self.cfg.use_multi_gpu = False
            self.cfg.fp16_train = False

        # setup grad scaler if needed for fp16
        self.grad_scaler: Optional[GradScaler] = None
        if self.cfg.fp16_train:
            self.grad_scaler: Optional[GradScaler] = GradScaler()

        # logs some infos
        self.logger.info(
                f"Running on cuda: {self.cfg.use_cuda}, multi-gpu: {self.cfg.use_multi_gpu}, "
                f"gpus found: {th.cuda.device_count()}, fp16 amp: {self.cfg.fp16_train}.")
        cudnn.enabled = self.cfg.cudnn_enabled
        cudnn.benchmark = self.cfg.cudnn_benchmark
        cudnn.deterministic = self.cfg.cudnn_deterministic

        # move models to device
        for model_name, model in self.model_mgr.model_dict.items():
            try:
                if self.cfg.use_cuda:
                    if not th.cuda.is_available():
                        raise RuntimeError(
                                "CUDA requested but not available! Use --no_cuda to run on CPU.")
                    if self.cfg.use_multi_gpu:
                        model = nn.DataParallel(model)
                    model = model.cuda()
                    self.model_mgr.model_dict[model_name] = model
            except RuntimeError as e:
                raise RuntimeError(
                        f"RuntimeError when putting model {type(model)} to cuda with DataParallel "
                        f"{self.cfg.use_multi_gpu}: {model.__class__.__name__}") from e

        # create metrics writer
        self.metrics = metric.MetricsWriter(self.exp)

        # print seed if it was set by the runner script
        self.logger.info(f"Random seed: {self.cfg.random_seed}")

        # dump yaml config to file
        utils_yaml.dump_yaml_config_file(self.exp.path_base / 'config.yaml', self.cfg.config_orig)

        # setup automatic checkpoint loading. this will be parsed in self.hook_post_init()
        ep_nums = self.exp.get_existing_checkpoints()
        self.load = False
        self.load_ep = -1
        self.load_model = load_model
        if self.load_model:
            assert not load_epoch, (
                    "When given filepath with load_model, --load_epoch must not be set.")
            self.load = True
        # automatically find best epoch otherwise
        elif len(ep_nums) > 0:
            if load_epoch:
                # load given epoch
                assert not load_best, "Load_epoch and load_best cannot be set at the same time."
                self.load_ep = load_epoch
            elif load_best:
                # load best epoch
                self.logger.info("Load best checkpoint...")
                best_ep = self.exp.find_best_epoch()
                if best_ep == -1:
                    # no validation done yet, load last
                    self.load_ep = ep_nums[-1]
                else:
                    self.load_ep = best_ep
                self.logger.info(f"Best ckpt to load: {self.load_ep}")
                self.load = True
            else:
                # load last epoch
                self.load_ep = ep_nums[-1]
                self.logger.info(f"Last ckpt to load: {self.load_ep}")
                self.load = True
        else:
            self.logger.info("No checkpoints found, starting from scratch.")

        # Per-epoch metrics where the average is not important.
        self.metrics.add_meter(Metrics.TRAIN_EPOCH, use_avg=False)
        self.metrics.add_meter(Metrics.TIME_TOTAL, use_avg=False)
        self.metrics.add_meter(Metrics.TIME_VAL, use_avg=False)
        self.metrics.add_meter(Metrics.VAL_LOSS, use_avg=False)
        self.metrics.add_meter(Metrics.VAL_BEST_FIELD, use_avg=False)

        # Per-step metrics
        self.metrics.add_meter(Metrics.TRAIN_LR, per_step=True, use_avg=False)
        self.metrics.add_meter(Metrics.TRAIN_GRAD_CLIP, per_step=True, reset_avg_each_epoch=True)
        self.metrics.add_meter(Metrics.TRAIN_LOSS, per_step=True, reset_avg_each_epoch=True)

        # Per-step Memory-RAM Profiling
        self.metrics.add_meter(Metrics.PROFILE_GPU_MEM_USED, per_step=True)
        self.metrics.add_meter(Metrics.PROFILE_GPU_LOAD, per_step=True)
        self.metrics.add_meter(Metrics.PROFILE_RAM_USED, per_step=True)
        self.metrics.add_meter(Metrics.PROFILE_GPU_MEM_TOTAL, per_step=True, use_avg=False)
        self.metrics.add_meter(Metrics.PROFILE_RAM_TOTAL, per_step=True, use_avg=False)

        # Step-based metrics for time, we only care about the total average
        self.metrics.add_meter(Metrics.TIME_STEP_FORWARD, per_step=True, use_value=False)
        self.metrics.add_meter(Metrics.TIME_STEP_BACKWARD, per_step=True, use_value=False)
        self.metrics.add_meter(Metrics.TIME_STEP_TOTAL, per_step=True, use_value=False)
        self.metrics.add_meter(Metrics.TIME_STEP_OTHER, per_step=True, use_value=False)

        # compute steps per epoch
        self.train_loader_length = train_loader_length

        # The following fields must be set by the inheriting trainer. In special cases (like
        # multiple optimizers with GANs), override methods get_opt_state and set_opt_state instead.
        self.optimizer: Optional[Optimizer] = None
        self.lr_scheduler: Optional[lr_scheduler.LRScheduler] = None

        # setup timers and other stuff that does not need to be saved (temporary trainer state)
        self.timer_step: float = 0
        self.timer_step_forward: float = 0
        self.timer_step_backward: float = 0
        self.timer_train_start: float = 0
        self.timer_train_epoch: float = 0
        self.timer_val_epoch: float = 0
        self.timedelta_step_forward: float = 0
        self.timedelta_step_backward: float = 0
        self.steps_per_epoch: int = 0

    # ---------- Must override these for training and validation ----------

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """
        Training loop over epochs, including validation.

        Args:
            train_loader: Dataloader for training set.
            val_loader: Dataloader for validation set.
        """
        raise NotImplementedError

    @th.no_grad()
    def validate_epoch(self, val_loader: DataLoader, **kwargs) -> (
            Tuple[float, float, bool, Tuple[Dict[str, float], Any]]):
        """
        Validate a single epoch.

        Args:
            val_loader: Dataloader for validation set.
            **kwargs: Optional keyword arguments for validation.

        Returns:
            Tuple of validation loss, validation score, epoch is best flag and any custom metrics.
        """
        raise NotImplementedError

    # ---------- Optionally override these if you need more than one optimizer ----------

    def get_opt_state(self) -> Dict[str, Dict[str, nn.Parameter]]:
        """
        Return the current optimizer and scheduler state.

        Returns:
            Dictionary of optimizer and scheduler state dict.
        """
        return {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict()
        }

    def set_opt_state(self, opt_state: Dict[str, Dict[str, nn.Parameter]]) -> None:
        """
        Set the current optimizer and scheduler state from the given state.

        Args:
            opt_state: Dictionary of optimizer and scheduler state dict.
        """
        self.optimizer.load_state_dict(opt_state["optimizer"])
        self.lr_scheduler.load_state_dict(opt_state["lr_scheduler"])

    # ---------- Misc. public methods ----------
    def check_cuda(self):
        """
        Check the config if cuda is active.

        Returns:
            Flag whether cuda is active or not.
        """
        if self.cfg.use_cuda:
            return True
        return False

    def check_early_stop(self) -> bool:
        """
        Check if training should be stopped at this point.

        Returns:
            Whether or not training should be stopped.
        """
        # this is called after post epoch hook which increased the epoch counter, so subtract one
        current_epoch = self.state.current_epoch - 1
        # find best epoch
        best_epoch = self.exp.find_best_epoch()
        if best_epoch == -1:
            # no validation done yet, assume current one is the best epoch
            best_epoch = current_epoch
        # calculate number of bad epochs
        bad_epochs = current_epoch - best_epoch
        # log infos
        self.logger.info(
                f"Experiment ---------- {self.exp.exp_group}/{self.exp.exp_name}/{self.exp.run_name} "
                f"---------- epoch current/best/bad: {current_epoch}/{best_epoch}/{bad_epochs}")
        if bad_epochs >= self.cfg.val.det_best_terminate_after:
            # stop early
            self.logger.info(f"No improvement since {bad_epochs} epochs, end of training.")
            return True
        # keep going
        return False

    def check_is_val_epoch(self) -> bool:
        """
        Check if validation is needed at the end of training epochs.

        Returns:
            Whether or not validation is needed.
        """
        # check if we need to validate
        do_val = (self.state.current_epoch % self.cfg.val.val_freq == 0
                  and self.cfg.val.val_freq > -1
                  and self.state.current_epoch >= self.cfg.val.val_start)
        # always validate the last epoch
        do_val = do_val or self.state.current_epoch == self.cfg.train.num_epochs
        return do_val

    def check_is_new_best(self, result: float) -> bool:
        """
        Check if the given result improves over the old best.

        Args:
            result: Validation result to compare with old best.

        Returns:
            Whether or not the result improves over the old best.
        """
        old_best = self.state.det_best_field_best

        # check if this is a new best
        is_best = self._check_if_current_score_is_best(result, old_best)

        # log info
        old_best_str = f"{old_best:.5f}" if old_best is not None else "NONE"
        self.logger.info(f"***** Improvement: {is_best} *****. Before: {old_best_str}, "
                         f"After {result:.5f}, Field: {self.cfg.val.det_best_field}, "
                         f"Mode {self.cfg.val.det_best_threshold_mode}")

        # update fields
        self.state.det_best_field_current = result
        if is_best:
            self.state.det_best_field_best = result

        return is_best

    def close(self) -> None:
        """
        Close logger and metric writers.
        """
        utils.remove_handlers_from_logger(self.logger)
        self.metrics.close()

    # ---------- Public hooks that run once per experiment ----------

    def hook_post_init(self) -> None:
        """
        Hook called after trainer init is done. Loads the correct epoch.
        """
        if self.load:
            assert not self.model_mgr.was_loaded, (
                    f"Error: Loading epoch {self.load_ep} but already weights have been loaded. "
                    f"If you load weights for warmstarting, you cannot run if the experiments "
                    f"has already saved checkpoints. Change the run name "
                    f"or use --reset to delete the experiment run.")
            if self.load_model:
                # load model from file. used for validation or
                # to start training from pretrained checkpoint.
                self.logger.info(f"Loading model from checkpoint file {self.load_model}")
                model_state = th.load(str(self.load_model))
                self.model_mgr.set_model_state(model_state)
            else:
                # load model given an epoch. also reload metrics and
                # optimization to correctly continue training.
                self.logger.info(f"Loading Ep {self.load_ep}.")
                self._load_checkpoint(self.load_ep)
                if not self.is_test:
                    # In training, add 1 to current epoch after loading since if we loaded epoch N,
                    # we are training epoch N+1 now. In validation, we are validating epoch N.
                    self.state.current_epoch += 1

    def hook_pre_train(self) -> None:
        """
        Hook called on training start. Remember start epoch, time the start, log info.
        """
        self.state.start_epoch = self.state.current_epoch
        self.timer_train_start = timer()
        self.logger.info(f"Training from {self.state.current_epoch} to {self.cfg.train.num_epochs}")
        self.logger.info("Training Models on devices " + ", ".join([
                f"{key}: {val.__class__.__name__} {next(val.parameters()).device}"
                for key, val in self.model_mgr.model_dict.items()]))

    def hook_post_train(self) -> None:
        """
        Hook called on training finish. Log info on total num epochs trained and duration.
        """
        self.logger.info(f"In total, training {self.state.current_epoch} epochs took "
                         f"{self.state.time_total:.3f}s "
                         f"({self.state.time_total - self.state.time_val:.3f}s "
                         f"train / {self.state.time_val:.3f}s val)")

    # ---------- Public hooks that run every epoch ----------

    def hook_pre_train_epoch(self) -> None:
        """
        Hook called before training an epoch.
        Set models to train, start timing start, reset meters, log info.
        """
        # set model to train mode
        self.model_mgr.set_all_models_train()
        # start timers
        self.timer_train_epoch = timer()
        self.timer_step = timer()
        # clear metrics
        self.metrics.hook_epoch_start()
        # log info
        self.logger.info(f"{str(datetime.datetime.now()).split('.')[0]} ---------- "
                         f"Training epoch: {self.state.current_epoch}")

    def hook_pre_val_epoch(self) -> None:
        """
        Hook called before validating an epoch. Set models to val, start timing.
        """
        # set models to validation mode
        self.model_mgr.set_all_models_eval()
        # start validation epoch timer
        self.timer_val_epoch = timer()
        #
        self.timer_step = timer()

    def hook_post_val_epoch(self, val_loss: float, is_best: bool) -> None:
        """
        Hook called after validation epoch is done. Updates basic validation meters.

        Args:
            val_loss: Validation loss.
            is_best: Whether this is a new best epoch.
        """
        # update validation timer
        self.state.time_val += timer() - self.timer_val_epoch

        # update loss and result
        self.metrics.update_meter(Metrics.VAL_LOSS, val_loss)
        self.metrics.update_meter(Metrics.VAL_BEST_FIELD, self.state.det_best_field_current)

        # update info dict for reloading
        self.state.infos_val_epochs.append(self.state.current_epoch)
        self.state.infos_val_steps.append(self.state.total_step)
        self.state.infos_val_is_good.append(is_best)

    def hook_post_train_and_val_epoch(self, is_val: bool, has_improved: bool) -> None:
        """
        Hook called after entire epoch (training + validation) is done.

        Args:
            is_val: Whether there was validation done this epoch.
            has_improved: If there was validation, whether there was an improvement (new best).
        """
        # update total timer
        self.state.time_total += timer() - self.timer_train_epoch

        # step LR scheduler after end of epoch
        if self.lr_scheduler is not None:
            self.lr_scheduler.step_epoch(is_val, has_improved)

        # log metrics
        self.metrics.update_meter(Metrics.TIME_TOTAL, self.state.time_total)
        self.metrics.update_meter(Metrics.TIME_VAL, self.state.time_val)
        self.metrics.update_meter(Metrics.TRAIN_EPOCH, self.state.current_epoch)

        # display step times
        fields = [Metrics.TIME_STEP_FORWARD, Metrics.TIME_STEP_BACKWARD, Metrics.TIME_STEP_OTHER]
        time_total = self.metrics.meters[Metrics.TIME_STEP_TOTAL].avg
        time_str_list = ["Step time: Total", f"{time_total * 1000:.0f}ms"]
        for field in fields:
            time_value = self.metrics.meters[field].avg
            time_name_short = str(field).split("/")[-1].split("_")[-1]
            time_str_list += [time_name_short, f"{time_value * 1000:.2f}ms",
                              f"{time_value / time_total:.1%}"]
        self.logger.info(" ".join(time_str_list))

        # feed step-based metrics to tensorboard and collector
        self.metrics.feed_metrics(False, self.state.total_step, self.state.current_epoch)

        # save checkpoint and metrics
        self._save_checkpoint()

        # cleanup files depending on saving config (default only keeps best and last epoch)
        self._cleanup_files()

        # increase epoch counter
        self.state.current_epoch += 1

    # ---------- Public hooks that run every step ----------

    def hook_pre_step_timer(self) -> None:
        """
        Hook called before forward pass. Sets timer.
        """
        self.timer_step_forward = timer()

    def hook_post_forward_step_timer(self) -> None:
        """
        Hook called after forward pass, before backward pass. Compute time delta and sets timer.
        """
        self.timer_step_backward = timer()
        self.timedelta_step_forward = self.timer_step_backward - self.timer_step_forward

    def hook_post_backward_step_timer(self) -> None:
        """
        Hook called after backward pass. Compute time delta.
        """
        self.timedelta_step_backward = timer() - self.timer_step_backward

    def hook_post_step(
            self, epoch_step: int, loss: th.Tensor, lr: float, additional_log: Optional[str] = None,
            disable_grad_clip: bool = False) -> None:
        """
        Hook called after one optimization step.

        Profile gpu and update step-based meters. Feed everything to tensorboard.
        Needs some information to be passed down from the trainer for proper logging.

        Args:
            epoch_step: Current step in the epoch.
            loss: Training loss.
            lr: Training learning rate.
            additional_log: Additional string to print in the train step log.
            disable_grad_clip: Disable gradient clipping if it's done already somewhere else
        """
        # compute total time for this step and restart the timer
        total_step_time = timer() - self.timer_step
        self.timer_step = timer()

        # clip gradients
        total_norm = 0
        if self.cfg.train.clip_gradient > -1 and not disable_grad_clip:
            # get all parameters to clip
            _params, _param_names, params_flat = self.model_mgr.get_all_params()
            # clip using pytorch
            total_norm = clip_grad_norm_(params_flat, self.cfg.train.clip_gradient)
            if total_norm > self.cfg.train.clip_gradient:
                # print log message if gradients where clipped
                grad_clip_coef = self.cfg.train.clip_gradient / (total_norm + 1e-6)
                self.logger.info(f"Clipping gradient: {total_norm} with coef {grad_clip_coef}")
            total_norm = total_norm.item()
        self.state.last_grad_norm = total_norm

        # print infos
        if epoch_step % self.cfg.logging.step_train == 0:
            total_train_time = (timer() - self.timer_train_epoch) / 60
            str_step = ("{:" + str(len(str(self.steps_per_epoch))) + "d}").format(epoch_step)
            print_string = "".join([
                    f"E{self.state.current_epoch}[{str_step}/{self.steps_per_epoch}] "
                    f"T {total_train_time:.3f}m ",
                    f"LR {lr:.1e} L {loss:.4f} ",
                    f"Grad {self.state.last_grad_norm:.3e} "
                    if self.state.last_grad_norm != 0 else "",
                    f"{additional_log}" if additional_log is not None else ""])
            self.logger.info(print_string)

        # check GPU / RAM profiling
        if ((self.state.epoch_step % self.cfg.logging.step_gpu == 0
             and self.cfg.logging.step_gpu > 0) or
                self.state.epoch_step == self.cfg.logging.step_gpu_once
                and self.cfg.logging.step_gpu_once > 0):
            # get the current profile values
            (gpu_names, total_memory_per, used_memory_per, load_per, ram_total, ram_used, ram_avail
             ) = utils_torch.profile_gpu_and_ram()
            # average / sum over all GPUs
            gpu_mem_used: float = sum(used_memory_per)
            gpu_mem_total: float = sum(total_memory_per)
            # gpu_mem_percent: float = gpu_mem_used / gpu_mem_total
            load_avg: float = sum(load_per) / max(1, len(load_per))

            self.metrics.update_meter(Metrics.PROFILE_GPU_MEM_USED, gpu_mem_used)
            self.metrics.update_meter(Metrics.PROFILE_GPU_MEM_TOTAL, gpu_mem_total)
            self.metrics.update_meter(Metrics.PROFILE_GPU_LOAD, load_avg)
            self.metrics.update_meter(Metrics.PROFILE_RAM_USED, ram_used)
            self.metrics.update_meter(Metrics.PROFILE_RAM_TOTAL, ram_total)
            # # these 2 are not logged as they are redundant with the others.
            # self.metrics.update_meter(Metrics.PROFILE_GPU_MEM_PERCENT, gpu_mem_percent)
            # self.metrics.update_meter(Metrics.PROFILE_RAM_AVAILABLE, ram_avail)

            # log the values
            gpu_names_str = " ".join(set(gpu_names))
            multi_load, multi_mem = "", ""
            if len(load_per) > 1:
                multi_load = " [" + ", ".join(f"{load:.0%}" for load in load_per) + "]"
                multi_mem = " [" + ", ".join(f"{mem:.1f}GB" for mem in used_memory_per) + "]"
            self.logger.info(
                    f"RAM GB used/avail/total: {ram_used:.1f}/{ram_avail:.1f}/{ram_total:.1f} - "
                    f"GPU {gpu_names_str} Load: {load_avg:.1%}{multi_load} "
                    f"Mem: {gpu_mem_used:.1f}GB/{gpu_mem_total:.1f}GB{multi_mem}")

        # update timings
        other_t = total_step_time - self.timedelta_step_forward - self.timedelta_step_backward
        self.metrics.update_meter(Metrics.TIME_STEP_FORWARD, self.timedelta_step_forward)
        self.metrics.update_meter(Metrics.TIME_STEP_BACKWARD, self.timedelta_step_backward)
        self.metrics.update_meter(Metrics.TIME_STEP_TOTAL, total_step_time)
        self.metrics.update_meter(Metrics.TIME_STEP_OTHER, other_t)
        # update clipped gradient
        self.metrics.update_meter(Metrics.TRAIN_GRAD_CLIP, self.state.last_grad_norm)
        # update LR
        self.metrics.update_meter(Metrics.TRAIN_LR, lr)
        if (self.state.epoch_step % self.cfg.logging.step_train == 0
                and self.cfg.logging.step_train > 0):
            # loss update necessary
            self.metrics.update_meter(Metrics.TRAIN_LOSS, loss.item())

        # Save epoch step and increase total step counter
        self.state.epoch_step = epoch_step
        self.state.total_step += 1

        # feed step-based metrics to tensorboard and collector
        self.metrics.feed_metrics(True, self.state.total_step, self.state.current_epoch)

        # End of batch, step lr scheduler depending on flag
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    # ---------- Non-public methods ----------

    def _check_if_current_score_is_best(self, current: float, best: float) -> bool:
        """
        Compare given current and best, return True if current is better than best + threshold.
        Depending on config, smaller or bigger is better and threshold is absolute or relative.

        Args:
            current: Current score.
            best: Best score so far.

        Returns:
            Whether current is better than best by some threshold.
        """
        cp_mode = self.cfg.val.det_best_compare_mode
        th_mode = self.cfg.val.det_best_threshold_mode

        if best is None:
            # no best exists, so current is automatically better
            return True
        if cp_mode == MetricComparisonConst.VAL_DET_BEST_MODE_MIN:
            # smaller values are better
            if th_mode == MetricComparisonConst.VAL_DET_BEST_TH_MODE_REL:
                # must be relatively better by epsilon
                rel_epsilon = 1 - self.cfg.val.det_best_threshold_value
                return current < best * rel_epsilon
            if th_mode == MetricComparisonConst.VAL_DET_BEST_TH_MODE_ABS:
                # must be absolutely better by epsilon
                return current < best - self.cfg.val.det_best_threshold_value
            raise ValueError(f"Threshold mode for metric comparison not understood: {th_mode}")
        if cp_mode == MetricComparisonConst.VAL_DET_BEST_MODE_MAX:
            # bigger values are better
            if th_mode == MetricComparisonConst.VAL_DET_BEST_TH_MODE_REL:
                # must be relatively better by epsilon
                rel_epsilon = 1 + self.cfg.val.det_best_threshold_value
                return current > best * rel_epsilon
            if th_mode == MetricComparisonConst.VAL_DET_BEST_TH_MODE_ABS:
                # must be absolutely better by epsilon
                return current > best + self.cfg.val.det_best_threshold_value
            raise ValueError(f"Threshold mode for metric comparison not understood: {th_mode}")
        raise ValueError(f"Compare mode for determining best field not understood: {cp_mode}")

    def _save_checkpoint(self) -> None:
        """
        Save current epoch.
        """
        # trainer state
        trainerstate_file = self.exp.get_trainerstate_file(self.state.current_epoch)
        self.state.save(trainerstate_file)

        # metrics state
        self.metrics.save_epoch(self.state.current_epoch)

        # models
        models_file = self.exp.get_models_file(self.state.current_epoch)
        state = self.model_mgr.get_model_state()
        th.save(state, str(models_file))

        # optimizer and scheduler
        opt_file = self.exp.get_optimizer_file(self.state.current_epoch)
        opt_state = self.get_opt_state()
        th.save(opt_state, str(opt_file))

    def _load_checkpoint(self, epoch) -> None:
        """
        Load given epoch.
        """
        # trainer state
        trainerstate_file = self.exp.get_trainerstate_file(epoch)
        self.state.load(trainerstate_file)

        # metrics state
        self.metrics.load_epoch(epoch)

        # models
        models_file = self.exp.get_models_file(epoch)
        model_state = th.load(str(models_file))
        self.model_mgr.set_model_state(model_state)

        # optimizer and scheduler
        if not self.is_test:
            opt_file = self.exp.get_optimizer_file(self.state.current_epoch)
            opt_state = th.load(str(opt_file))
            self.set_opt_state(opt_state)
        else:
            self.logger.info("Don't load optimizer and scheduler during inference.")

    def _cleanup_files(self) -> None:
        """
        Delete epoch and info files to save space, depending on configuration.
        """
        ep_nums = self.exp.get_existing_checkpoints()
        if len(ep_nums) == 0:
            # no checkpoints exist
            return
        # always save best and last
        best_ep = self.exp.find_best_epoch()
        last_ep = ep_nums[-1]
        # remember which epochs have been cleaned up
        cleaned = []
        for ep_num in ep_nums:
            # always keep the best episode
            if ep_num == best_ep:
                continue
            # always keep the last episode
            if ep_num == last_ep:
                continue
            # if the save checkpoint frequency is set, some intermediate checkpoints should be kept
            if self.cfg.saving.keep_freq > 0:
                if ep_num % self.cfg.saving.keep_freq == 0:
                    continue
            # delete safely (don't crash if they don't exist for some reason)
            for file in [self.exp.get_models_file(ep_num), self.exp.get_optimizer_file(ep_num),
                         self.exp.get_trainerstate_file(ep_num),
                         self.exp.get_metrics_epoch_file(ep_num),
                         self.exp.get_metrics_step_file(ep_num)] + self.get_files_for_cleanup(
                    ep_num):
                if file.is_file():
                    os.remove(file)
                else:
                    self.logger.warning(f"Tried to delete {file} but couldn't find it.")
            cleaned.append(ep_num)
        if len(cleaned) > 0:
            self.logger.debug(f"Deleted epochs: {cleaned}")

    def get_files_for_cleanup(self, _epoch: int) -> List[Path]:
        """
        Implement this in the child trainer.

        Args:
            _epoch: Epoch to cleanup

        Returns:
            List of files to cleanup.
        """
        return []
