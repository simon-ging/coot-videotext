"""
LR Schedulers completely rewritten from scratch.

These fit better to some use cases than the PyTorch LR schedulers.

Features:
    All required information is passed to the schedulers:
        (total number of epochs, training steps per epoch, validation improvements)
    Option for warmup per step or per epoch included by default.

Private: InvSqRootWithWarmup, PolynomialLR, SGDWarmRestarts
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from torch.optim.optimizer import Optimizer

from nntrainer import typext, utils


def make_lr_scheduler(optimizer: Optimizer, cfg: SchedulerConfig, base_lr: float, num_epochs: int,
                      train_loader_length: int, logger: Optional[logging.Logger] = None) -> LRScheduler:
    """
    Create LR scheduler.

    Args:

        optimizer: Optimizer.
        cfg: Scheduler config.
        base_lr: Optimizer base LR.
        train_loader_length: Total number of steps per train epoch.
        num_epochs: Planned total number of epochs.
        logger: Logger to print LR scheduler infos to.

    Returns:
        LR Scheduler.
    """
    if logger is None:
        logger = logging.getLogger(utils.LOGGER_NAME)
    logger.info(f"LR Scheduler: {cfg.name} LR {base_lr} Epochs {num_epochs} "
                f"steps per epoch {train_loader_length}")

    # create scheduler
    if cfg.name == SchedulerConst.REDUCE_OPW:
        lr_sched: LRScheduler = NewROPWarmup(optimizer, base_lr, cfg, num_epochs, train_loader_length, logger)
    elif cfg.name == SchedulerConst.NONE:
        lr_sched = ConstantLR(optimizer, base_lr, cfg, num_epochs, train_loader_length, logger)
    else:
        raise ValueError(f"LR Scheduler unknown: {cfg.name}")
    return lr_sched


# ---------- Configuration ----------

class SchedulerConfig(typext.ConfigClass):
    """
    Scheduler Configuration Class

    Args:
        config: Configuration dictionary to be loaded, scheduler part.
    """

    def __init__(self, config: Dict) -> None:
        # scheduler name
        self.name: str = config.pop("name")
        # warmup can be enabled for all schedulers
        self.warmup_type: str = config.pop("warmup_type")
        self.warmup_epochs: int = config.pop("warmup_epochs")

        if self.name == SchedulerConst.REDUCE_OPW:
            # fields required for reduce on plateau scheduler
            self.rop_factor: float = config.pop("rop_factor")
            self.rop_patience: int = config.pop("rop_patience")
            self.rop_cooldown: int = config.pop("rop_cooldown")
            self.rop_min_lr_factor: float = config.pop("rop_min_lr_factor")


class SchedulerConst(typext.ConstantHolder):
    """
    Store lr scheduler names.
    """
    NONE = utils.NONE
    REDUCE_OPW = "reduce_opw"  # Reduce on Plateau with Warmup


class SchedulerWarmupConst(typext.ConstantHolder):
    """
    Store Warmup Types for the Reduce On Plateau Scheduler.

     Notes:
        STEP: Increase LR linearly every training step.
        EPOCH: Increase LR linearly, but change it only once at the start of epochs.
        NONE: No warmup.
    """
    NONE = utils.NONE
    STEP = "step"
    EPOCH = "epoch"


# ---------- Base Scheduler class ----------
class LRScheduler:
    """
    Base LR scheduler. Optimizer and this scheduler init must happen before checkpoint loading.

    Usage:
        After each training step, call method step. After each epoch, call method step_epoch.

    The current reference LR is saved in self.current_lr (corresponds to the optimizer LR parameter).
    The current LRs per parameter group are saved in self.current_lr_list (This is needed for when some parameters
        need a different learning rate, e.g. during fine-tuning. Usually this will be the reference LR times some
        factor.)

    Args:
        optimizer: Optimizer to schedule the LRs for.
        base_lr: Base LR for all parameters.
        cfg: Scheduler config.
        num_epochs: Planned total number of epochs, this is needed for e.g. Cosine scheduling.
        train_loader_length: Number of steps per training epoch: ceil(len_dataloader / batch_size), this is
            needed for warming up linearly each step.
        logger: Logger for debugging.
    """

    def __init__(
            self, optimizer: Optimizer, base_lr: float, cfg: SchedulerConfig,
            num_epochs: int, train_loader_length: int, logger: logging.Logger):
        # attach optimizer
        assert isinstance(optimizer, Optimizer), f"{type(optimizer).__name__} is not an Optimizer"
        self.optimizer: Optimizer = optimizer

        # save other args
        self.base_lr: float = base_lr
        self.cfg: SchedulerConfig = cfg
        self.num_epochs = num_epochs
        self.num_steps_per_train_epoch = train_loader_length
        self.logger = logger

        # init current and old lr
        self.current_lr: float = self.base_lr
        self.old_lr: float = self.base_lr

        # initialize learning rates in the optimizer
        self.base_lr_list: List[float] = []
        for group in optimizer.param_groups:
            assert "initial_lr" not in group, "Optimizer has already set initial_lr, is that an error?"
            group["initial_lr"] = group["lr"]
            self.base_lr_list.append(group["initial_lr"])

        # init current and old lr list
        self.current_lr_list = self.base_lr_list
        self.old_lr_list = self.base_lr_list
        self.current_global_step = -1
        self.current_epoch = -1
        self.step()
        self.step_epoch(False, False)

    # ---------- Methods to implement when inheriting ----------

    def get_lrs_from_step(self) -> Tuple[List[float], float]:
        """
        Get learning rates given the current global step.

        Returns:
            Tuple of:
                Learning rates per optimizer param group.
                Reference learning rate for logging.
        """
        raise NotImplementedError

    def get_lrs_from_epoch(self, is_val: bool, has_improved: bool) -> Tuple[List[float], float]:
        """
        Get learning rates given the current epoch.

        Args:
            is_val: Whether there was validation done this epoch.
            has_improved: If there was validation, whether there was an improvement (new best).

        Returns:
            Tuple of:
                Learning rates per optimizer param group.
                Reference learning rate for logging.
        """
        raise NotImplementedError

    # ---------- Public interface ----------

    def state_dict(self):
        """
        Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict):
        """
        Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self) -> None:
        """
        Scheduler step, called once after every training step.
        """
        # increase step counter
        self.current_global_step += 1

        # make sure scheduler and training stay synchronized, otherwise there will probably be strange silent bugs
        min_possible_train_step = self.current_epoch * self.num_steps_per_train_epoch
        max_possible_train_step = (self.current_epoch + 1) * self.num_steps_per_train_epoch
        assert min_possible_train_step < self.current_global_step <= max_possible_train_step, (
            f"Mismatch between scheduler step {self.current_global_step} and trainer step. Either scheduler.step() "
            f"and scheduler.step_epoch() weren't called properly (see LRScheduler docstring) "
            f"or the wrong number of steps per train epoch was "
            f"passed. Given that were in epoch {self.current_epoch} with {self.num_steps_per_train_epoch} train "
            f"steps per epoch, the current scheduler global step should be between {min_possible_train_step} and "
            f"{max_possible_train_step}")

        # check if we are still in warmup
        if self._is_warmup():
            # run warmup and don't run the scheduler
            self._apply_warmup()
            return

        # save old lrs and get new ones
        self.old_lr_list = self.current_lr_list
        self.current_lr_list, self.current_lr = self.get_lrs_from_step()

        # update lrs in the optimizer
        self._update_lrs()

    def step_epoch(self, is_val: bool, has_improved: bool) -> None:
        """
        Scheduler step, called once after every epoch.

        Args:
            is_val: Whether there was validation done this epoch.
            has_improved: If there was validation, whether there was an improvement (new best).
        """
        # increase epoch counter
        self.current_epoch += 1

        # check if we are still in warmup
        if self._is_warmup():
            # run warmup and don't run the scheduler
            self._apply_warmup()
            return

        # save old lrs and update new ones
        self.old_lr_list = self.current_lr_list
        self.current_lr_list, self.current_lr = self.get_lrs_from_epoch(is_val, has_improved)

        # update lrs in the optimizer
        self._update_lrs()

    def get_current_step_for_print(self) -> str:
        """
        Return current step and epoch as string.

        Returns:
            String representation of current global step.
        """
        # Represent epoch and step with fixed with for some nice-looking log.
        return ("E:{:" + str(len(str(self.num_epochs))) + "d} S:{:" + str(len(str(
            self.num_epochs * self.num_steps_per_train_epoch))) + "} (scheduler)").format(
            self.current_epoch, self.current_global_step)

    # ---------- Non-public methods ----------

    def _update_lrs(self) -> None:
        """
        Update learning rates in the optimizer.
        """
        # only update the optimizer if there has been a change in learning rates
        needs_update = False
        for old_lr, current_lr in zip(self.old_lr_list, self.current_lr_list):
            if old_lr != current_lr:
                needs_update = True
                break
        if not needs_update:
            return
        self.logger.debug(f"{self.get_current_step_for_print()} LR updated to {self.current_lr}")
        for param_group, lr in zip(self.optimizer.param_groups, self.current_lr_list):
            param_group["lr"] = lr

    def _is_warmup(self) -> bool:
        """
        Check if LR is currently still warming up.

        Returns:
            is_warmup Bool.
        """
        if self.cfg.warmup_type == SchedulerWarmupConst.NONE:
            return False
        assert self.cfg.warmup_type in [SchedulerWarmupConst.EPOCH, SchedulerWarmupConst.STEP], (
            f"Unknown warmup type {self.cfg.warmup_type}")
        return self.current_epoch < self.cfg.warmup_epochs

    def _apply_warmup(self) -> None:
        """
        Calculate LRs for warmup.
        """
        if self.cfg.warmup_type == SchedulerWarmupConst.EPOCH:
            # scale LR linearly with epochs
            factor = (self.current_epoch + 1) / max(self.cfg.warmup_epochs, 1)
        elif self.cfg.warmup_type == SchedulerWarmupConst.STEP:
            # scale LR linearly with steps
            factor = (self.current_global_step + 1) / (self.cfg.warmup_epochs * self.num_steps_per_train_epoch + 1)
            # correct factor such that the last step isn't > 1
            # factor = min(factor, 1.0)
        else:
            raise ValueError(f"Unknown warmup type {self.cfg.warmup_type}")

        self.current_lr = factor * self.base_lr
        self.old_lr_list = self.current_lr_list
        self.current_lr_list = [lr * factor for lr in self.base_lr_list]
        self._update_lrs()


# ---------- Module implementations ----------


class ConstantLR(LRScheduler):
    """
    Constant Learning Rate scheduler.

    Usage:
        After each training step, call method step. After each epoch, call method step_epoch.
    """

    def get_lrs_from_step(self) -> Tuple[List[float], float]:
        """
        Get learning rates given the current global step.

        Returns:
            Tuple of:
                Learning rates per optimizer param group.
                Reference learning rate for logging.
        """
        return self.base_lr_list, self.base_lr

    def get_lrs_from_epoch(self, _is_val: bool, _has_improved: bool) -> Tuple[List[float], float]:
        """
        Scheduler step, called once after every epoch.

        Args:
            _is_val: Whether there was validation done this epoch (Unused for this scheduler).
            _has_improved: If there was validation, whether there was an improvement (new best)
                (Unused for this scheduler).

        Returns:
            Tuple of:
                Learning rates per optimizer param group.
                Reference learning rate for logging.
        """
        return self.base_lr_list, self.base_lr


class NewROPWarmup(LRScheduler):
    """
    Reduce on Plateau scheduler.

    Usage:
        After each training step, call method step. After each epoch, call method step_epoch.


    Hyperparameters:
        factor: Reduction factor for reducing the LR.
        patience: Number of bad epochs before reducing.
        cooldown: Number of epochs to wait after reduction.
        min_lr: Minimum LR to reduce to.

    Args:
        optimizer: Optimizer to schedule the LRs for.
        base_lr: Base LR for all parameters.
        cfg: Scheduler config.
        num_epochs: Planned total number of epochs, this is needed for e.g. Cosine scheduling.
        train_loader_length: Number of steps per training epoch: ceil(len_dataloader / batch_size), this is
            needed for warming up linearly each step.
        logger: Logger for debugging.
    """

    def __init__(self, optimizer: Optimizer, base_lr: float, cfg: SchedulerConfig, num_epochs: int,
                 train_loader_length: int, logger: logging.Logger):
        self.reduce_steps = 0
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        super().__init__(optimizer, base_lr, cfg, num_epochs, train_loader_length, logger)

    def get_lrs_from_step(self) -> Tuple[List[float], float]:
        """
        Get learning rates given the current global step.

        Returns:
            Tuple of:
                Learning rates per optimizer param group.
                Reference learning rate for logging.
        """
        return self.current_lr_list, self.current_lr

    def get_lrs_from_epoch(self, is_val: bool, has_improved: bool) -> Tuple[List[float], float]:
        """
        Get learning rates given the current epoch.

        Args:
            is_val: Whether there was validation done this epoch.
            has_improved: If there was validation, whether there was an improvement (new best).

        Returns:
            Tuple of:
                Learning rates per optimizer param group.
                Reference learning rate for logging.
        """
        print_reduction_message = False
        if is_val:
            # validation was done, need to do the reducer checks.

            # check improvement
            if has_improved:
                # good epoch, reset counter
                self.num_bad_epochs = 0
            else:
                # bad epoch, increase counter
                self.num_bad_epochs += 1

            # check cooldown
            if self.cooldown_counter > 0:
                # cool down after the last reduction and ignore bad epochs while cooling down.
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0

            # check needs reduction

            if self.num_bad_epochs > self.cfg.rop_patience:
                # too many bad epochs, reduce learning rate
                self.reduce_steps += 1
                self.cooldown_counter = self.cfg.rop_cooldown
                self.num_bad_epochs = 0
                # only print a reduction message if LR hadn't been reduced to the minimum already
                if not self.cfg.rop_factor ** (self.reduce_steps - 1) < self.cfg.rop_min_lr_factor:
                    print_reduction_message = True

        # calculate LR factor
        factor = max(self.cfg.rop_factor ** self.reduce_steps, self.cfg.rop_min_lr_factor)
        new_lr = self.base_lr * factor

        # print some message on reduction
        if print_reduction_message:
            self.logger.info(f"{self.get_current_step_for_print()} On Plateau: Reduce LR to {new_lr}")

        # return learning rates given the factor
        return [lr * factor for lr in self.base_lr_list], new_lr
