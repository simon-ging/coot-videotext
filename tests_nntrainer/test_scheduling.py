"""
Test learning rate scheduler.
"""

from typing import List, Optional, Tuple

import numpy as np
from torch import nn, optim

from nntrainer.lr_scheduler import (
    LRScheduler, SchedulerConfig, SchedulerConst, SchedulerWarmupConst, make_lr_scheduler)


# pylint: disable=unused-variable
def _setup_optimization() -> Tuple[nn.Module, optim.Optimizer, float]:
    """
    Create optimizer setup for testing.

    Returns:
        Tuple of Network, Optimizer, LR.
    """
    lin1 = nn.Linear(2, 2)
    lin2 = nn.Linear(2, 2)
    net = nn.Sequential(lin1, lin2)
    lr = 1e-3
    params = [{"params": lin1.parameters(), "lr": lr}, {"params": lin2.parameters(), "lr": lr * .1}]

    opt = optim.SGD(params, lr=lr)
    return net, opt, lr


def _test_scheduler(scheduler: LRScheduler, num_epochs: int, steps_per_train_epoch: int,
                    epoch_is_val: Optional[List[bool]] = None, epoch_has_improved: Optional[List[bool]] = None
                    ) -> List[float]:
    """
    Run given scheduler for some epochs and make sure the LRs in the optimizer get updated correctly.

    Args:
        scheduler: Scheduler to test.
        num_epochs: Number of epochs.
        steps_per_train_epoch: Steps per epoch.
        epoch_is_val: Optional simulator for whether the epoch was a validation epoch.
        epoch_has_improved: Optional simulator for whether there was improvement in this epoch.

    Returns:
        List of learning rates, one for each step.
    """
    lr = scheduler.base_lr
    opt = scheduler.optimizer
    # make sure LRs have been set correctly on scheduler creation
    assert np.all([group['lr'] == scheduler.current_lr / lr * group["initial_lr"] for group in opt.param_groups])

    # run optimization and save LRs
    save_lrs = [scheduler.current_lr]
    for ep in range(num_epochs):
        for step in range(steps_per_train_epoch):
            scheduler.step()
            save_lrs.append(scheduler.current_lr)
            assert np.all([group['lr'] == scheduler.current_lr / lr * group["initial_lr"]
                           for group in opt.param_groups])
        is_val = False if epoch_is_val is None else epoch_is_val[ep]
        has_improved = False if epoch_has_improved is None else epoch_has_improved[ep]
        scheduler.step_epoch(is_val, has_improved)
    return save_lrs


def test_const() -> None:
    """
    Test Constant LR scheduler.
    """
    # setup experiments: 2 parameter groups with different learning rates
    num_epochs = 8
    steps_per_train_epoch = 3
    print("-" * 50, "ConstantLR with step warmup")
    net, opt, lr = _setup_optimization()

    # create constant lr scheduler with warmup per step
    cfg = SchedulerConfig(
        {"name": SchedulerConst.NONE, "warmup_type": SchedulerWarmupConst.STEP, "warmup_epochs": 5})
    scheduler = make_lr_scheduler(opt, cfg, lr, num_epochs, steps_per_train_epoch)
    save_lrs = _test_scheduler(scheduler, num_epochs, steps_per_train_epoch)

    # now we should have linear warmup on each step from 0 to 15, then no more changes for steps 16-24
    assert np.all(save_lrs == [
        6.25e-05, 0.000125, 0.0001875, 0.00025, 0.0003125, 0.000375, 0.0004375, 0.0005, 0.0005625000000000001, 0.000625,
        0.0006875, 0.00075, 0.0008125000000000001, 0.000875, 0.0009375, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
        0.001, 0.001, 0.001])
    del net, opt, scheduler

    # create the same constant lr scheduler but with warmup per epoch
    print("-" * 50, "ConstantLR with epoch warmup")
    net, opt, lr = _setup_optimization()
    cfg = SchedulerConfig(
        {"name": SchedulerConst.NONE, "warmup_type": SchedulerWarmupConst.EPOCH, "warmup_epochs": 5})
    scheduler = make_lr_scheduler(opt, cfg, lr, num_epochs, steps_per_train_epoch)
    save_lrs = _test_scheduler(scheduler, num_epochs, steps_per_train_epoch)

    # now we should have linear warmup at the beginning of each epoch, then no changes for steps 16-24
    assert np.all(save_lrs == [
        0.0002, 0.0002, 0.0002, 0.0002, 0.0004, 0.0004, 0.0004, 0.0006, 0.0006, 0.0006, 0.0008, 0.0008, 0.0008, 0.001,
        0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])


def test_rop() -> None:
    """
    Test reduce on plateau scheduler.
    """

    # setup experiments: 2 parameter groups with different learning rates
    print("-" * 50, "ReduceOnPlateau with step warmup")
    num_epochs = 25
    steps_per_train_epoch = 3
    net, opt, lr = _setup_optimization()

    # simulate validation
    epoch_is_val = [False] * 3 + [True] * 22
    epoch_has_improved = [
        False, False, False, True, True,
        True, False, False, False, True,
        False, False, False, False, False,
        False, False, False, False, False,
        False, False, False, False, False]
    patience = 2
    cooldown = 1
    # what should happen is: warmup for 5 epochs, then reduction after the (patience + 1) bad epochs 7 to 9.
    # again reduction after bad epochs 11 to 13, 14 is cooldown, reduction after bad epochs 15 to 17 where finally
    # the min_lr_factor of 0.2 is met and there should be no more reduction after that (and no more log output).
    expected_result = [
        6.25e-05, 0.000125, 0.0001875, 0.00025, 0.0003125, 0.000375, 0.0004375, 0.0005, 0.0005625000000000001, 0.000625,
        0.0006875, 0.00075, 0.0008125000000000001, 0.000875, 0.0009375, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
        0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
        0.0005, 0.0005, 0.0005, 0.0005, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025,
        0.00025, 0.00025, 0.00025, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002,
        0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002]

    # create constant lr scheduler with warmup per step
    cfg = SchedulerConfig(
        {
            "name": SchedulerConst.REDUCE_OPW, "warmup_type": SchedulerWarmupConst.STEP, "warmup_epochs": 5,
            "rop_factor": 0.5, "rop_patience": patience, "rop_cooldown": cooldown, "rop_min_lr_factor": 0.2})

    scheduler = make_lr_scheduler(opt, cfg, lr, num_epochs, steps_per_train_epoch)
    save_lrs = _test_scheduler(scheduler, num_epochs, steps_per_train_epoch, epoch_is_val=epoch_is_val,
                               epoch_has_improved=epoch_has_improved)
    assert np.all(save_lrs == expected_result)


if __name__ == "__main__":
    test_const()
    test_rop()
