"""
Setup a simple 2-layer MLP experiment on the MNIST dataset.
"""

import logging
from functools import partial
from timeit import default_timer as timer
from typing import Dict, Optional, Tuple

import torch as th
from torch import nn
from torch.utils import data as th_data
from tqdm import tqdm

from nntrainer import lr_scheduler, models, optimization, trainer_base, trainer_configs, typext
from nntrainer.utils import ConfigNamesConst as Conf, TrainerPathConst as Paths


MNISTExperimentType = "mlp"


# ---------- Define configuration file ----------

class MLPNetConfig(typext.ConfigClass):
    """
    Simple MLP network config.

    Args:
        config: Configuration dictionary to be loaded.
    """

    def __init__(self, config: Dict) -> None:
        self.input_dim: int = config.pop("input_dim")
        self.num_classes: int = config.pop("num_classes")
        self.num_layers: int = config.pop("num_layers")
        self.activation = models.ActivationConfig(config.pop("activation"))
        self.hidden_dim: int = config.pop("hidden_dim")


class MLPMNISTExperimentConfig(trainer_configs.BaseExperimentConfig):
    """
    MLP MNIST experiment config file.

    Args:
        config: Configuration dictionary to be loaded.
    """

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.name = "config_default"
        self.train = trainer_configs.BaseTrainConfig(config.pop(Conf.TRAIN))
        self.val = trainer_configs.BaseValConfig(config.pop(Conf.VAL))
        self.dataset_train = trainer_configs.BaseDatasetConfig(config.pop(Conf.DATASET_TRAIN))
        self.dataset_val = trainer_configs.BaseDatasetConfig(config.pop(Conf.DATASET_VAL))
        self.logging = trainer_configs.BaseLoggingConfig(config.pop(Conf.LOGGING))
        self.saving = trainer_configs.BaseSavingConfig(config.pop(Conf.SAVING))
        self.optimizer = optimization.OptimizerConfig(config.pop(Conf.OPTIMIZER))
        self.lr_scheduler = lr_scheduler.SchedulerConfig(config.pop(Conf.LR_SCHEDULER))
        self.mlp = MLPNetConfig(config.pop("mlp"))


# ---------- Define model and model manager ----------

class MLPModel(nn.Module):
    """
    Create a very simple MLP.

    Args:
        cfg: Model configuration.
    """

    def __init__(self, cfg: MLPNetConfig):
        super().__init__()
        assert cfg.num_layers >= 2, f"Number of layers must be >= 2 but is: {cfg.num_layers}"

        activation_fn = partial(models.make_activation_module, cfg.activation.name, cfg.activation)

        module_list = [nn.Flatten(), nn.Linear(cfg.input_dim, cfg.hidden_dim), activation_fn()]
        for _ in range(cfg.num_layers - 2):
            module_list += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), activation_fn()]
        module_list += [nn.Linear(cfg.hidden_dim, cfg.num_classes)]
        self.net = nn.Sequential(*module_list)

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        return self.net(inputs)


class MLPModelManager(models.BaseModelManager):
    """
    Class to hold the MLP model and to do the forward pass.

    Args:
        cfg: Experiment config.
    """

    def __init__(self, cfg: MLPMNISTExperimentConfig):
        super().__init__(cfg)
        self.model_dict["mlp"] = MLPModel(cfg.mlp)

    def forward_pass(self, inputs: th.Tensor) -> th.Tensor:
        """
        Do forward pass.

        Args:
            inputs: Input with arbitrary shape.

        Returns:
            Predictions.
        """
        return self.model_dict["mlp"](inputs)


# ---------- Define Trainer ----------


class MLPMNISTTrainer(trainer_base.BaseTrainer):
    """
    Trainer for MLP on MNIST.

    Notes:
        The parent TrainerBase takes care of all the basic stuff: Setting up directories and logging,
        determining device and moving models to cuda, setting up checkpoint loading and metrics.

    Args:
        cfg: Loaded configuration instance.
        model_mgr: Model manager.
        exp_dir: Experiment group.
        exp_name: Experiment name.
        run_name: Experiment run.
        train_loader_length: Length of the train loader, required for some LR schedulers.
        log_dir: Directory to put results.
        log_level: Log level.
        logger: Logger.
        print_graph: Print graph and forward pass of the model.
        reset: Delete entire experiment and restart from scratch.
        load_best: Whether to load the best epoch (default loads last epoch to continue training).
        load_epoch: Whether to load a specific epoch.
        inference_only: Removes some parts that are not needed during inference for speedup.
    """

    def __init__(
            self, cfg: MLPMNISTExperimentConfig, model_mgr: MLPModelManager, exp_dir: str, exp_name: str, run_name: str,
            train_loader_length: int, *,
            log_dir: str = Paths.DIR_EXPERIMENTS, log_level: Optional[int] = None,
            logger: Optional[logging.Logger] = None, print_graph: bool = False, reset: bool = False,
            load_best: bool = False, load_epoch: Optional[int] = None, inference_only: bool = False):
        super().__init__(
            cfg, model_mgr, exp_dir, exp_name, run_name, train_loader_length, "mlpmnist", log_dir=log_dir,
            log_level=log_level, logger=logger, print_graph=print_graph, reset=reset, load_best=load_best,
            load_epoch=load_epoch, is_test=inference_only)
        # ---------- setup ----------

        # update type hints from base classes to inherited classes
        self.cfg: MLPMNISTExperimentConfig = self.cfg
        self.model_mgr: MLPModelManager = self.model_mgr

        # update trainer state if loading is requested
        if self.load:
            self.state.current_epoch = self.load_ep

        # ---------- loss ----------

        # contrastive
        assert self.cfg.train.loss_func == "crossentropy"
        self.loss_ce = nn.CrossEntropyLoss()

        # ---------- additional metrics ----------

        # metrics logged once per epoch, log only value
        for field in ("val_base/accuracy",):
            self.metrics.add_meter(field, use_avg=False)

        # create optimizer
        params, _param_names, _params_flat = self.model_mgr.get_all_params()
        self.optimizer = optimization.make_optimizer(self.cfg.optimizer, params)

        # create lr scheduler
        self.lr_scheduler = lr_scheduler.make_lr_scheduler(
            self.optimizer, self.cfg.lr_scheduler, self.cfg.optimizer.lr, self.cfg.train.num_epochs,
            self.train_loader_length, logger=self.logger)

        self.hook_post_init()

    def compute_ce_loss(self, predictions: th.Tensor, labels: th.Tensor) -> th.Tensor:
        """
        Compute cross entropy classification loss.

        Args:
            predictions: Class predictions with shape (batch_size, num_classes)
            labels: Labels with shape (batch_size)

        Returns:
            Scalar loss.
        """
        return self.loss_ce(predictions, labels)

    def train_model(self, train_loader: th_data.DataLoader, val_loader: th_data.DataLoader) -> None:
        """
        Training loop.

        Args:
            train_loader: Train dataloader.
            val_loader: Validation dataloader.
        """
        self.hook_pre_train()  # pre-training hook: time book-keeping etc.
        self.steps_per_epoch = len(train_loader)  # save length of epoch

        # ---------- Epoch Loop ----------
        for _epoch in range(self.state.current_epoch, self.cfg.train.num_epochs):
            # pre-epoch hook: set models to train, time book-keeping
            self.hook_pre_train_epoch()

            # check for early stopping
            if self.check_early_stop():
                break

            # ---------- Dataloader Iteration ----------
            for step, (inputs, labels) in enumerate(train_loader):
                # move data to cuda
                if self.check_cuda():
                    inputs = inputs.cuda(non_blocking=self.cfg.cuda_non_blocking)
                    labels = labels.cuda(non_blocking=self.cfg.cuda_non_blocking)

                self.hook_pre_step_timer()  # hook for step timing

                # ---------- forward pass ----------
                predictions = self.model_mgr.forward_pass(inputs)
                loss = self.compute_ce_loss(predictions, labels)

                self.hook_post_forward_step_timer()  # hook for step timing

                # ---------- backward pass ----------
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.hook_post_backward_step_timer()  # hook for step timing

                # post-step hook: gradient clipping, profile gpu, update metrics, count step, step LR scheduler, log
                self.hook_post_step(step, loss, self.lr_scheduler.current_lr)

            # ---------- validation ----------
            do_val = self.check_is_val_epoch()

            is_best = False
            if do_val:
                # do the validation
                _loss_total, _val_score, is_best, _acc_total = self.validate_epoch(val_loader)

            # post-epoch hook: step lr scheduler, time bookkeeping, save checkpoint and metrics
            self.hook_post_train_and_val_epoch(do_val, is_best)

        # show end of training log message
        self.hook_post_train()

    @th.no_grad()
    def validate_epoch(self, data_loader: th_data.DataLoader) -> (
            Tuple[float, float]):
        """
        Validate a single epoch.

        Args:
            data_loader: Dataloader for validation

        Returns:
            Tuple of:
                Validation loss.
                Validation score (= accuracy).
                Whether this is the best epoch so far.
                Validation accuracy.
        """
        self.hook_pre_val_epoch()  # pre val epoch hook: set models to val and start timers
        forward_time_total = 0
        loss_total: th.Tensor = 0.
        acc_total: th.Tensor = 0.

        # ---------- Dataloader Iteration ----------
        num_steps = 0
        num_datapoints = 0
        pbar = tqdm(total=len(data_loader), desc=f"Validating epoch {self.state.current_epoch}")
        for _step, (inputs, labels) in enumerate(data_loader):
            # move data to cuda
            if self.check_cuda():
                inputs = inputs.cuda(non_blocking=self.cfg.cuda_non_blocking)
                labels = labels.cuda(non_blocking=self.cfg.cuda_non_blocking)

            # pre-step hook, currently unused
            self.hook_pre_step_timer()  # hook for step timing

            # ---------- forward pass ----------
            predictions = self.model_mgr.forward_pass(inputs)
            loss_total += self.compute_ce_loss(predictions, labels)

            num_datapoints += len(predictions)
            acc_total += th.sum(th.argmax(predictions, dim=-1) == labels)

            self.hook_post_forward_step_timer()
            forward_time_total += self.timedelta_step_forward
            num_steps += 1

            pbar.update()
        pbar.close()

        # ---------- validation done, calculate metrics ----------
        loss_total = loss_total.item() / num_steps
        acc_total = acc_total.item() / num_datapoints
        forward_time_total /= num_steps

        assert self.cfg.val.det_best_field == "val_accuracy"
        val_score = acc_total
        is_best = self.check_is_new_best(val_score)
        self.hook_post_val_epoch(loss_total, is_best)
        # feed additional meters: validation accuracy
        self.metrics.update_meter("val_base/accuracy", acc_total)
        # log validation results
        self.logger.info(f"Acc.: {acc_total:.2%}, Loss: {loss_total:.5f}, Time: {timer() - self.timer_val_epoch:.3f}s, "
                         f"Per forward step: {forward_time_total * 1000:.3f}ms")

        return loss_total, val_score, is_best, acc_total
