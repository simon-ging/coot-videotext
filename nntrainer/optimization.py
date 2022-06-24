"""
Optimizers.
"""

import math
from typing import Dict, Iterable

import torch as th
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from nntrainer import typext


class OptimizerConst(typext.ConstantHolder):
    """
    Optimizer name constants.
    """
    ADAM = "adam"
    RADAM = "radam"


class OptimizerConfig(typext.ConfigClass):
    """
    Optimizer Configuration Class

    Args:
        config: Configuration dictionary to be loaded, optimizer part.
    """

    def __init__(self, config: Dict) -> None:
        self.name: str = config.pop("name")
        self.lr: float = config.pop("lr")
        self.weight_decay: float = config.pop("weight_decay")
        self.weight_decay_for_bias: bool = config.pop("weight_decay_for_bias")
        self.momentum: float = config.pop("momentum")
        self.sgd_nesterov: bool = config.pop("sgd_nesterov")
        self.adam_beta2: float = config.pop("adam_beta2")
        self.adam_eps: float = config.pop("adam_eps")
        self.adam_amsgrad: bool = config.pop("adam_amsgrad")
        self.radam_degentosgd: bool = config.pop("radam_degentosgd")
        self.lr_decay_mult: bool = config.pop("lr_decay_mult")


def make_optimizer(cfg: OptimizerConfig, params: Iterable[th.Tensor]) -> Optimizer:
    """
    Initializer optimizer given some configuration and parameters.

    Args:
        cfg: Optimizer type and hyperparameter configuration.
        params: Parameters to optimizer.

    Returns:
        Normalization function class.
    """
    if cfg.name == OptimizerConst.ADAM:
        optimizer: Optimizer = Adam(params, lr=cfg.lr, betas=(cfg.momentum, cfg.adam_beta2), eps=cfg.adam_eps,
                                    weight_decay=cfg.weight_decay, amsgrad=cfg.adam_amsgrad)
    elif cfg.name == OptimizerConst.RADAM:
        optimizer = RAdam(params, lr=cfg.lr, betas=(cfg.momentum, cfg.adam_beta2), eps=cfg.adam_eps,
                          weight_decay=cfg.weight_decay,
                          degenerated_to_sgd=cfg.radam_degentosgd)
    else:
        raise NotImplementedError(f"Unknown optimizer {cfg.name}")

    # apply special lr / weight decay if given by the model.
    lr = cfg.lr
    wd = cfg.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] =\
            wd * param_group['decay_mult']

    return optimizer


# ---------- Module implementation. ----------

class RAdam(Optimizer):
    """
    RAdam Optimizer from https://github.com/LiyuanLucasLiu/RAdam
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, degenerated_to_sgd=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(
                params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (
                        param['betas'][0] != betas[0] or param['betas'][1] !=
                        betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        'RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = th.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = th.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (
                            1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
                                    N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (
                                            1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'],
                                         p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class PlainRAdam(Optimizer):
    """
    PlainRAdam Optimizer from https://github.com/LiyuanLucasLiu/RAdam
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, degenerated_to_sgd=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        'RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = th.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = th.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'],
                                         p_data_fp32)
                    step_size = group['lr'] * math.sqrt(
                        (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
                                N_sma - 2) / N_sma * N_sma_max / (
                                N_sma_max - 2)) / (
                                        1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif self.degenerated_to_sgd:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'],
                                         p_data_fp32)
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p_data_fp32.add_(-step_size, exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss
