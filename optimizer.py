import math
from typing import List

import torch
from easydict import EasyDict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer


def get_optimizer(cfg: EasyDict, params: List):
    cfg_name = cfg.name
    if cfg.name == 'adam':
        optimizer = torch.optim.Adam(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg_name == 'radam':
        optimizer = RAdam(
            params, lr=cfg.lr, betas=(cfg.momentum, cfg.adam_beta2),
            eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
    else:
        raise Exception(f'Unknown Optimizer {cfg.name}')
    for param_group in optimizer.param_groups:
        if 'initial_lr' in param_group:
            raise ValueError
        param_group['initial_lr'] = param_group['lr']
    return optimizer


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
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
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

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
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    n_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    n_sma_max = 2 / (1 - beta2) - 1
                    n_sma = n_sma_max - 2 * state['step'] * beta2_t / (
                            1 - beta2_t)
                    buffered[1] = n_sma
                    if n_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (n_sma - 4) / (n_sma_max - 4) * (
                                    n_sma - 2) / n_sma * n_sma_max / (
                                    n_sma_max - 2)) / (
                                            1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size
                if n_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'],
                                         p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg,
                                         denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'],
                                         p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)
        return loss


# noinspection PyAttributeOutsideInit,PyUnresolvedReferences
class ReduceLROnPlateauWarmup(ReduceLROnPlateau):
    def __init__(self, optimizer: Optimizer, warmup_epochs, **kwargs):
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, **kwargs)
        self.base_lrs = []
        for group in optimizer.param_groups:
            self.base_lrs.append(group["lr"])
        self.step_rop(self.mode_worse, False, None)

    def step_rop(self, metrics, do_eval, epoch=None):
        assert epoch is None
        epoch = self.last_epoch + 1
        if epoch <= self.warmup_epochs:
            factor = epoch / self.warmup_epochs
            self.last_epoch = epoch
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * factor
        elif not do_eval:
            pass
        else:
            super().step(metrics, epoch=epoch)
