"""
Base class for the model manager that handles generic model-related tasks.

This way, trainer and model can be separated in the code.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch as th
from torch import nn

import nntrainer.trainer_configs
from nntrainer.utils_torch import edit_moduledot_in_state_keys


class BaseModelManager:
    def __init__(self, cfg: nntrainer.trainer_configs.DefaultExperimentConfig):
        """
        Class to hold all models. This is not a nn.Module

        Args:
            cfg: Experiment config.
        """
        # this dict contains all the models
        self.model_dict: Dict[str, nn.Module] = {}
        self.was_loaded: bool = False
        self.cfg: nntrainer.trainer_configs.DefaultExperimentConfig = cfg
        self.is_train = True

    def is_autocast_enabled(self) -> bool:
        """
        Given train or val state and config, determine whether autocast should be enabled.

        Returns:
            Bool.
        """
        return self.cfg.fp16_train if self.is_train else self.cfg.fp16_val

    def get_all_params(self) -> Tuple[Any, Any, Any]:
        """
        Since there are multiple networks used by this trainer, this
        function can be used to get all the parameters at once.


        Returns:
            params, param_names, params_flat
        """
        # loop models and collect parameters
        params, param_names, params_flat = [], [], []
        for _model_name, model in self.model_dict.items():
            _params, _param_names, _params_flat = self.get_params_opt_simple(model)
            params.extend(_params)
            param_names.extend(_param_names)
            params_flat.extend(_params_flat)
        return params, param_names, params_flat

    def set_all_models_train(self) -> None:
        """
        Set all networks to train mode.
        """
        self.is_train = True
        for model in self.model_dict.values():
            model.train()

    def set_all_models_eval(self) -> None:
        """
        Set all networks to eval mode.
        """
        self.is_train = False
        for model in self.model_dict.values():
            model.eval()

    def get_model_state(self) -> Dict[str, Dict[str, th.Tensor]]:
        """
        Get all state dicts of all networks into a single variable

        Returns:
            Dict with model names and keys and state dict of the model as value.
        """
        return_dict = {}
        for model_name, model in self.model_dict.items():
            return_dict[model_name] = model.state_dict()
        return return_dict

    def set_model_state(self, state: Dict[str, Dict[str, th.Tensor]]) -> None:
        """
        Use the dict of state dicts created by get_model_state to load all network weights.

        Args:
            state: Dict with model names and keys and state dict of the model as value.
        """
        self.was_loaded = True

        # backwards compatibility to coot-videotext
        if isinstance(state, list):
            for i, model_name in enumerate(self.model_dict.keys()):
                print(f"Backward compatible loading for coot-videotext: {model_name}")
                this_state = state[i]
                new_state = {}
                for param_name, param in this_state.items():
                    for replace_from, replace_to in {
                            "input_norm.": "norm_input.",
                            "input_fc.": "input_fc.mlp.",
                            # "norm_input.gain": "norm_input.weight",
                            # "layer_normalization.gain": "layer_normalization.weight",
                            "pooler.genpool": "pooler.pools.0.genpool"
                    }.items():
                        param_name = param_name.replace(replace_from, replace_to)
                    new_state[param_name] = param
                new_state = edit_moduledot_in_state_keys(new_state, not self.cfg.use_multi_gpu)
                self.model_dict[model_name].load_state_dict(new_state)
            return
        # backwards compatibility to recurrent_transformer (original MART repository style checkpoints)
        if sorted(list(state.keys())) == ["epoch", "model", "model_cfg", "opt"]:
            state_dict = state["model"]
            print(
                    f"Backward compatible loading for recurrent_transformer epoch {state['epoch']} with "
                    f"{sum([np.product(param.shape) for param in state_dict.values()])} parameters")
            state_dict = edit_moduledot_in_state_keys(state_dict, not self.cfg.use_multi_gpu)
            self.model_dict['model'].load_state_dict(state_dict)
            return
        # newest version of loading. keys in the state correspond to keys in the model_dict.
        for model_name, state_dict in state.items():
            state_dict = edit_moduledot_in_state_keys(state_dict, not self.cfg.use_multi_gpu)
            self.model_dict[model_name].load_state_dict(state_dict)
            # sep = "\n"
            # print(f"Loaded model: {model_name}params:\n{sep.join(name for name in state_dict.keys())}")

    def get_params_opt_simple(self, model: nn.Module) -> (
            Tuple[List[Dict[str, Any]], List[str], List[th.Tensor]]):
        """
        Args:
            model: Model to get the parameters from.

        Returns:
            Tuple of:
                List of:
                    Dict of:
                        'params': The parameter
                        'decay_mult': Multiply weight decay with this factor
                        'lr_mult': Multiplay learning rate with this factor
                List of:
                    parameter names
                List of:
                    parameters
        """
        params_dict: Dict[str, th.Tensor] = dict(model.named_parameters())
        params, param_names, params_flat = [], [], []
        # print(cfg.training.representation)
        for key, value in params_dict.items():
            decay_mult = 1.0
            if self.cfg.optimizer.weight_decay_for_bias and 'bias' in key:
                decay_mult = 0.0
            params += [{
                    'params': value,
                    'decay_mult': decay_mult,
                    'lr_mult': 1.0
            }]
            param_names += [key]
            params_flat += [value]

        return params, param_names, params_flat
