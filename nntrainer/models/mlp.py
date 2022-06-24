"""
Fully connected network model.
"""
from functools import partial
from typing import Any, Dict, List, Optional

from torch import nn

from nntrainer import models, typext, utils


class ResidualsEnum(typext.ConstantHolder):
    """
    Residuals.

    None: No residual.
    Passthrough: Pass input directly as the residual.
    Linear: Pass input to a Linear module (useful when dimensions don't fit)
    """
    NONE = utils.NONE
    PASSTHROUGH = "passthrough"
    LINEAR = "linear"


class MLPConfig(typext.ConfigClass):
    """
    MLP Submodule

    Args:
        config: Configuration dictionary to be loaded.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.output_dim: int = config.pop("output_dim")
        self.num_layers: int = config.pop("num_layers")
        self.hidden_dim: int = config.pop("hidden_dim")
        self.activation_middle = models.ActivationConfig(config.pop("activation_middle"))
        self.activation_output = models.ActivationConfig(config.pop("activation_output"))
        self.dropout_middle: float = config.pop("dropout_middle")
        self.dropout_output: float = config.pop("dropout_output")
        self.norm_middle = models.NormalizationConfig(config.pop("norm_middle"))
        self.norm_output = models.NormalizationConfig(config.pop("norm_output"))
        self.residual: str = config.pop("residual")


class MLP(nn.Module):
    """
    Multi-Layer Fully-Connected Network with lots of configurations options.

    An example using all options would be:
        Linear(d_in, d_hidden)
        Dropout(p_hidden)
        NormHidden
        ActivationHidden
        Linear(d_hidden, d_out)
        Dropout(p_out)
        Add residual input
        ActivationOutput
        NormOutput

    Notes:
        Doing e.g. LayerNorm inside the hidden layers is unusual and should not be done without
        experimenting whether it's good.
        Not really sure where to apply dropout exactly (?) Check Bert implementation on what exactly happens.
    """

    def __init__(self, input_dim, cfg: MLPConfig):
        super().__init__()

        # setup auto-hidden_dim
        if cfg.hidden_dim == 0:
            cfg.hidden_dim = cfg.output_dim

        activation_middle = partial(models.make_activation_module, cfg.activation_middle.name, cfg.activation_middle)
        activation_output = partial(models.make_activation_module, cfg.activation_output.name, cfg.activation_output)
        norm_middle = partial(models.make_normalization_module, cfg.hidden_dim, cfg.norm_middle.name, cfg.norm_middle)
        norm_output = partial(models.make_normalization_module, cfg.hidden_dim, cfg.norm_output.name, cfg.norm_output)

        assert cfg.num_layers > 0, "MLP with 0 layers"
        fc_layers: List[nn.Module] = []
        if cfg.num_layers == 1:
            # ---------- 1-layer ----------
            # linear input to output
            fc_layers.append(nn.Linear(input_dim, cfg.output_dim))
            # output dropout
            if cfg.dropout_output > 0:
                fc_layers.append(nn.Dropout(cfg.dropout_output))
        else:
            # ---------- N-layer, N>=2 ----------
            # first layer: linear input to hidden
            fc_layers.append(nn.Linear(input_dim, cfg.hidden_dim))
            # hidden dropout
            if cfg.dropout_middle > 0:
                fc_layers.append(nn.Dropout(cfg.dropout_middle))
            # hidden norm
            if cfg.norm_middle != utils.NONE:
                fc_layers.append(norm_middle())

            # loop all middle layers (all except first and last)
            for _n in range(1, cfg.num_layers - 1):
                # hidden activation
                if cfg.activation_middle != utils.NONE:
                    fc_layers.append(activation_middle())
                # linear hidden to hidden
                fc_layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
                # hidden dropout
                if cfg.dropout_middle > 0:
                    fc_layers.append(nn.Dropout(cfg.dropout_middle))
                # hidden norm
                if cfg.norm_middle != utils.NONE:
                    fc_layers.append(norm_middle())
            # last layer
            # another hidden activation
            if cfg.activation_middle != utils.NONE:
                fc_layers.append(activation_middle())
            # linear hidden to output
            fc_layers.append(nn.Linear(cfg.hidden_dim, cfg.output_dim))
            # output dropout
            if cfg.dropout_output > 0:
                fc_layers.append(nn.Dropout(cfg.dropout_output))

        # create stacked sequential out of the layers above
        self.mlp = nn.Sequential(*fc_layers)

        # output activation
        self.activation_output = None
        if cfg.activation_output != utils.NONE:
            self.activation_output = activation_output()

        # residual
        self.residual: Optional[nn.Module] = None
        if cfg.residual == ResidualsEnum.PASSTHROUGH:  # passthrough no-op layer
            self.residual = nn.Sequential()
            assert input_dim == cfg.output_dim, (
                f"Residual when input dim is {input_dim} and output dim is {cfg.output_dim} will crash.")
        elif cfg.residual == ResidualsEnum.LINEAR:  # single linear layer
            self.residual = nn.Linear(input_dim, cfg.output_dim)
        elif cfg.residual == ResidualsEnum.NONE:  # no residual
            pass
        else:
            raise ValueError(f"Unknown residual in MLP config: {cfg.residual}")

        # output norm
        self.norm_output = None
        if cfg.norm_output != utils.NONE:
            self.norm_output = norm_output()

    def forward(self, x):
        # apply MLP
        linear_out = self.mlp(x)

        # add residual
        if self.residual is not None:
            res = self.residual(x)
            linear_out += res

        # apply output activation
        if self.activation_output is not None:
            linear_out = self.activation_output(linear_out)

        # apply output normalization
        if self.norm_output is not None:
            linear_out = self.norm_output(linear_out)

        return linear_out
