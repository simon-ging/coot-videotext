"""
Definition of constants and configurations for retrieval.
"""

import logging
import traceback
from typing import Any, Dict

from coot.loss_fn import ContrastiveLossConfig, LossesConst
from nntrainer import lr_scheduler, models, optimization, trainer_configs, typext, utils
from nntrainer.utils import ConfigNamesConst as Conf


class RetrievalConfig(trainer_configs.BaseExperimentConfig):
    """
    Definition to load the yaml config files for training a retrieval model. This is where the actual config dict
    goes and is processed.

    Args:
        config: Configuration dictionary to be loaded, logging part.
        is_train: Whether there will be training or not.
    """

    def __init__(self, config: Dict[str, Any], *, is_train: bool = True) -> None:
        super().__init__(config)
        self.name = "config_ret"
        self.dim_feat_global: int = config.pop("dim_feat_global", 768)
        self.dim_feat_local: int = config.pop("dim_feat_local", 384)
        if not is_train:
            # Disable dataset caching
            logger = logging.getLogger(utils.LOGGER_NAME)
            logger.debug("Disable dataset caching during validation.")
            config["dataset_val"]["preload_vid_feat"] = False
            config["dataset_val"]["preload_text_feat"] = False

        try:
            self.train = RetrievalTrainConfig(config.pop(Conf.TRAIN))
            self.val = RetrievalValConfig(config.pop(Conf.VAL))
            self.dataset_train = RetrievalDatasetConfig(config.pop(Conf.DATASET_TRAIN))
            self.dataset_val = RetrievalDatasetConfig(config.pop(Conf.DATASET_VAL))
            self.logging = trainer_configs.BaseLoggingConfig(config.pop(Conf.LOGGING))
            self.saving = trainer_configs.BaseSavingConfig(config.pop(Conf.SAVING))
            self.optimizer = optimization.OptimizerConfig(config.pop(Conf.OPTIMIZER))
            self.lr_scheduler = lr_scheduler.SchedulerConfig(config.pop(Conf.LR_SCHEDULER))
            self.model_cfgs = {}
            for key in RetrievalNetworksConst.values():
                self.model_cfgs[key] = models.TransformerConfig(config.pop(key))
        except KeyError as e:
            print()
            print(traceback.format_exc())
            print(f"ERROR: {e} not defined in config {self.__class__.__name__}\n")
            raise e

        self.post_init()


class RetrievalValConfig(trainer_configs.BaseValConfig):
    """
    Retrieval validation configuration class.

    Args:
        config: Configuration dictionary to be loaded, training part.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.val_clips: bool = config.pop("val_clips")
        assert isinstance(self.val_clips, bool)
        self.val_clips_freq: int = config.pop("val_clips_freq")
        assert isinstance(self.val_clips_freq, int)


class RetrievalTrainConfig(trainer_configs.BaseTrainConfig):
    """
    Retrieval trainer configuration class.

    Args:
        config: Configuration dictionary to be loaded, training part.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.loss_cycle_cons: float = config.pop("loss_cycle_cons")
        loss_config = config.pop("contrastive_loss_config")
        if self.loss_func == LossesConst.CONTRASTIVE:
            self.contrastive_loss_config = ContrastiveLossConfig(loss_config)


class RetrievalTrainerState(trainer_configs.BaseTrainerState):
    """
    This state will be saved together with models and optimizer.

    Put fields here that are required to be known during training.
    """
    # # Currently, no other fields are needed
    # another_field: float = 0


class RetrievalDatasetConfig(trainer_configs.BaseDatasetConfig):
    """
    Retrieval dataset configuration class.

    Args:
        config: Configuration dictionary to be loaded, dataset part.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.metadata_name: str = config.pop("metadata_name")
        self.vid_feat_name: str = config.pop("vid_feat_name")
        self.vid_feat_source: str = config.pop("vid_feat_source")
        self.vid_feat_dim: int = config.pop("vid_feat_dim")
        self.text_feat_name: str = config.pop("text_feat_name")
        self.text_feat_source: str = config.pop("text_feat_source")
        self.text_feat_dim: int = config.pop("text_feat_dim")
        self.min_frames: int = config.pop("min_frames")  # unused
        self.max_frames: int = config.pop("max_frames")
        self.use_clips: bool = config.pop("use_clips")  # unused
        self.min_clips: int = config.pop("min_clips")  # unused
        self.max_clips: int = config.pop("max_clips")  # unused
        self.include_background: bool = config.pop("include_background")  # unused
        self.add_stop_frame: int = config.pop("add_stop_frame")
        self.expand_segments: int = config.pop("expand_segments")
        self.frames_noise: float = config.pop("frames_noise")
        self.words_noise: float = config.pop("words_noise")
        self.text_preprocessing: str = config.pop("text_preprocessing")
        self.preload_vid_feat: bool = config.pop("preload_vid_feat")
        self.preload_text_feat: bool = config.pop("preload_text_feat")

        assert self.data_type == ExperimentTypesConst.RETRIEVAL
        assert isinstance(self.metadata_name, str)
        assert isinstance(self.vid_feat_name, str)
        assert isinstance(self.text_feat_name, str)
        assert isinstance(self.min_frames, int)
        assert isinstance(self.max_frames, int)
        assert isinstance(self.vid_feat_dim, int)
        assert isinstance(self.text_feat_dim, int)
        assert isinstance(self.use_clips, bool)
        assert isinstance(self.min_clips, int)
        assert isinstance(self.max_clips, int)
        assert isinstance(self.include_background, bool)
        assert isinstance(self.add_stop_frame, int)
        assert isinstance(self.expand_segments, int)
        assert isinstance(self.frames_noise, (int, float)) and self.frames_noise >= 0
        assert isinstance(self.words_noise, (int, float)) and self.words_noise >= 0
        assert isinstance(self.text_preprocessing, str)
        assert isinstance(self.preload_vid_feat, bool)
        assert isinstance(self.preload_text_feat, bool)


# ---------- Constants ----------

class DataTypesConst(typext.ConstantHolder):
    """
    Store config field values for COOT.
    """
    COOT_OUTPUT = "coot_output"
    RETRIEVAL = "retrieval"


class ExperimentTypesConst(typext.ConstantHolder):
    """
    Store model types for COOT.
    """
    RETRIEVAL = "retrieval"
    CAPTION = "caption"


class CootMetersConst(typext.ConstantHolder):
    """
    Additional metric fields.
    """
    TRAIN_LOSS_CC = "train/loss_cc"
    TRAIN_LOSS_CONTRASTIVE = "train/loss_contr"
    VAL_LOSS_CC = "val/loss_cc"
    VAL_LOSS_CONTRASTIVE = "val/loss_contr"
    RET_MODALITIES = ["vid2par", "par2vid", "cli2sen", "sen2cli"]
    RET_MODALITIES_SHORT = ["v2p", "p2v", "c2s", "s2c"]
    RET_METRICS = ["r1", "r5", "r10", "r50", "medr", "meanr"]


class RetrievalNetworksConst(typext.ConstantHolder):
    """
    Store network names for COOT.
    """
    NET_VIDEO_LOCAL = "net_video_local"
    NET_VIDEO_GLOBAL = "net_video_global"
    NET_TEXT_LOCAL = "net_text_local"
    NET_TEXT_GLOBAL = "net_text_global"
