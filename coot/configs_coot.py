"""
Definition of constants and configurations for both retrieval and captioning.
"""

from nntrainer import typext


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
    CAPTIONING = "captioning"
    RETRIEVAL = "retrieval"


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


class CootNetworksConst(typext.ConstantHolder):
    """
    Store network names for COOT.
    """
    NET_VIDEO_LOCAL = "net_video_local"
    NET_VIDEO_GLOBAL = "net_video_global"
    NET_TEXT_LOCAL = "net_text_local"
    NET_TEXT_GLOBAL = "net_text_global"
