"""
Definition of constants and configurations for captioning with MART.
"""

from typing import Any, Dict, Optional

from nntrainer import trainer_configs
from nntrainer.typext import ConstantHolder


# ---------- Path constants ----------
class MartPathConst(ConstantHolder):
    CACHE_DIR = "cache_caption"
    COOT_FEAT_DIR = "provided_embeddings"
    ANNOTATIONS_DIR = "annotations"
    VIDEO_FEATURE_DIR = "data/mart_video_feature"


# ---------- MART config ----------
class MartDatasetConfig(trainer_configs.BaseDatasetConfig):
    """
    MART Dataset Configuration class

    Args:
        config: Configuration dictionary to be loaded, dataset part.
    """

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.preload: bool = config.pop("preload")


class MartConfig(trainer_configs.BaseExperimentConfig):
    """
    Definition to load the yaml config files for training a MART captioning model.
    This is where the actual config dict goes and is processed.


    Args:
        config: Configuration dictionary to be loaded, logging part.

    Attributes:
        train: Training config
        val: Validation config
        dataset_train: Train dataset config
        dataset_val: Val dataset config
        logging: Log frequency

        max_n_sen: for recurrent, max number of sentences, 6 for anet, 12 for yc2
        max_n_sen_add_val: Increase max_n_sen during validation (default 10)
        max_t_len: max length of text (sentence or paragraph), 30 for anet, 20 for yc2
        max_v_len: max length of video feature (default 100)
        type_vocab_size: Size of sequence type vocabulary: video as 0, text as 1 (default 2)
        word_vec_size: GloVE embeddings size (default 300)

        coot_model_name: Model name to load embeddings from. (default None)
        coot_mode: Which COOT representations to input into the captioning model. (default vidclip, choices
            vid: only video, vidclip: video + clip, vidclipctx: video + clip + video context, clip: only clip)
        coot_dim_vid: video feature size (default 768)
        coot_dim_clip: clip feature size (default 384)
        video_feature_size: 2048 appearance + 1024 flow. Change depending on COOT embeddings:
            vidclip: coot_dim_vid + coot_dim_clip, clip: coot_dim_clip, etc. (default 3072)

        debug: Activate debugging. Unused / untested (default False)

        attention_probs_dropout_prob: Dropout on attention mask (default 0.1)
        hidden_dropout_prob: Dropout on hidden states (default 0.1)
        hidden_size: Model hidden size (default 768)
        intermediate_size: Model intermediate size (default 768)
        layer_norm_eps: Epsilon parameter for Layernorm (default 1e-12)
        max_position_embeddings: Position embeddings limit (default 25)
        memory_dropout_prob: Dropout on memory cells (default 0.1)
        num_attention_heads: Number of attention heads (default 12)
        num_hidden_layers: number of transformer layers (default 2)
        n_memory_cells: number of memory cells in each layer (default 1)
        share_wd_cls_weight: share weight matrix of the word embedding with the final classifier (default False)
        recurrent: Run recurrent model (default False)
        untied: Run untied model (default False)
        mtrans: Masked transformer model for single sentence generation (default False)
        xl: transformer xl model, enforces recurrent = True, since the data loading part is the same (default False)
        xl_grad: enable back-propagation for xl model, only useful when `-xl` flag is enabled.
            Note, the original transformerXL model does not allow back-propagation. (default False)
        use_glove: Disable loading GloVE embeddings. (default None)
        freeze_glove: do not train GloVe vectors (default False)
        model_type: This is inferred from the fields recurrent, untied, mtrans, xl, xl_grad

        label_smoothing: Use soft target instead of one-hot hard target (default 0.1)

        save_mode: all: save models at each epoch. best: only save the best model (default best, choices: all, best)
        use_beam: use beam search, otherwise greedy search (default False)
        beam_size: beam size (default 2)
        n_best: stop searching when get n_best from beam search (default 1)

        ema_decay: Use exponential moving average at training, float in (0, 1) and -1: do not use.
            ema_param = new_param * ema_decay + (1-ema_decay) * last_param (default 0.9999)
        initializer_range: Weight initializer range (default 0.02)
        lr: Learning rate (default 0.0001)
        lr_warmup_proportion: Proportion of training to perform linear learning rate warmup for.
            E.g., 0.1 = 10%% of training. (default 0.1)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.name = "config_ret"

        # mandatory groups, needed for nntrainer to work correctly
        self.train = trainer_configs.BaseTrainConfig(config.pop("train"))
        self.val = trainer_configs.BaseValConfig(config.pop("val"))
        self.dataset_train = MartDatasetConfig(config.pop("dataset_train"))
        self.dataset_val = MartDatasetConfig(config.pop("dataset_val"))
        self.logging = trainer_configs.BaseLoggingConfig(config.pop("logging"))
        self.saving = trainer_configs.BaseSavingConfig(config.pop("saving"))

        # more training
        self.label_smoothing: float = config.pop("label_smoothing")

        # more validation
        self.save_mode: str = config.pop("save_mode")
        self.use_beam: bool = config.pop("use_beam")
        self.beam_size: int = config.pop("beam_size")
        self.n_best: int = config.pop("n_best")
        self.min_sen_len: int = config.pop("min_sen_len")
        self.max_sen_len: int = config.pop("max_sen_len")
        self.block_ngram_repeat: int = config.pop("block_ngram_repeat")
        self.length_penalty_name: str = config.pop("length_penalty_name")
        self.length_penalty_alpha: float = config.pop("length_penalty_alpha")

        # dataset
        self.max_n_sen: int = config.pop("max_n_sen")
        self.max_n_sen_add_val: int = config.pop("max_n_sen_add_val")
        self.max_t_len: int = config.pop("max_t_len")
        self.max_v_len: int = config.pop("max_v_len")
        self.type_vocab_size: int = config.pop("type_vocab_size")
        self.word_vec_size: int = config.pop("word_vec_size")

        # dataset: coot features
        self.coot_model_name: Optional[str] = config.pop("coot_model_name")
        self.coot_dim_clip: int = config.pop("coot_dim_clip")
        self.coot_dim_vid: int = config.pop("coot_dim_vid")
        self.coot_mode: str = config.pop("coot_mode")
        self.video_feature_size: int = config.pop("video_feature_size")

        # technical
        self.debug: bool = config.pop("debug")

        # model
        self.attention_probs_dropout_prob: float = config.pop("attention_probs_dropout_prob")
        self.hidden_dropout_prob: float = config.pop("hidden_dropout_prob")
        self.hidden_size: int = config.pop("hidden_size")
        self.intermediate_size: int = config.pop("intermediate_size")
        self.layer_norm_eps: float = config.pop("layer_norm_eps")
        self.memory_dropout_prob: float = config.pop("memory_dropout_prob")
        self.num_attention_heads: int = config.pop("num_attention_heads")
        self.num_hidden_layers: int = config.pop("num_hidden_layers")
        self.n_memory_cells: int = config.pop("n_memory_cells")
        self.share_wd_cls_weight: bool = config.pop("share_wd_cls_weight")
        self.recurrent: bool = config.pop("recurrent")
        self.untied: bool = config.pop("untied")
        self.mtrans: bool = config.pop("mtrans")
        self.xl: bool = config.pop("xl")
        self.xl_grad: bool = config.pop("xl_grad")
        self.use_glove: bool = config.pop("use_glove")
        self.freeze_glove: bool = config.pop("freeze_glove")

        # optimization
        self.ema_decay: float = config.pop("ema_decay")
        self.initializer_range: float = config.pop("initializer_range")
        self.lr: float = config.pop("lr")
        self.lr_warmup_proportion: float = config.pop("lr_warmup_proportion")
        self.infty: int = config.pop("infty", 0)
        self.eps: float = config.pop("eps", 1e-6)

        # max position embeddings is calculated as the max joint sequence length
        self.max_position_embeddings: int = self.max_v_len + self.max_t_len

        # must be set manually as it depends on the dataset
        self.vocab_size: Optional[int] = None

        # assert the config is valid
        if self.xl:
            assert self.recurrent, "recurrent must be True if TransformerXL is used."
        if self.xl_grad:
            assert self.xl, "xl must be True when using xl_grad"
        assert not (self.recurrent and self.untied), "cannot be True for both"
        assert not (self.recurrent and self.mtrans), "cannot be True for both"
        assert not (self.untied and self.mtrans), "cannot be True for both"
        if self.share_wd_cls_weight:
            assert self.word_vec_size == self.hidden_size, (
                "hidden size has to be the same as word embedding size when "
                "sharing the word embedding weight and the final classifier weight")

        # infer model type
        if self.recurrent:  # recurrent paragraphs
            if self.xl:
                if self.xl_grad:
                    self.model_type = "xl_grad"
                else:
                    self.model_type = "xl"
            else:
                self.model_type = "re"
        else:  # single sentence
            if self.untied:
                self.model_type = "untied_single"
            elif self.mtrans:
                self.model_type = "mtrans_single"
            else:
                self.model_type = "single"

        self.post_init()


class MartMetersConst(ConstantHolder):
    """
    Additional metric fields.
    """
    TRAIN_LOSS_PER_WORD = "train/loss_word"
    TRAIN_ACC = "train/acc"

    VAL_LOSS_PER_WORD = "val/loss_word"
    VAL_ACC = "val/acc"
    GRAD = "train/grad"
