"""
Model manager for retrieval.

COOT is 4 times dlbase.models.TransformerLegacy
"""

import torch as th
from torch.cuda.amp import autocast

from coot.dataset_retrieval import RetrievalDataBatchTuple as Batch
from coot.configs_retrieval import RetrievalConfig, RetrievalNetworksConst
from nntrainer import models, typext


class RetrievalVisualEmbTuple(typext.TypedNamedTuple):
    """
    Definition of computed visual embeddings

    Notes:
        vid_emb: Video embedding with shape (batch, global_emb_dim)
        clip_emb: Clip embedding with shape (total_num_clips, local_emb_dim)
        vid_context: Video context with shape (batch, local_emb_dim)
        clip_emb_reshaped: Clip embeddings reshaped for input to the global model
            with shape (batch, max_num_clips, local_emb_dim)
        clip_emb_mask: Mask for the reshaped Clip embeddings with shape (batch, max_num_clips)
        clip_emb_lens: Lengths of the reshaped Clip embeddings with shape (batch)
    """
    vid_emb: th.Tensor
    clip_emb: th.Tensor
    vid_context: th.Tensor
    clip_emb_reshape: th.Tensor
    clip_emb_mask: th.Tensor
    clip_emb_lens: th.Tensor


class RetrievalTextEmbTuple(typext.TypedNamedTuple):
    """
    Definition of computed text embeddings:

    Notes:
        par_emb: Paragraph embedding with shape (batch, global_emb_dim)
        sent_emb: Sentence embedding with shape (total_num_sents, local_emb_dim)
        par_context: Paragraph context with shape (batch, local_emb_dim)
        sent_emb_reshaped: Sentence embeddings reshaped for input to the global model
            with shape (batch, max_num_sents, local_emb_dim)
        sent_emb_mask: Mask for the reshaped sentence embeddings with shape (batch, max_num_sents)
        sent_emb_lens: Lengths of the reshaped sentence embeddings with shape (batch)
    """
    par_emb: th.Tensor
    sent_emb: th.Tensor
    par_context: th.Tensor
    sent_emb_reshape: th.Tensor
    sent_emb_mask: th.Tensor
    sent_emb_lens: th.Tensor


class RetrievalModelManager(models.BaseModelManager):
    """
    Interface to create the 4 coot models and do the forward pass.
    """

    def __init__(self, cfg: RetrievalConfig):
        super().__init__(cfg)
        # update config type hints
        self.cfg: RetrievalConfig = self.cfg

        # find out input dimensions to the network
        input_dims = {
            RetrievalNetworksConst.NET_VIDEO_LOCAL: cfg.dataset_val.vid_feat_dim,
            RetrievalNetworksConst.NET_VIDEO_GLOBAL: cfg.model_cfgs[RetrievalNetworksConst.NET_VIDEO_LOCAL].output_dim,
            RetrievalNetworksConst.NET_TEXT_LOCAL: cfg.dataset_val.text_feat_dim,
            RetrievalNetworksConst.NET_TEXT_GLOBAL: cfg.model_cfgs[RetrievalNetworksConst.NET_TEXT_LOCAL].output_dim,
        }

        # create the 4 networks
        for key in RetrievalNetworksConst.values():
            # load model config
            current_cfg: models.TransformerConfig = cfg.model_cfgs[key]
            # create the network
            if current_cfg.name == models.TransformerTypesConst.TRANSFORMER_LEGACY:
                # old transformer
                self.model_dict[key] = models.TransformerLegacy(current_cfg, input_dims[key])
            else:
                raise NotImplementedError(f"Coot model type {current_cfg.name} undefined")

    def encode_visual(self, batch: Batch) -> RetrievalVisualEmbTuple:
        """
        Encode visual features to visual embeddings.

        Args:
            batch: Data batch.

        Returns:
            Video embeddings tuple.
        """
        with autocast(enabled=self.is_autocast_enabled()):
            # reference models for easier usage
            net_vid_local = self.model_dict[RetrievalNetworksConst.NET_VIDEO_LOCAL]
            net_vid_global = self.model_dict[RetrievalNetworksConst.NET_VIDEO_GLOBAL]
            net_vid_local_config = self.cfg.model_cfgs[RetrievalNetworksConst.NET_VIDEO_LOCAL]
            net_vid_global_config = self.cfg.model_cfgs[RetrievalNetworksConst.NET_VIDEO_GLOBAL]

            # compute video context
            vid_context, _ = net_vid_local(batch.vid_feat, batch.vid_feat_mask, batch.vid_feat_len, None)
            if net_vid_global_config.use_context:
                # need context for cross-attention later
                if net_vid_global_config.name == models.TransformerTypesConst.RNN_LEGACY:
                    # need special context for RNN
                    vid_context_hidden = vid_context.unsqueeze(0)
                    vid_context_hidden = vid_context_hidden.repeat(
                            net_vid_global_config.selfatn.num_layers, 1, 1)
                else:
                    # otherwise regular context for attention
                    vid_context_hidden = vid_context
            else:
                # no context needed
                vid_context_hidden = None

            # compute clip embedding
            clip_emb, _ = net_vid_local(batch.clip_feat, batch.clip_feat_mask, batch.clip_feat_len, None)
            batch_size = len(batch.clip_num)
            max_clip_len = th.max(batch.clip_num)
            clip_feat_dim = net_vid_local_config.output_dim
            clip_emb_reshape = th.zeros((batch_size, max_clip_len, clip_feat_dim)).float()
            clip_emb_mask = th.ones((batch_size, max_clip_len)).bool()
            clip_emb_lens = th.zeros((batch_size,)).long()
            if self.cfg.use_cuda:
                clip_emb_reshape = clip_emb_reshape.cuda(non_blocking=True)
                clip_emb_mask = clip_emb_mask.cuda(non_blocking=True)
                clip_emb_lens = clip_emb_lens.cuda(non_blocking=True)
            pointer = 0
            for batch_num, clip_len in enumerate(batch.clip_num):
                clip_emb_reshape[batch_num, :clip_len, :] = clip_emb[pointer:pointer + clip_len, :]
                clip_emb_mask[batch_num, :clip_len] = 0
                clip_emb_lens[batch_num] = clip_len
                pointer += clip_len

            # compute video embedding
            vid_emb, _ = net_vid_global(clip_emb_reshape, clip_emb_mask, batch.clip_num, vid_context_hidden)
            return RetrievalVisualEmbTuple(
                vid_emb, clip_emb, vid_context, clip_emb_reshape, clip_emb_mask, clip_emb_lens)

    def encode_text(self, batch: Batch) -> RetrievalTextEmbTuple:
        """
        Encode text features to text embeddings.

        Args:
            batch: Batch data.

        Returns:
            Text embeddings tuple.
        """
        with autocast(enabled=self.is_autocast_enabled()):
            # reference models for easier usage
            net_text_local = self.model_dict[RetrievalNetworksConst.NET_TEXT_LOCAL]
            net_text_global = self.model_dict[RetrievalNetworksConst.NET_TEXT_GLOBAL]
            net_text_local_config = self.cfg.model_cfgs[RetrievalNetworksConst.NET_TEXT_LOCAL]
            net_text_global_config = self.cfg.model_cfgs[RetrievalNetworksConst.NET_TEXT_GLOBAL]

            # compute paragraph context
            par_context, _ = net_text_local(batch.par_feat, batch.par_feat_mask, batch.par_feat_len, None)
            if net_text_global_config.use_context:
                # need context for cross-attention later
                if net_text_global_config.name == models.TransformerTypesConst.RNN_LEGACY:
                    # need special context for RNN
                    par_gru_hidden = par_context.unsqueeze(0)
                    par_gru_hidden = par_gru_hidden.repeat(net_text_global_config.num_layers, 1, 1)
                else:
                    # otherwise regular context for attention
                    par_gru_hidden = par_context
            else:
                # no context needed
                par_gru_hidden = None

            # compute sentence embedding
            sent_emb, _ = net_text_local(batch.sent_feat, batch.sent_feat_mask, batch.sent_feat_len, None)
            batch_size = len(batch.sent_num)
            sent_feat_dim = net_text_local_config.output_dim
            max_sent_len = th.max(batch.sent_num)
            sent_emb_reshape = th.zeros((batch_size, max_sent_len, sent_feat_dim)).float()
            sent_emb_mask = th.ones((batch_size, max_sent_len)).bool()
            sent_emb_lens = th.zeros((batch_size,)).long()
            if self.cfg.use_cuda:
                sent_emb_reshape = sent_emb_reshape.cuda(non_blocking=True)
                sent_emb_mask = sent_emb_mask.cuda(non_blocking=True)
                sent_emb_lens = sent_emb_lens.cuda(non_blocking=True)
            pointer = 0
            for batch_num, sent_len in enumerate(batch.sent_num):
                sent_emb_reshape[batch_num, :sent_len, :] =\
                    sent_emb[pointer:pointer + sent_len, :]
                sent_emb_mask[batch_num, :sent_len] = 0
                sent_emb_lens[batch_num] = sent_len
                pointer += sent_len

            # compute paragraph embedding
            par_emb, _ = net_text_global(sent_emb_reshape, sent_emb_mask, batch.sent_num, par_gru_hidden)
            return RetrievalTextEmbTuple(par_emb, sent_emb, par_context, sent_emb_reshape, sent_emb_mask, sent_emb_lens)
