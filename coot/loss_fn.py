"""
Loss functions.
"""

from typing import Callable, Dict

import torch as th
from torch import nn

from nntrainer import typext
from nntrainer.typext import INF


class LossesConst(typext.ConstantHolder):
    CONTRASTIVE = "contrastive"
    CROSSENTROPY = "crossentropy"


def cosine_sim(visual_emb: th.Tensor, text_emb: th.Tensor) -> th.Tensor:
    """
    Calculate cosine similarity.

    Args:
        visual_emb: Visual embedding with shape (num_datapoints, dim_embedding)
        text_emb: Text embedding with shape (num_datapoints, dim_embedding)

    Returns:
        Cosine similariies with shape (num_datapoints, num_datapoints)
    """
    return visual_emb.mm(text_emb.t())


class ContrastiveLossConfig(typext.ConfigClass):
    """
    Contrastive loss Configuration Class

    Args:
        config: Configuration dictionary to be loaded, saving part.
    """

    def __init__(self, config: Dict) -> None:
        self.margin: float = config.pop("margin")
        self.weight_high: float = config.pop("weight_high")
        self.weight_high_internal: float = config.pop("weight_high_internal")
        self.weight_low: float = config.pop("weight_low")
        self.weight_low_internal: float = config.pop("weight_low_internal")
        self.weight_context: float = config.pop("weight_context")
        self.weight_context_internal: float = config.pop("weight_context_internal")


class ContrastiveLoss(nn.Module):
    """
    Regular Contrastive Loss between 2 groups of embeddings
    """
    def __init__(self, margin: float, max_violation: bool = False, norm: bool = True, use_cuda: bool = True):
        super().__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.norm = norm
        self.max_violation = max_violation
        self.use_cuda = use_cuda

    def forward(self, im, s):
        """
        Inputs shape (batch, embed_dim)

        Args:
            im: Visual embeddings (batch, embed_dim)
            s: Text embeddings (batch, embed_dim)

        Returns:
        """
        # compute image-sentence score matrix - how close is im(y) to s(x)
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals, where there is just the margin left
        mask: th.Tensor = th.eye(scores.shape[0]).bool()
        if self.use_cuda:
            mask = mask.cuda(non_blocking=True)
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        if self.norm:
            return (cost_s.sum() + cost_im.sum()).div(im.shape[0] * s.shape[0])
        return cost_s.sum() + cost_im.sum()


def compute_mean_distance_l2(c, s):
    return th.mean((c - s) ** 2, dim=-1)


def compute_mean_distance_negative_l2(c, s):
    return -compute_mean_distance_l2(c, s)


class CycleConsistencyLoss(nn.Module):
    """
    Cycle Consistency Loss

    Default values are the resulted best
    """
    def __init__(self, num_samples: int = 1, compute_half_cycles: bool = False, use_cuda: bool = True,
                 verbose: bool = False, print_fn: Callable = print):
        super().__init__()
        self.compute_half_cycles = compute_half_cycles
        self.use_cuda = use_cuda
        self.print_fn = print_fn
        self.verbose = verbose
        self.num_samples = num_samples
        self.num_samples_tensor = (th.ones(1) * self.num_samples)
        if self.use_cuda:
            self.num_samples_tensor = self.num_samples_tensor.cuda(
                non_blocking=True)

        # define loss functions (currently L2)
        self.loss_distance_fn = compute_mean_distance_l2
        self.proximity_fn = compute_mean_distance_negative_l2
        self.proximity_mask_val = -INF

        self.softmax_temp = 1
        self.softmax = nn.Softmax(dim=-1)
        self.weight_index_simple = 1
        self.weight_index_gauss = 0
        self.lambda_index_gauss = 1
        self.var_denom_eps = 1e-8
        self.var_log_eps = 1

    def forward(self, clip_emb: th.FloatTensor, clip_mask: th.BoolTensor, clip_lens: th.LongTensor,
                sent_emb: th.FloatTensor, sent_mask: th.BoolTensor, sent_lens: th.LongTensor):
        """
        Args:
            clip_emb: (batch_size, num_clips, feat_dim)
            clip_mask: (batch_size, num_clips), False = real, True = masked
            clip_lens: (batch_size), corresponds to mask
            sent_emb: (batch_size, num_sents, feat_dim)
            sent_mask: (batch_size, num_sents), False = real, True = masked
            sent_lens: (batch_size), corresponds to mask

        Returns:
            CC clip loss, CC sentence loss
        """
        # Invert masks here s.t. padded sequence elements are 0
        clip_mask = ~clip_mask
        sent_mask = ~sent_mask

        # Get maximum of the sequence lengths
        clip_max_len = clip_mask.shape[1]
        sent_max_len = sent_mask.shape[1]

        # go from clips to sentences
        clip_sent_nn, clip_alpha, clip_alpha_raw = self.get_soft_nn(clip_emb, clip_mask, sent_emb, sent_mask)

        # calculate loss clips to sentences
        clip_sent_loss = None
        if self.compute_half_cycles:
            clip_sent_loss = self.get_total_loss(clip_emb, clip_sent_nn, clip_mask, clip_lens, clip_max_len, clip_alpha,
                                                 clip_alpha_raw)

        # go from those new sentences back to clips
        clip_clip_nn, clip_beta, clip_beta_raw = self.get_soft_nn(clip_sent_nn, clip_mask, clip_emb, clip_mask)

        # calculate loss on clip cycle consistency
        clip_clip_loss = self.get_total_loss(clip_emb, clip_clip_nn, clip_mask, clip_lens, clip_max_len, clip_beta,
                                             clip_beta_raw)

        # go from sentences to clips
        sent_clip_nn, sent_alpha, sent_alpha_raw = self.get_soft_nn(sent_emb, sent_mask, clip_emb, clip_mask)

        # calculate loss sentences to clips
        sent_clip_loss = None
        if self.compute_half_cycles:
            sent_clip_loss = self.get_total_loss(sent_emb, sent_clip_nn, sent_mask, sent_lens, sent_max_len, sent_alpha,
                                                 sent_alpha_raw)

        # go from those new clips back to sentences
        sent_sent_nn, sent_beta, sent_beta_raw = self.get_soft_nn(sent_clip_nn, sent_mask, sent_emb, sent_mask)

        # calculate loss on sentence cycle consistency
        sent_sent_loss = self.get_total_loss(sent_emb, sent_sent_nn, sent_mask, sent_lens, sent_max_len, sent_beta,
                                             sent_beta_raw)

        return clip_clip_loss, sent_sent_loss, clip_sent_loss, sent_clip_loss

    def get_mxn_repr(self, source_emb, source_mask, target_emb, target_mask):
        """
        Unsqueeze tensors and modify the mask accordingly to do N*M
        computations on sequence lengths N and M.
        Used to e.g. calculate distance between all source and all target
        embeddings.

        Args:
            source_emb: (batch_size, len_seq_source, feat_dim)
            source_mask: (batch_size, len_seq_source), 1 = real, 0 = masked
            target_emb: (batch_size, len_seq_target, feat_dim)
            target_mask: (batch_size, len_seq_target), 1 = real, 0 = masked

        Returns:
            source_rep, target_rep, total_mask
        """
        # unsqueeze source_emb in 2nd dimension
        source_rep = source_emb.unsqueeze(2)

        # unsqueeze target_emb in 1st dimension
        target_rep = target_emb.unsqueeze(1)

        # build mask that is 0 whenever either source or target mask is 0
        # only source mask is NOT enough for soft NN (tested)
        total_mask = source_mask.unsqueeze(2) & target_mask.unsqueeze(1)

        return source_rep, target_rep, total_mask

    def get_soft_nn(self, source_emb, source_mask, target_emb, target_mask):
        """
        Find soft nearest neighbors of each source_emb, looking for
        neighbors in target_emb.

        Args:
            source_emb: (batch_size, len_seq_source, feat_dim)
            source_mask: (batch_size, len_seq_source), 1 = real, 0 = masked
            target_emb: (batch_size, len_seq_target, feat_dim)
            target_mask: (batch_size, len_seq_target), 1 = real, 0 = masked

        Returns:
            soft_nn: (batch_size, len_seq_source) one nearest neighbor in the
                target space for each embedding in the source space
            weights: (batch_size, len_seq_source, len_seq_target) softmax
                output of similarity between each source and target pair.
                determines how much weight is given to each target embedding
                when calculating the nearest neighbor for a given source
                embedding.
            distance: (batch_size, len_seq_source, len_seq_target) unnormalized
                similarity weight (useful for e.g. crossentropyloss that
                expects unnormalized probabilities)
        """
        # get representation that allows to work on all
        # possible combinations of source and taret
        source_rep, target_rep, total_mask = self.get_mxn_repr(source_emb, source_mask, target_emb, target_mask)

        # calculate some distance on all combinations at once
        # in this case, negative L2 distance as measure of proximity
        distance = self.proximity_fn(source_rep, target_rep)
        # shape (batch_size, num_clips, num_clips)
        # d holds distances (batch_size, source_num, target_num)

        # set masked distances to (almost) negative infinity
        distance.masked_fill_(~total_mask, self.proximity_mask_val)
        # shape (batch_size, source_max_len, target_max_len)
        # masked values are set to very high negative number for softmax

        # calculate weights with softmax and some temperature
        # higher temp: uniform dist. lower temp: hard argmax
        weights_alpha = self.softmax(distance / self.softmax_temp)

        # with weights, calculate soft nearest neighbor in target
        # embedding space
        soft_nn = target_emb.unsqueeze(dim=1) * weights_alpha.unsqueeze(dim=3)
        soft_nn = th.sum(soft_nn, dim=2)

        return soft_nn, weights_alpha, distance

    # pylint: disable=unused-argument
    def get_total_loss(self, emb_orig, emb_nn, emb_mask, emb_lens, emb_max_len, beta, beta_raw):
        """
        Given embeddings and their cycled nearest neighbors,
        calculate total loss given the config flags

        Args:
            emb_orig: (batch_size, len_seq, feat_dim)
            emb_nn: (batch_size, len_seq, feat_dim)
            emb_mask: (batch_size, len_seq), 1 = real, 0 = masked
            emb_lens: (batch_size), corresponds to mask
            emb_max_len: int, th.max over emb lens dim -1
            beta: (batch_size, len_seq, len_seq) softmax weights
            beta_raw: (batch_size, len_seq, len_seq) similarity scores before
                softmax

        Returns:
            float loss
        """
        l_seq = th.zeros_like(emb_mask).float()
        batch_size, _ = emb_mask.shape
        if self.use_cuda:
            l_seq = l_seq.cuda(non_blocking=True)
        if self.weight_index_gauss != 0 or self.weight_index_simple != 0:
            (loss_simple_per_seq, loss_gauss_per_seq, var_reg_per_seq) = self.compute_loss_index_gauss(
                emb_mask, emb_lens, emb_max_len, beta)
            l_seq += (loss_gauss_per_seq + var_reg_per_seq) * self.weight_index_gauss
            l_seq += loss_simple_per_seq * self.weight_index_simple

        # subsample loss if requested
        if self.num_samples != -1:
            # check max amount of samples possible (depends on number of clips)
            n_samp = th.min(emb_lens, self.num_samples_tensor)
            # draw n_samp random integers without replacement in range emb_lens
            total_loss = 0
            for _batch, (c_loss, c_mask, c_nsamp) in enumerate(zip(l_seq, emb_mask, n_samp)):
                idx = th.multinomial(c_mask.float(), int(c_nsamp))
                total_loss += c_loss[idx].mean()
            total_loss /= batch_size
        else:
            # no subsampling, average over all losses
            total_loss = (l_seq.sum(dim=-1) / emb_lens).mean(dim=-1)

        return total_loss

    def compute_loss_index_gauss(self, emb_mask, _emb_lens, emb_max_len, beta):
        """
        Compute distance between original index and soft index.
        Takes into account variance between original and soft index.
        Also returns the version without variance.

        Returns total loss and loss per sequence / per batch, to be able
        to sample only some of the losses.

        Args:
            emb_mask: value mask (batch, seq_len), 1 = real value, 0 = masked
            _emb_lens: unused lengths of sequence, shape (batch)
            emb_max_len: th.max over emb_lens dim -1
            beta: softmax weight used to calculate the nearest neighbor
                (batch, seq_len, seq_len): dim 1 is the nearest neighbors in the
                sentence space the computation started, dim 2 is the original
                embeddings

        Returns:
            loss_gauss, loss_simple, loss_gauss_per_batch,
            loss_simple_per_batch, loss_gauss_per_seq, loss_simple_per_seq
        """
        # original index = arange
        idx_orig = th.arange(emb_max_len)
        if self.use_cuda:
            idx_orig = idx_orig.cuda(non_blocking=True)
        # add batch dim
        idx_orig.unsqueeze_(0)
        # shape (1, seq_len)

        # compute soft nearest neighbor index as sum of original indices
        # weighted by the softmax weights
        index_nn = th.sum(idx_orig.unsqueeze(1) * beta, dim=-1)
        # shape (batch, seq_len)

        # get mask and indices in correct representation
        idx_nn_rep, idx_orig_rep, emb_mask_rep = self.get_mxn_repr(index_nn, emb_mask, idx_orig, emb_mask)

        # get distance of each NN index to each original index
        # add an artificial dimension as feature dimension to make the same
        # math work out on indices that works on embeddings
        distance = self.loss_distance_fn(idx_nn_rep.unsqueeze(-1), idx_orig_rep.unsqueeze(-1))
        # shape (batch, seq_len, seq_len)

        # mask values that exceed the sequence length
        distance.masked_fill_(~emb_mask_rep, 0)

        # diagonal of last 2 dims of this distance tensor contains distance
        # from soft index i to hard index i, this is the loss distance
        loss_simple_per_seq = distance.diagonal(dim1=-2, dim2=-1)
        # shape (batch, seq_len)

        # to get variance, multiply with beta (softmax output weights)
        # and sum over the row
        variance = th.sum(distance * beta, dim=-1)

        # calculate regularizer loss on the variance and apply mask
        var_reg_per_seq = self.lambda_index_gauss * .5 * th.log(self.var_log_eps + variance)
        var_reg_per_seq.masked_fill_(emb_mask, 0)
        # shape (batch, seq_len)

        # calculate loss (no need to apply mask since distance is masked to 0)
        loss_gauss_per_seq = loss_simple_per_seq / (variance + self.var_denom_eps) + var_reg_per_seq

        # for now return all the losses, in case we want to sample some
        # sequences only
        return loss_simple_per_seq, loss_gauss_per_seq, var_reg_per_seq
