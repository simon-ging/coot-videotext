"""
https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/beam_search.py

References:
    Copyright (c) 2017 Adam Lerer
    Licensed under The MIT License, see https://choosealicense.com/licenses/mit/
    @inproceedings{klein-etal-2017-opennmt,
        title = "{O}pen{NMT}: Open-Source Toolkit for Neural Machine Translation",
        author = "Klein, Guillaume  and
          Kim, Yoon  and
          Deng, Yuntian  and
          Senellart, Jean  and
          Rush, Alexander",
        booktitle = "Proceedings of {ACL} 2017, System Demonstrations",
        month = jul,
        year = "2017",
        address = "Vancouver, Canada",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/P17-4012",
        pages = "67--72",
    }

    History:
    https://github.com/jayleicn/recurrent-transformer
    https://github.com/OpenNMT/OpenNMT-py
    Current version 2021 https://github.com/gingsi/coot-videotext
"""
from __future__ import annotations
import logging

import torch


logger = logging.getLogger(__name__)


class DecodeStrategy(object):
    """
    Base class for generation strategies.

    Args:
        pad (int): Magic integer in output vocab.
        bos (int): Magic integer in output vocab.
        eos (int): Magic integer in output vocab.
        batch_size (int): Current batch size.
        parallel_paths (int): Decoding strategies like beam search
            use parallel paths. Each batch is repeated ``parallel_paths``
            times in relevant state tensors.
        min_length (int): Shortest acceptable generation, not counting
            begin-of-sentence or end-of-sentence.
        max_length (int): Longest acceptable sequence, not counting
            begin-of-sentence (presumably there has been no EOS
            yet if max_length is used as a cutoff).
        block_ngram_repeat (int): Block beams where
            ``block_ngram_repeat``-grams repeat.
        exclusion_tokens (set[int]): If a gram contains any of these
            tokens, it may repeat.
        use_cuda: Move tensors to GPU

    Attributes:
        pad (int): See above.
        bos (int): See above.
        eos (int): See above.
        predictions (list[list[LongTensor]]): For each batch, holds a
            list of beam prediction sequences.
        scores (list[list[FloatTensor]]): For each batch, holds a
            list of scores.
        attention (list[list[FloatTensor or list[]]]): For each
            batch, holds a list of attention sequence tensors
            (or empty lists) having shape ``(step, inp_seq_len)`` where
            ``inp_seq_len`` is the length of the sample (not the max
            length of all inp seqs).
        alive_seq (LongTensor): Shape ``(B x parallel_paths, step)``.
            This sequence grows in the ``step`` axis on each call to
            :func:`advance()`.
        is_finished (ByteTensor or NoneType): Shape
            ``(B, parallel_paths)``. Initialized to ``None``.
        alive_attn (FloatTensor or NoneType): If tensor, shape is
            ``(step, B x parallel_paths, inp_seq_len)``, where ``inp_seq_len``
            is the (max) length of the input sequence.
        min_length (int): See above.
        max_length (int): See above.
        block_ngram_repeat (int): See above.
        exclusion_tokens (set[int]): See above.
        done (bool): See above.
    """

    def __init__(
            self, pad, bos, eos, batch_size, parallel_paths, min_length, block_ngram_repeat, exclusion_tokens,
            max_length, use_cuda: bool = True):
        # magic indices
        self.pad = pad
        self.bos = bos
        self.eos = eos

        # result caching
        self.predictions = [[] for _ in range(batch_size)]
        self.scores = [[] for _ in range(batch_size)]
        self.attention = [[] for _ in range(batch_size)]

        # (N * B, step_size=1)
        self.alive_seq = torch.full([batch_size * parallel_paths, 1], self.bos, dtype=torch.long)
        self.is_finished = torch.zeros([batch_size, parallel_paths], dtype=torch.uint8)
        if use_cuda:
            self.alive_seq = self.alive_seq.cuda()
            self.is_finished = self.is_finished.cuda()

        self.alive_attn = None

        self.min_length = min_length
        self.max_length = max_length
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens

        self.done = False

    def __len__(self):
        return self.alive_seq.shape[1]  # steps length

    def ensure_min_length(self, log_probs):
        if len(self) <= self.min_length:
            log_probs[:, self.eos] = -1e20

    def ensure_max_length(self):
        # add one to account for BOS. Don't account for EOS because hitting
        # this implies it hasn't been found.
        if len(self) == self.max_length + 1:
            self.is_finished.fill_(1)

    def block_ngram_repeats(self, log_probs):
        # log_probs (N * B, vocab_size)
        cur_len = len(self)
        if self.block_ngram_repeat > 0 and cur_len > 1:
            for path_idx in range(self.alive_seq.shape[0]):  # N * B
                # skip BOS
                hyp = self.alive_seq[path_idx, 1:]
                ngrams = set()
                fail = False
                gram = []
                for i in range(cur_len - 1):
                    # Last n tokens, n = block_ngram_repeat
                    gram = (gram + [hyp[i].item()])[-self.block_ngram_repeat:]
                    # skip the blocking if any token in gram is excluded
                    if set(gram) & self.exclusion_tokens:
                        continue
                    if tuple(gram) in ngrams:
                        fail = True
                    ngrams.add(tuple(gram))
                if fail:
                    log_probs[path_idx] = -10e20  # all the words in this path

    def advance(self, log_probs):
        """
        DecodeStrategy subclasses should override :func:`advance()`.

        Advance is used to update ``self.alive_seq``, ``self.is_finished``,
        and, when appropriate, ``self.alive_attn``.
        """

        raise NotImplementedError()

    def update_finished(self):
        """
        DecodeStrategy subclasses should override :func:`update_finished()`.

        ``update_finished`` is used to update ``self.predictions``,
        ``self.scores``, and other "output" attributes.
        """

        raise NotImplementedError()


def length_penalty_builder(length_penalty_name="none"):
    """
    implement length penalty
    """

    def length_wu(cur_len, alpha=0.):
        """
        GNMT length re-ranking score.
        See "Google's Neural Machine Translation System" :cite:`wu2016google`.
        """
        return ((5 + cur_len) / 6.0) ** alpha

    def length_average(cur_len, _alpha=0.):
        """
        Returns the current sequence length.
        """
        return cur_len

    def length_none(_cur_len, _alpha=0.):
        """
        Returns unmodified scores.
        """
        return 1.0

    if length_penalty_name == "none":
        return length_none
    elif length_penalty_name == "wu":
        return length_wu
    elif length_penalty_name == "avg":
        return length_average
    else:
        raise NotImplementedError


class BeamSearch(DecodeStrategy):
    """
    Generation beam search.

    Note that the attributes list is not exhaustive. Rather, it highlights
    tensors to document their shape. (Since the state variables' "batch"
    size decreases as beams finish, we denote this axis with a B rather than
    ``batch_size``).

    Args:
        beam_size (int): Number of beams to use (see base ``parallel_paths``).
        batch_size (int): See base.
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.
        n_best (int): Don't stop until at least this many beams have
            reached EOS.
        min_length (int): See base.
        max_length (int): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.
        use_cuda: Move tensors to GPU

    Attributes:
        top_beam_finished (ByteTensor): Shape ``(B,)``.
        _batch_offset (LongTensor): Shape ``(B,)``.
        _beam_offset (LongTensor): Shape ``(batch_size x beam_size,)``.
        alive_seq (LongTensor): See base.
        topk_log_probs (FloatTensor): Shape ``(B x beam_size,)``. These
            are the scores used for the topk operation.
        select_indices (LongTensor or NoneType): Shape
            ``(B x beam_size,)``. This is just a flat view of the
            ``_batch_index``.
        topk_scores (FloatTensor): Shape
            ``(B, beam_size)``. These are the
            scores a sequence will receive if it finishes.
        topk_ids (LongTensor): Shape ``(B, beam_size)``. These are the
            word indices of the topk predictions.
        _batch_index (LongTensor): Shape ``(B, beam_size)``.
        _prev_penalty (FloatTensor or NoneType): Shape
            ``(B, beam_size)``. Initialized to ``None``.
        _coverage (FloatTensor or NoneType): Shape
            ``(1, B x beam_size, inp_seq_len)``.
        hypotheses (list[list[Tuple[Tensor]]]): Contains a tuple
            of score (float), sequence (long), and attention (float or None).
    """

    def __init__(self, beam_size, batch_size, pad, bos, eos, n_best,
                 min_length, max_length, block_ngram_repeat, exclusion_tokens,
                 length_penalty_name=None, length_penalty_alpha=0., use_cuda: bool = True):
        super().__init__(
            pad, bos, eos, batch_size, beam_size, min_length,
            block_ngram_repeat, exclusion_tokens, max_length, use_cuda=use_cuda)
        # beam parameters
        self.beam_size = beam_size
        self.n_best = n_best
        self.batch_size = batch_size
        self.length_penalty_name = length_penalty_name
        self.length_penalty_func = length_penalty_builder(length_penalty_name)
        self.length_penalty_alpha = length_penalty_alpha

        # result caching
        self.hypotheses = [[] for _ in range(batch_size)]

        # beam state
        self.top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8)
        self.best_scores = torch.full([batch_size], -1e10, dtype=torch.float)  # (N, )

        self._batch_offset = torch.arange(batch_size, dtype=torch.long)  # (N, )
        self._beam_offset = torch.arange(0, batch_size * beam_size, step=beam_size, dtype=torch.long)  # (N, )
        # (B*N), guess: store the current beam probabilities
        self.topk_log_probs = torch.tensor([0.0] + [float("-inf")] * (beam_size - 1)).repeat(batch_size)
        self.select_indices = None

        # buffers for the topk scores and 'backpointer'
        self.topk_scores = torch.empty((batch_size, beam_size), dtype=torch.float)  # (N, B)
        self.topk_ids = torch.empty((batch_size, beam_size), dtype=torch.long)  # (N, B)
        self._batch_index = torch.empty([batch_size, beam_size], dtype=torch.long)  # (N, B)
        self.done = False
        # "global state" of the old beam
        self._prev_penalty = None
        self._coverage = None

        if use_cuda:
            self.best_scores = self.best_scores.cuda()
            self._beam_offset = self._beam_offset.cuda()
            self.topk_log_probs = self.topk_log_probs.cuda()
            self.topk_scores = self.topk_scores.cuda()
            self.topk_ids = self.topk_ids.cuda()
            self._batch_index = self._batch_index.cuda()

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def current_origin(self):
        return self.select_indices

    @property
    def current_backptr(self):
        # for testing
        return self.select_indices.view(self.batch_size, self.beam_size)\
            .fmod(self.beam_size)

    def advance(self, log_probs):
        """
        current step log_probs (N * B, vocab_size), attn (1, N * B, L)
        Which attention is this??? Guess: the one with the encoder outputs
        """
        vocab_size = log_probs.size(-1)

        # using integer division to get an integer _B without casting
        _B = log_probs.shape[0] // self.beam_size  # batch_size

        # force the output to be longer than self.min_length,
        # by setting prob(EOS) to be a very small number when < min_length
        self.ensure_min_length(log_probs)

        # Multiply probs by the beam probability.
        # logger.info("after log_probs {} {}".format(log_probs.shape, log_probs))
        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)
        # logger.info("after log_probs {} {}".format(log_probs.shape, log_probs))

        self.block_ngram_repeats(log_probs)

        # if the sequence ends now, then the penalty is the current
        # length + 1, to include the EOS token, length_penalty is a float number
        step = len(self)
        length_penalty = self.length_penalty_func(step + 1, self.length_penalty_alpha)

        # Flatten probs into a list of possibilities.
        # pick topk in all the paths
        curr_scores = log_probs / length_penalty
        curr_scores = curr_scores.reshape(_B, self.beam_size * vocab_size)
        # self.topk_scores and self.topk_ids => (N, B)
        torch.topk(curr_scores, self.beam_size, dim=-1,
                   out=(self.topk_scores, self.topk_ids))

        # Recover log probs.
        # Length penalty is just a scalar. It doesn't matter if it's applied
        # before or after the topk.
        torch.mul(self.topk_scores, length_penalty, out=self.topk_log_probs)

        # Resolve beam origin and map to batch index flat representation.
        torch.div(self.topk_ids, vocab_size, out=self._batch_index)  # _batch_index (N * B)
        self._batch_index += self._beam_offset[:_B].unsqueeze(1)
        self.select_indices = self._batch_index.view(_B * self.beam_size)

        self.topk_ids.fmod_(vocab_size)  # resolve true word ids

        # Append last prediction.
        self.alive_seq = torch.cat(
            [self.alive_seq.index_select(0, self.select_indices),
             self.topk_ids.view(_B * self.beam_size, 1)], -1)  # (N * B, step_size)

        self.is_finished = self.topk_ids.eq(self.eos)  # (N, B)
        self.ensure_max_length()

    def update_finished(self):
        # Penalize beams that finished.
        _B_old = self.topk_log_probs.shape[0]  # batch_size might be changing??? as the beams finished
        step = self.alive_seq.shape[-1]  # 1 greater than the step in advance, as we advanced 1 step
        self.topk_log_probs.masked_fill_(self.is_finished, -1e10)
        # on real data (newstest2017) with the pretrained transformer,
        # it's faster to not move this back to the original device
        self.is_finished = self.is_finished.to('cpu')  # (N, B)
        self.top_beam_finished |= self.is_finished[:, 0].eq(1)  # (N, ) initialized as zeros
        predictions = self.alive_seq.view(_B_old, self.beam_size, step)
        non_finished_batch = []
        for i in range(self.is_finished.size(0)):  # (N, )
            b = self._batch_offset[i]  # (0, ..., N-1)
            finished_hyp = self.is_finished[i].nonzero().view(-1)
            # Store finished hypotheses for this batch.
            for j in finished_hyp:
                self.hypotheses[b].append([self.topk_scores[i, j],
                                           predictions[i, j, 1:]])
            # End condition is the top beam finished and we can return
            # n_best hypotheses.
            finish_flag = self.top_beam_finished[i] != 0
            if finish_flag and len(self.hypotheses[b]) >= self.n_best:
                best_hyp = sorted(
                    self.hypotheses[b], key=lambda x: x[0], reverse=True)  # sort by scores
                for n, (score, pred) in enumerate(best_hyp):
                    if n >= self.n_best:
                        break
                    self.scores[b].append(score)
                    self.predictions[b].append(pred)
            else:
                non_finished_batch.append(i)
        non_finished = torch.tensor(non_finished_batch)
        # If all sentences are translated, no need to go further.
        if len(non_finished) == 0:
            self.done = True
            return

        _B_new = non_finished.shape[0]
        # Remove finished batches for the next step. (Not finished beam!!!)
        self.top_beam_finished = self.top_beam_finished.index_select(
            0, non_finished)
        self._batch_offset = self._batch_offset.index_select(0, non_finished)
        non_finished = non_finished.to(self.topk_ids.device)
        self.topk_log_probs = self.topk_log_probs.index_select(0,
                                                               non_finished)
        self._batch_index = self._batch_index.index_select(0, non_finished)
        self.select_indices = self._batch_index.view(_B_new * self.beam_size)
        self.alive_seq = predictions.index_select(0, non_finished)\
            .view(-1, self.alive_seq.size(-1))
        self.topk_scores = self.topk_scores.index_select(0, non_finished)
        self.topk_ids = self.topk_ids.index_select(0, non_finished)
