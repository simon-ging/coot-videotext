"""
Text generation, greedy or beam search.

References:
    Copyright (c) 2017 Jie Lei
    Licensed under The MIT License, see https://choosealicense.com/licenses/mit/
    @inproceedings{lei2020mart,
        title={MART: Memory-Augmented Recurrent Transformer for Coherent Video Paragraph Captioning},
        author={Lei, Jie and Wang, Liwei and Shen, Yelong and Yu, Dong and Berg, Tamara L and Bansal, Mohit},
        booktitle={ACL},
        year={2020}
    }

    History:
    https://github.com/jayleicn/recurrent-transformer
    Current version 2021 https://github.com/gingsi/coot-videotext
"""

import copy
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from mart.beam_search import BeamSearch
from mart.configs_mart import MartConfig
from mart.recursive_caption_dataset import RecursiveCaptionDataset as RCDataset
from nntrainer import utils


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1).transpose(0, 1).repeat(count, 1).transpose(0, 1).contiguous().view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def mask_tokens_after_eos(input_ids, input_masks,
                          eos_token_id=RCDataset.EOS, pad_token_id=RCDataset.PAD):
    """
    replace values after `[EOS]` with `[PAD]`,
    used to compute memory for next sentence generation
    """
    for row_idx in range(len(input_ids)):
        # possibly more than one `[EOS]`
        # noinspection PyUnresolvedReferences
        cur_eos_idxs = (input_ids[row_idx] == eos_token_id).nonzero(as_tuple=False)
        if len(cur_eos_idxs) != 0:
            cur_eos_idx = cur_eos_idxs[0, 0].item()
            input_ids[row_idx, cur_eos_idx + 1:] = pad_token_id
            input_masks[row_idx, cur_eos_idx + 1:] = 0
    return input_ids, input_masks


class Translator(object):
    """
    Load with trained model and handle the beam search.
    """

    def __init__(self, model: nn.Module, cfg: MartConfig, logger: Optional[logging.Logger] = None):
        self.model = model
        self.cfg = cfg
        self.logger = logger
        if self.logger is None:
            self.logger = utils.create_logger_without_file("translator", log_level=utils.LogLevelsConst.INFO)

    def translate_batch_beam(self, input_ids_list, video_features_list, input_masks_list, token_type_ids_list,
                             rt_model, beam_size, n_best, min_length, max_length, block_ngram_repeat, exclusion_idxs,
                             length_penalty_name, length_penalty_alpha, use_cuda: bool = True):
        # prep the beam object
        base_beam = BeamSearch(
            beam_size, n_best=n_best, batch_size=len(input_ids_list[0]), pad=RCDataset.PAD, eos=RCDataset.EOS,
            bos=RCDataset.BOS, min_length=min_length, max_length=max_length,
            block_ngram_repeat=block_ngram_repeat, exclusion_tokens=exclusion_idxs,
            length_penalty_name=length_penalty_name, length_penalty_alpha=length_penalty_alpha, use_cuda=use_cuda)

        def duplicate_for_beam(prev_ms_, input_ids, video_features, input_masks, token_type_ids, beam_size_):
            input_ids = tile(input_ids, beam_size_, dim=0)  # (N * beam_size, L)
            video_features = tile(video_features, beam_size_, dim=0)  # (N * beam_size, L, D_v)
            input_masks = tile(input_masks, beam_size_, dim=0)
            token_type_ids = tile(token_type_ids, beam_size_, dim=0)
            prev_ms_ = [tile(e, beam_size_, dim=0) for e in prev_ms_]\
                if prev_ms_[0] is not None else [None] * len(input_ids)
            return prev_ms_, input_ids, video_features, input_masks, token_type_ids

        def copy_for_memory(*inputs):
            return [copy.deepcopy(e) for e in inputs]

        def beam_decoding_step(prev_ms_, input_ids, video_features, input_masks, token_type_ids, model,
                               max_v_len, max_t_len, beam_size_):
            # unused arguments , start_idx=RCDataset.BOS, unk_idx=RCDataset.UNK):
            """
            prev_ms: [(N, M, D), ] * num_hidden_layers or None at first step.
            input_ids: (N, L),
            video_features: (N, L, D_v)
            input_masks: (N, L)
            token_type_ids: (N, L)
            """
            init_ms, init_input_ids, init_video_features, init_input_masks, init_token_type_ids = copy_for_memory(
                prev_ms_, input_ids, video_features, input_masks, token_type_ids)

            prev_ms_, input_ids, video_features, input_masks, token_type_ids = duplicate_for_beam(
                prev_ms_, input_ids, video_features, input_masks, token_type_ids, beam_size_=beam_size_)

            beam = copy.deepcopy(base_beam)  # copy global variable as local

            # logger.info("batch_size {}, beam_size {}".format(len(input_ids_list[0]), beam_size))
            # logger.info("input_ids {} {}".format(input_ids.shape, input_ids[:6]))
            # logger.info("video_features {}".format(video_features.shape))
            # logger.info("input_masks {} {}".format(input_masks.shape, input_masks[:6]))
            # logger.info("token_type_ids {} {}".format(token_type_ids.shape, token_type_ids[:6]))

            for dec_idx in range(max_v_len, max_v_len + max_t_len):
                # logger.info(" dec_idx {} beam.current_predictions {} {}"
                #             .format(dec_idx, beam.current_predictions.shape, beam.current_predictions))
                input_ids[:, dec_idx] = beam.current_predictions
                input_masks[:, dec_idx] = 1
                copied_prev_ms = copy.deepcopy(prev_ms_)  # since the func is changing data inside
                _, _, pred_scores = model.forward_step(
                    copied_prev_ms, input_ids, video_features, input_masks, token_type_ids)
                pred_scores[:, RCDataset.UNK] = -1e10  # remove `[UNK]` token
                logprobs = torch.log(F.softmax(pred_scores[:, dec_idx], dim=1))  # (N * beam_size, vocab_size)
                # next_words = logprobs.max(1)[1]
                # logger.info("next_words {}".format(next_words))
                # import sys
                # sys.exit(1)
                beam.advance(logprobs)
                any_beam_is_finished = beam.is_finished.any()
                if any_beam_is_finished:
                    beam.update_finished()
                    if beam.done:
                        break

                if any_beam_is_finished:
                    # update input args
                    select_indices = beam.current_origin  # N * B, i.e. batch_size * beam_size
                    input_ids = input_ids.index_select(0, select_indices)
                    video_features = video_features.index_select(0, select_indices)
                    input_masks = input_masks.index_select(0, select_indices)
                    token_type_ids = token_type_ids.index_select(0, select_indices)
                    # logger.info("prev_ms {} {}".format(prev_ms[0], type(prev_ms[0])))
                    # logger.info("select_indices {} {}".format(len(select_indices), select_indices))
                    if prev_ms_[0] is None:
                        prev_ms_ = [None] * len(select_indices)
                    else:
                        # noinspection PyUnresolvedReferences
                        prev_ms_ = [e.index_select(0, select_indices) for e in prev_ms_]

            # Note: In the MART repo was the comment "TO DO update memory"
            # fill in generated words
            for batch_idx in range(len(beam.predictions)):
                cur_sen_ids = beam.predictions[batch_idx][0].cpu().tolist()  # use the top sentences
                cur_sen_ids = [RCDataset.BOS] + cur_sen_ids + [RCDataset.EOS]
                cur_sen_len = len(cur_sen_ids)
                init_input_ids[batch_idx, max_v_len: max_v_len + cur_sen_len] = init_input_ids.new(cur_sen_ids)
                init_input_masks[batch_idx, max_v_len: max_v_len + cur_sen_len] = 1

            # compute memory, mimic the way memory is generated at training time
            init_input_ids, init_input_masks = mask_tokens_after_eos(init_input_ids, init_input_masks)
            cur_ms, _, pred_scores = model.forward_step(
                init_ms, init_input_ids, init_video_features, init_input_masks, init_token_type_ids)

            # logger.info("beam.predictions {}".format(beam.predictions))
            # logger.info("beam.scores {}".format(beam.scores))
            # import sys
            # sys.exit(1)
            # return cur_ms, [e[0][0] for e in beam.predictions]
            return cur_ms, init_input_ids[:, max_v_len:]

        input_ids_list, input_masks_list = self.prepare_video_only_inputs(
            input_ids_list, input_masks_list, token_type_ids_list)
        for cur_input_masks in input_ids_list:
            assert torch.sum(cur_input_masks[:, self.cfg.max_v_len + 1:]) == 0, (
                "Initially, all text tokens should be masked.")

        config = rt_model.cfg
        with torch.no_grad():
            prev_ms = [None] * config.num_hidden_layers
            step_size = len(input_ids_list)
            dec_res_list = []
            for idx in range(step_size):
                prev_ms, dec_res = beam_decoding_step(
                    prev_ms, input_ids_list[idx], video_features_list[idx],
                    input_masks_list[idx], token_type_ids_list[idx],
                    rt_model, config.max_v_len, config.max_t_len, beam_size)
                dec_res_list.append(dec_res)
            return dec_res_list

    def translate_batch_greedy(self, input_ids_list, video_features_list, input_masks_list, token_type_ids_list,
                               rt_model):
        def greedy_decoding_step(prev_ms_, input_ids, video_features, input_masks, token_type_ids,
                                 model, max_v_len, max_t_len, start_idx=RCDataset.BOS, unk_idx=RCDataset.UNK):
            """
            RTransformer The first few args are the same to the input to the forward_step func

            Notes:
                1, Copy the prev_ms each word generation step, as the func will modify this value,
                which will cause discrepancy between training and inference
                2, After finish the current sentence generation step, replace the words generated
                after the `[EOS]` token with `[PAD]`. The replaced input_ids should be used to generate
                next memory state tensor.
            """
            bsz = len(input_ids)
            next_symbols = torch.LongTensor([start_idx] * bsz)  # (N, )
            for dec_idx in range(max_v_len, max_v_len + max_t_len):
                input_ids[:, dec_idx] = next_symbols
                input_masks[:, dec_idx] = 1
                # if dec_idx < max_v_len + 5:
                #     logger.info("prev_ms {} {}".format(type(prev_ms[0]), prev_ms[0]))
                copied_prev_ms = copy.deepcopy(prev_ms_)  # since the func is changing data inside
                _, _, pred_scores = model.forward_step(
                    copied_prev_ms, input_ids, video_features, input_masks, token_type_ids)
                # suppress unk token; (N, L, vocab_size)
                pred_scores[:, :, unk_idx] = -1e10
                # next_words = pred_scores.max(2)[1][:, dec_idx]
                next_words = pred_scores[:, dec_idx].max(1)[1]
                next_symbols = next_words

            # compute memory, mimic the way memory is generated at training time
            input_ids, input_masks = mask_tokens_after_eos(input_ids, input_masks)
            cur_ms, _, pred_scores = model.forward_step(
                prev_ms_, input_ids, video_features, input_masks, token_type_ids)

            # logger.info("input_ids[:, max_v_len:] {}".format(input_ids[:, max_v_len:]))
            # import sys
            # sys.exit(1)

            return cur_ms, input_ids[:, max_v_len:]  # (N, max_t_len == L-max_v_len)

        input_ids_list, input_masks_list = self.prepare_video_only_inputs(
            input_ids_list, input_masks_list, token_type_ids_list)
        for cur_input_masks in input_ids_list:
            assert torch.sum(cur_input_masks[:, self.cfg.max_v_len + 1:]) == 0,\
                "Initially, all text tokens should be masked"

        config = rt_model.cfg
        with torch.no_grad():
            prev_ms = [None] * config.num_hidden_layers
            step_size = len(input_ids_list)
            dec_seq_list = []
            for idx in range(step_size):
                prev_ms, dec_seq = greedy_decoding_step(
                    prev_ms, input_ids_list[idx], video_features_list[idx],
                    input_masks_list[idx], token_type_ids_list[idx],
                    rt_model, config.max_v_len, config.max_t_len)
                dec_seq_list.append(dec_seq)
            return dec_seq_list

    def translate_batch_greedy_xl(self, input_ids_list, video_features_list, input_masks_list, token_type_ids_list,
                                  rt_model):
        def greedy_decoding_step(prev_ms_, input_ids, video_features, token_type_ids, input_masks, prev_masks_,
                                 model, max_v_len, max_t_len, start_idx=RCDataset.BOS, unk_idx=RCDataset.UNK):
            """
            TransformerXL: The first few args are the same to the input to the forward_step func

            Notes:
                1, Copy the prev_ms each word generation step, as the func will modify this value,
                which will cause discrepancy between training and inference
                2, After finish the current sentence generation step, replace the words generated
                after the `[EOS]` token with `[PAD]`. The replaced input_ids should be used to generate
                next memory state tensor.
            """
            bsz = len(input_ids)
            next_symbols = torch.LongTensor([start_idx] * bsz)  # (N, )
            for dec_idx in range(max_v_len, max_v_len + max_t_len):
                input_ids[:, dec_idx] = next_symbols
                input_masks[:, dec_idx] = 1  # no need to worry about generated <PAD>
                # if dec_idx < max_v_len + 5:
                #     logger.info("prev_ms {} {}".format(type(prev_ms[0]), prev_ms[0]))
                copied_prev_ms = copy.deepcopy(prev_ms_)  # since the func is changing data inside
                _, _, pred_scores = model.forward_step(
                    copied_prev_ms, input_ids, video_features, token_type_ids, input_masks, prev_masks_)
                # suppress unk token; (N, L, vocab_size)
                pred_scores[:, :, unk_idx] = -1e10
                # next_words = pred_scores.max(2)[1][:, dec_idx]
                next_words = pred_scores[:, dec_idx].max(1)[1]
                next_symbols = next_words

            # compute memory, mimic the way memory is generated at training time
            input_ids, input_masks = mask_tokens_after_eos(input_ids, input_masks)
            cur_ms, _, pred_scores = model.forward_step(
                prev_ms_, input_ids, video_features, token_type_ids, input_masks, prev_masks_)

            # logger.info("input_ids[:, max_v_len:] {}".format(input_ids[:, max_v_len:]))
            # import sys
            # sys.exit(1)

            return cur_ms, input_ids[:, max_v_len:], input_masks  # (N, max_t_len == L-max_v_len)

        input_ids_list, input_masks_list = self.prepare_video_only_inputs(
            input_ids_list, input_masks_list, token_type_ids_list)
        for cur_input_masks in input_ids_list:
            assert torch.sum(cur_input_masks[:, self.cfg.max_v_len + 1:]) == 0,\
                "Initially, all text tokens should be masked"

        config = rt_model.cfg
        with torch.no_grad():
            prev_ms = rt_model.init_mems()
            step_size = len(input_ids_list)
            dec_seq_list = []
            prev_masks = None
            for idx in range(step_size):
                prev_ms, dec_seq, prev_masks = greedy_decoding_step(
                    prev_ms, input_ids_list[idx], video_features_list[idx],
                    token_type_ids_list[idx], input_masks_list[idx], prev_masks,
                    rt_model, config.max_v_len, config.max_t_len)
                dec_seq_list.append(dec_seq)
            return dec_seq_list

    def translate_batch_single_sentence_greedy(self, input_ids, video_features, input_masks, token_type_ids,
                                               model, start_idx=RCDataset.BOS, unk_idx=RCDataset.UNK):
        """
        The first few args are the same to the input to the forward_step func

        Notes:
            1, Copy the prev_ms each word generation step, as the func will modify this value,
            which will cause discrepancy between training and inference
            2, After finish the current sentence generation step, replace the words generated
            after the `[EOS]` token with `[PAD]`. The replaced input_ids should be used to generate
            next memory state tensor.
        """
        input_ids, input_masks = self.prepare_video_only_inputs(input_ids, input_masks, token_type_ids)
        assert torch.sum(input_masks[:, self.cfg.max_v_len + 1:]) == 0, "Initially, all text tokens should be masked"
        config = model.cfg
        max_v_len = config.max_v_len
        max_t_len = config.max_t_len
        bsz = len(input_ids)
        next_symbols = torch.LongTensor([start_idx] * bsz)  # (N, )
        for dec_idx in range(max_v_len, max_v_len + max_t_len):
            input_ids[:, dec_idx] = next_symbols
            input_masks[:, dec_idx] = 1
            # if dec_idx < max_v_len + 5:
            #     logger.info("prev_ms {} {}".format(type(prev_ms[0]), prev_ms[0]))
            _, pred_scores = model.forward(input_ids, video_features, input_masks, token_type_ids, None)
            # suppress unk token; (N, L, vocab_size)
            pred_scores[:, :, unk_idx] = -1e10
            # next_words = pred_scores.max(2)[1][:, dec_idx]
            next_words = pred_scores[:, dec_idx].max(1)[1]
            next_symbols = next_words
        return input_ids[:, max_v_len:]  # (N, max_t_len == L-max_v_len)

    @classmethod
    def translate_batch_single_sentence_untied_greedy(
            cls, video_features, video_masks, text_input_ids, text_masks, text_input_labels,
            model, start_idx=RCDataset.BOS, unk_idx=RCDataset.UNK):
        """
        The first few args are the same to the input to the forward_step func

        Notes:
            1, Copy the prev_ms each word generation step, as the func will modify this value,
            which will cause discrepancy between training and inference
            2, After finish the current sentence generation step, replace the words generated
            after the `[EOS]` token with `[PAD]`. The replaced input_ids should be used to generate
            next memory state tensor.
        """
        encoder_outputs = model.encode(video_features, video_masks)  # (N, Lv, D)

        config = model.cfg
        max_t_len = config.max_t_len
        bsz = len(text_input_ids)
        text_input_ids = text_input_ids.new_zeros(text_input_ids.size())  # all zeros
        text_masks = text_masks.new_zeros(text_masks.size())  # all zeros
        next_symbols = torch.LongTensor([start_idx] * bsz)  # (N, )
        for dec_idx in range(max_t_len):
            text_input_ids[:, dec_idx] = next_symbols
            text_masks[:, dec_idx] = 1
            _, pred_scores = model.decode(
                text_input_ids, text_masks, text_input_labels, encoder_outputs, video_masks)
            # suppress unk token; (N, L, vocab_size)
            pred_scores[:, :, unk_idx] = -1e10
            # next_words = pred_scores.max(2)[1][:, dec_idx]
            next_words = pred_scores[:, dec_idx].max(1)[1]
            next_symbols = next_words
        return text_input_ids  # (N, Lt)

    def translate_batch(self, model_inputs, use_beam=False, recurrent=True, untied=False, xl=False, mtrans=False):
        """
        while we used *_list as the input names, they could be non-list for single sentence decoding case
        """
        if use_beam:
            if recurrent:
                input_ids_list, video_features_list, input_masks_list, token_type_ids_list = model_inputs
                return self.translate_batch_beam(
                    input_ids_list, video_features_list, input_masks_list, token_type_ids_list,
                    self.model, beam_size=self.cfg.beam_size, n_best=self.cfg.n_best,
                    min_length=self.cfg.min_sen_len, max_length=self.cfg.max_sen_len - 2,
                    block_ngram_repeat=self.cfg.block_ngram_repeat, exclusion_idxs=[],
                    length_penalty_name=self.cfg.length_penalty_name,
                    length_penalty_alpha=self.cfg.length_penalty_alpha, use_cuda=self.cfg.use_cuda)
            else:
                raise NotImplementedError
        else:
            if recurrent:
                input_ids_list, video_features_list, input_masks_list, token_type_ids_list = model_inputs
                if xl:
                    return self.translate_batch_greedy_xl(
                        input_ids_list, video_features_list, input_masks_list, token_type_ids_list, self.model)
                else:
                    return self.translate_batch_greedy(
                        input_ids_list, video_features_list, input_masks_list, token_type_ids_list, self.model)
            else:  # single sentence
                if untied or mtrans:
                    video_features, video_masks, text_input_ids, text_masks, text_input_labels = model_inputs
                    return self.translate_batch_single_sentence_untied_greedy(
                        video_features, video_masks, text_input_ids, text_masks, text_input_labels, self.model)
                else:
                    input_ids_list, video_features_list, input_masks_list, token_type_ids_list = model_inputs
                    return self.translate_batch_single_sentence_greedy(
                        input_ids_list, video_features_list, input_masks_list, token_type_ids_list,
                        self.model)

    @classmethod
    def prepare_video_only_inputs(cls, input_ids, input_masks, segment_ids):
        """
        replace text_ids (except `[BOS]`) in input_ids with `[PAD]` token, for decoding.
        This function is essential!!!
        Args:
            input_ids: (N, L) or [(N, L)] * step_size
            input_masks: (N, L) or [(N, L)] * step_size
            segment_ids: (N, L) or [(N, L)] * step_size
        """
        if isinstance(input_ids, list):
            video_only_input_ids_list = []
            video_only_input_masks_list = []
            for e1, e2, e3 in zip(input_ids, input_masks, segment_ids):
                text_mask = e3 == 1  # text positions (`1`) are replaced
                e1[text_mask] = RCDataset.PAD
                e2[text_mask] = 0  # mark as invalid bits
                video_only_input_ids_list.append(e1)
                video_only_input_masks_list.append(e2)
            return video_only_input_ids_list, video_only_input_masks_list
        else:
            text_mask = segment_ids == 1
            input_ids[text_mask] = RCDataset.PAD
            input_masks[text_mask] = 0
            return input_ids, input_masks

    @classmethod
    def sort_res(cls, res_dict):
        """
        res_dict: the submission json entry `results`
        """
        final_res_dict = {}
        for k, v in list(res_dict.items()):
            final_res_dict[k] = sorted(v, key=lambda x: float(x["timestamp"][0]))
        return final_res_dict
