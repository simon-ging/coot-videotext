from collections import OrderedDict

import numpy as np
import torch
from easydict import EasyDict
from torch import nn

import utils


class CootModel:
    def __init__(self, cfg: EasyDict, use_cuda: bool, use_multi_gpu: bool):
        self.use_cuda = use_cuda
        self.use_multi_gpu = use_multi_gpu
        self.model_list = []
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.net_video_pooler = Transformer(
            cfg.net_video_pooler, cfg.dataset.feature_dim)
        self.net_video_pooler = self.to_device_fn(self.net_video_pooler)
        self.net_video_sequencer = Transformer(
            cfg.net_video_sequencer, cfg.net_video_pooler.output_dim)
        self.net_video_sequencer = self.to_device_fn(self.net_video_sequencer)
        self.net_text_pooler = Transformer(
            cfg.net_text_pooler, cfg.text_encoder.feature_dim)
        self.net_text_pooler = self.to_device_fn(self.net_text_pooler)
        self.net_text_sequencer = Transformer(
            cfg.net_text_sequencer, cfg.net_text_pooler.output_dim)
        self.net_text_sequencer = self.to_device_fn(self.net_text_sequencer)
        self.model_list = [self.net_video_pooler, self.net_video_sequencer,
                           self.net_text_pooler, self.net_text_sequencer]

    def encode_video(
            self, vid_frames, vid_frames_mask, vid_frames_len,
            clip_num, clip_frames, clip_frames_len, clip_frames_mask):
        # compute video context
        vid_context = self.net_video_pooler(
            vid_frames, vid_frames_mask, vid_frames_len, None)
        if self.cfg.net_video_sequencer.use_context:
            if self.cfg.net_video_sequencer.name == "rnn":
                vid_context_hidden = vid_context.unsqueeze(0)
                vid_context_hidden = vid_context_hidden.repeat(
                    self.cfg.net_video_sequencer.num_layers, 1, 1)
            elif self.cfg.net_video_sequencer.name == "atn":
                vid_context_hidden = vid_context
            else:
                raise NotImplementedError
        else:
            vid_context_hidden = None

        # compute clip embedding
        clip_emb = self.net_video_pooler(
            clip_frames, clip_frames_mask, clip_frames_len, None)
        batch_size = len(clip_num)
        max_clip_len = torch.max(clip_num)
        clip_feat_dim = self.cfg.net_video_pooler.output_dim
        clip_emb_reshape = torch.zeros(
            (batch_size, max_clip_len, clip_feat_dim))
        clip_emb_mask = torch.zeros((batch_size, max_clip_len))
        clip_emb_lens = torch.zeros((batch_size,))
        if self.use_cuda:
            clip_emb_reshape = clip_emb_reshape.cuda(non_blocking=True)
            clip_emb_mask = clip_emb_mask.cuda(non_blocking=True)
            clip_emb_lens = clip_emb_lens.cuda(non_blocking=True)
        pointer = 0
        for batch, clip_len in enumerate(clip_num):
            clip_emb_reshape[batch, :clip_len, :] =\
                clip_emb[pointer:pointer + clip_len, :]
            clip_emb_mask[batch, :clip_len] = 1
            clip_emb_lens[batch] = clip_len
            pointer += clip_len

        # compute video embedding
        vid_emb = self.net_video_sequencer(
            clip_emb_reshape, clip_emb_mask, clip_num, vid_context_hidden)
        return (vid_emb, clip_emb, vid_context,
                clip_emb_reshape, clip_emb_mask, clip_emb_lens)

    def encode_paragraph(
            self, par_cap_vectors, par_cap_mask, par_cap_len,
            sent_num, sent_cap_vectors, sent_cap_mask, sent_cap_len):
        # compute paragraph context
        par_context = self.net_text_pooler(
            par_cap_vectors, par_cap_mask, par_cap_len, None)
        if self.cfg.net_text_sequencer.use_context:
            if self.cfg.net_text_sequencer.name == "rnn":
                par_gru_hidden = par_context.unsqueeze(0)
                par_gru_hidden = par_gru_hidden.repeat(
                    self.cfg.net_text_sequencer.num_layers, 1, 1)
            elif self.cfg.net_text_sequencer.name == "atn":
                par_gru_hidden = par_context
            else:
                raise NotImplementedError
        else:
            par_gru_hidden = None

        # compute sentence embedding
        sent_emb = self.net_text_pooler(
            sent_cap_vectors, sent_cap_mask, sent_cap_len, None)
        batch_size = len(sent_num)
        sent_feat_dim = self.cfg.net_text_pooler.output_dim
        max_sent_len = torch.max(sent_num)
        sent_emb_reshape = torch.zeros(
            (batch_size, max_sent_len, sent_feat_dim))
        sent_emb_mask = torch.zeros((batch_size, max_sent_len))
        sent_emb_lens = torch.zeros((batch_size,))
        if self.use_cuda:
            sent_emb_reshape = sent_emb_reshape.cuda(non_blocking=True)
            sent_emb_mask = sent_emb_mask.cuda(non_blocking=True)
            sent_emb_lens = sent_emb_lens.cuda(non_blocking=True)
        pointer = 0
        for batch, sent_len in enumerate(sent_num):
            sent_emb_reshape[batch, :sent_len, :] =\
                sent_emb[pointer:pointer + sent_len, :]
            sent_emb_mask[batch, :sent_len] = 1
            sent_emb_lens[batch] = sent_len
            pointer += sent_len

        # compute paragraph embedding
        par_emb = self.net_text_sequencer(
            sent_emb_reshape, sent_emb_mask, sent_num, par_gru_hidden)
        return (par_emb, sent_emb, par_context,
                sent_emb_reshape, sent_emb_mask, sent_emb_lens)

    def eval(self):
        for model in self.model_list:
            model.eval()
        torch.set_grad_enabled(False)

    def train(self):
        for model in self.model_list:
            model.train()
        torch.set_grad_enabled(True)

    def to_device_fn(self, model):
        if self.use_multi_gpu:
            model = nn.DataParallel(model)
        model = model.to(self.device)
        return model

    def get_params(self):
        params = []
        for model in self.model_list:
            params_dict = OrderedDict(model.named_parameters())
            _params = []
            for key, value in params_dict.items():
                _params += [{
                    'params': value
                }]
            params.extend(_params)
        return params

    def load_checkpoint(self, ckpt: str):
        state = torch.load(str(ckpt))
        for i, m in enumerate(self.model_list):
            state_dict = state[i]
            if self.use_multi_gpu:
                newer_state_dict = OrderedDict()
                for key, val in state_dict.items():
                    assert not key.startswith("module.")
                    new_key = "module." + key
                    newer_state_dict[new_key] = val
                m.load_state_dict(newer_state_dict)
            else:
                m.load_state_dict(state_dict)
            i += 1

    def save_checkpoint(self, ckpt: str):
        model_states = []
        for m in self.model_list:
            state_dict = m.state_dict()
            if self.use_multi_gpu:
                new_state_dict = OrderedDict()
                for key, val in state_dict.items():
                    assert key.startswith("module.")
                    new_key = key[7:]
                    new_state_dict[new_key] = val
                model_states.append(new_state_dict)
            else:
                model_states.append(state_dict)
        torch.save(model_states, str(ckpt))


class LayerNormalization(nn.Module):
    def __init__(self, features_count, epsilon=1e-6):
        super().__init__()
        self.gain = nn.Parameter(
            torch.ones(features_count), requires_grad=True)
        self.bias = nn.Parameter(
            torch.zeros(features_count), requires_grad=True)
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gain * (x - mean) / (std + self.epsilon) + self.bias


def build_pooler(input_dim, cfg: EasyDict) -> nn.Module:
    if cfg.pooler == "atn":
        pooler = AtnPool(
            input_dim, cfg.atn_pool_dim, cfg.atn_pool_heads, cfg.dropout)
    elif cfg.pooler == "avg":
        pooler = AvgPool()
    else:
        raise ValueError(f"unknown pooler {cfg.pooler}")
    return pooler


class Transformer(nn.Module):
    def __init__(self, cfg: EasyDict, feature_dim: int):
        super().__init__()

        self.input_norm = LayerNormalization(feature_dim)
        self.input_fc = None
        input_dim = feature_dim
        if cfg.input_fc:
            self.input_fc = nn.Sequential(
                nn.Linear(feature_dim, cfg.input_fc_output_dim), nn.GELU())
            input_dim = cfg.input_fc_output_dim
        self.embedding = PositionalEncoding(
            input_dim, cfg.dropout, max_len=1000)

        self.tf = TransformerEncoder(
            cfg.num_layers, input_dim, cfg.num_heads, input_dim,
            cfg.dropout)

        self.use_context = cfg.use_context
        if self.use_context:
            self.tf_context = TransformerEncoder(
                cfg.atn_ctx_num_layers, input_dim, cfg.atn_ctx_num_heads,
                input_dim, cfg.dropout)

        self.pooler = build_pooler(input_dim, cfg)

        init_network(self, 0.01)

    def forward(self, features, mask, lengths, hidden_state):
        features = self.input_norm(features)
        if self.input_fc is not None:
            features = self.input_fc(features)
        features = self.embedding(features)
        features = self.tf(features, features, features, mask)
        add_after_pool = None
        if self.use_context:
            hidden_state = hidden_state.unsqueeze(1)
            ctx = self.tf_context(
                hidden_state, features, features, mask)
            add_after_pool = ctx.squeeze(1)
        pooled = self.pooler(features, mask, lengths)
        if add_after_pool is not None:
            pooled = torch.cat([pooled, add_after_pool], dim=-1)
        return pooled


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout_prob=0., max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, dim).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        dimension = torch.arange(0, dim).float()
        div_term = 10000 ** (2 * dimension / dim)
        pe[:, 0::2] = torch.sin(position / div_term[0::2])
        pe[:, 1::2] = torch.cos(position / div_term[1::2])
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim

    def forward(self, x, step=None):
        if step is None:
            x = x + self.pe[:x.size(1), :]
        else:
            x = x + self.pe[:, step]
        x = self.dropout(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, layers_count, d_model, heads_count, d_ff, dropout_prob):
        super().__init__()
        self.d_model = d_model
        assert layers_count > 0
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads_count, d_ff, dropout_prob)
                for _ in range(layers_count)])

    def forward(self, query, key, value, mask):
        batch_size, query_len, embed_dim = query.shape
        batch_size, key_len, embed_dim = key.shape
        mask = (1 - mask.unsqueeze(1).expand(batch_size, query_len, key_len))
        mask = mask == 1
        sources = None
        for encoder_layer in self.encoder_layers:
            sources = encoder_layer(query, key, value, mask)
        return sources


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads_count, d_ff, dropout_prob):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention_layer = Sublayer(
            MultiHeadAttention(heads_count, d_model, dropout_prob), d_model)
        self.pointwise_feedforward_layer = Sublayer(
            PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob), d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value, sources_mask):
        sources = self.self_attention_layer(query, key, value, sources_mask)
        sources = self.dropout(sources)
        sources = self.pointwise_feedforward_layer(sources)
        return sources


class Sublayer(nn.Module):
    def __init__(self, sublayer, d_model):
        super(Sublayer, self).__init__()
        self.sublayer = sublayer
        self.layer_normalization = LayerNormalization(d_model)

    def forward(self, *args):
        x = args[0]
        x = self.sublayer(*args) + x
        return self.layer_normalization(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads_count, d_model, dropout_prob):
        super().__init__()
        assert d_model % heads_count == 0,\
            f"model dim {d_model} not divisible by {heads_count} heads"
        self.d_head = d_model // heads_count
        self.heads_count = heads_count
        self.query_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.key_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.value_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.final_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=3)
        self.attention = None

    def forward(self, query, key, value, mask=None):
        batch_size, query_len, d_model = query.size()
        d_head = d_model // self.heads_count
        query_projected = self.query_projection(query)
        key_projected = self.key_projection(key)
        value_projected = self.value_projection(value)
        batch_size, key_len, d_model = key_projected.size()
        batch_size, value_len, d_model = value_projected.size()
        query_heads = query_projected.view(
            batch_size, query_len, self.heads_count, d_head).transpose(1, 2)
        key_heads = key_projected.view(
            batch_size, key_len, self.heads_count, d_head).transpose(1, 2)
        value_heads = value_projected.view(
            batch_size, value_len, self.heads_count, d_head).transpose(1, 2)
        attention_weights = self.scaled_dot_product(
            query_heads, key_heads)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(attention_weights)
            attention_weights = attention_weights.masked_fill(
                mask_expanded, -1e18)
        attention = self.softmax(attention_weights)
        attention_dropped = self.dropout(attention)
        context_heads = torch.matmul(
            attention_dropped, value_heads)
        context_sequence = context_heads.transpose(1, 2)
        context = context_sequence.reshape(
            batch_size, query_len, d_model)
        final_output = self.final_projection(context)
        return final_output

    def scaled_dot_product(self, query_heads, key_heads):
        key_heads_transposed = key_heads.transpose(2, 3)
        dot_product = torch.matmul(
            query_heads, key_heads_transposed)
        attention_weights = dot_product / np.sqrt(self.d_head)
        return attention_weights


class PointwiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_ff, d_model, dropout_prob):
        super(PointwiseFeedForwardNetwork, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout_prob),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_prob))

    def forward(self, x):
        return self.feed_forward(x)


class AvgPool(nn.Module):
    def forward(self, features, mask, lengths):
        _ = mask
        len_div = lengths.unsqueeze(-1).float()
        result_sum = torch.sum(features, dim=1)
        result = result_sum / len_div
        return result


class AtnPool(nn.Module):
    def __init__(
            self, d_input, d_attn, n_heads, dropout_prob):
        super().__init__()
        self.d_head = d_attn // n_heads
        self.d_head_output = d_input // n_heads
        self.num_heads = n_heads

        def init_(tensor_):
            tensor_.data = (utils.truncated_normal_fill(
                tensor_.data.shape, std=0.01))

        w1_head = torch.zeros(n_heads, d_input, self.d_head)
        b1_head = torch.zeros(n_heads, self.d_head)
        w2_head = torch.zeros(n_heads, self.d_head, self.d_head_output)
        b2_head = torch.zeros(n_heads, self.d_head_output)
        init_(w1_head)
        init_(b1_head)
        init_(w2_head)
        init_(b2_head)
        self.genpool_w1_head = nn.Parameter(w1_head, requires_grad=True)
        self.genpool_b1_head = nn.Parameter(b1_head, requires_grad=True)
        self.genpool_w2_head = nn.Parameter(w2_head, requires_grad=True)
        self.genpool_b2_head = nn.Parameter(b2_head, requires_grad=True)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=2)
        self.softmax_temp = 1
        self.genpool_one = nn.Parameter(torch.ones(1), requires_grad=False)

    def extra_repr(self) -> str:
        strs = []
        for p in [self.genpool_w1_head, self.genpool_b1_head,
                  self.genpool_w2_head, self.genpool_b2_head]:
            strs.append(f"pool linear {p.shape}")
        return "\n".join(strs)

    def forward(self, features, mask, lengths):
        _ = lengths
        batch_size, seq_len, input_dim = features.shape
        b1 = torch.matmul(
            features.unsqueeze(1),
            self.genpool_w1_head.unsqueeze(0))
        b1 += self.genpool_b1_head.unsqueeze(1).unsqueeze(0)
        b1 = self.activation(self.dropout1(b1))
        b1 = torch.matmul(
            b1, self.genpool_w2_head.unsqueeze(0))
        b1 += self.genpool_b2_head.unsqueeze(1).unsqueeze(0)
        b1 = self.dropout2(b1)
        b1.masked_fill_((mask == 0).unsqueeze(1).unsqueeze(-1), -1e19)
        smweights = self.softmax(b1 / self.softmax_temp)
        smweights = self.dropout3(smweights)
        smweights = smweights.transpose(1, 2).reshape(
            -1, seq_len, input_dim)
        pooled = (features * smweights).sum(dim=1)
        return pooled


def init_weight_(w, init_gain=1):
    w.copy_(utils.truncated_normal_fill(w.shape, std=init_gain))


def init_network(net: nn.Module, init_std: float):
    for key, val in net.named_parameters():
        if "weight" in key or "bias" in key:
            init_weight_(val.data, init_std)
