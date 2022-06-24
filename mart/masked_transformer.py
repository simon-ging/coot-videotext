"""
Classic Vanilla Transformer (BERT-like).

References:
    Copyright (c) 2018, salesforce.com, inc.
    All rights reserved.
    SPDX-License-Identifier: BSD-3-Clause, see https://opensource.org/licenses/BSD-3-Clause

    History:
    https://github.com/salesforce/densecap
    https://github.com/jayleicn/recurrent-transformer
    Current version 2021 https://github.com/gingsi/coot-videotext
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from mart.configs_mart import MartConfig
from mart.loss_caption import LabelSmoothingLoss


INF = 1e10


def positional_encodings_like(x, t=None):
    if t is None:
        positions = torch.arange(0, x.size(1)).float()
        if x.is_cuda:
            positions = positions.cuda(x.get_device())
    else:
        positions = t
    encodings = torch.zeros(*x.size()[1:])
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())

    for channel in range(x.size(-1)):
        if channel % 2 == 0:
            encodings[:, channel] = torch.sin(
                positions / 10000 ** (channel / x.size(2)))
        else:
            encodings[:, channel] = torch.cos(
                positions / 10000 ** ((channel - 1) / x.size(2)))
    return encodings


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResidualBlock(nn.Module):
    def __init__(self, layer, d_model, drop_ratio):
        super(ResidualBlock, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)

    def forward(self, *x):
        return self.layernorm(x[0] + self.dropout(self.layer(*x)))


class Attention(nn.Module):

    def __init__(self, d_key, drop_ratio, causal):
        super(Attention, self).__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal

    def forward(self, query, key, value):
        dot_products = torch.bmm(query, key.transpose(1, 2))
        if query.dim() == 3 and (self is None or self.causal):
            tri = torch.ones(key.size(1), key.size(1)).triu(1) * INF
            if key.is_cuda:
                tri = tri.cuda(key.get_device())
            dot_products.data.sub_(tri.unsqueeze(0))
        return torch.bmm(
            self.dropout(F.softmax(dot_products / self.scale, dim=-1)), value)


class MultiHead(nn.Module):
    def __init__(self, d_key, d_value, n_heads, drop_ratio, causal=False):
        super(MultiHead, self).__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal)
        self.wq = nn.Linear(d_key, d_key, bias=False)
        self.wk = nn.Linear(d_key, d_key, bias=False)
        self.wv = nn.Linear(d_value, d_value, bias=False)
        self.wo = nn.Linear(d_value, d_key, bias=False)
        self.n_heads = n_heads

    def forward(self, query, key, value):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (
            x.chunk(self.n_heads, -1) for x in (query, key, value))
        return self.wo(torch.cat([self.attention(q, k, v)
                                  for q, k, v in zip(query, key, value)], -1))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, n_heads, drop_ratio):
        super(EncoderLayer, self).__init__()
        self.selfattn = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio, causal=False),
            d_model, drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)

    def forward(self, x):
        return self.feedforward(self.selfattn(x, x, x))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, n_heads, drop_ratio):
        super(DecoderLayer, self).__init__()
        self.selfattn = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio, causal=True),
            d_model, drop_ratio)
        self.attention = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio),
            d_model, drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)

    def forward(self, x, encoding):
        """
        Args:
            x: (N, Lt, D)
            encoding: (N, Lv, D)
        """
        x = self.selfattn(x, x, x)  # (N, Lt, D)
        return self.feedforward(
            self.attention(x, encoding, encoding))  # (N, Lt, D)


class Encoder(nn.Module):
    def __init__(self, vfeat_size, d_model, d_hidden, n_layers, n_heads,
                 drop_ratio):
        super(Encoder, self).__init__()
        self.video_embeddings = nn.Sequential(
            LayerNorm(vfeat_size),
            nn.Dropout(drop_ratio),
            nn.Linear(vfeat_size, d_model)
        )
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_hidden, n_heads, drop_ratio)
             for _ in range(n_layers)])
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x, mask=None):
        """
        Args:
            x: (N, Lv, Dv)
            mask: (N, Lv)

        Returns:
        """
        x = self.video_embeddings(x)  # (N, Lv, D)
        x = x + positional_encodings_like(x)
        x = self.dropout(x)
        mask.unsqueeze_(-1)
        if mask is not None:
            x = x * mask
        encoding = []
        for layer in self.layers:
            x = layer(x)
            if mask is not None:
                x = x * mask
            encoding.append(x)
        return encoding


class Decoder(nn.Module):
    def __init__(self, d_model, d_hidden, vocab_size, n_layers, n_heads,
                 drop_ratio):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_hidden, n_heads, drop_ratio)
             for _ in range(n_layers)])
        self.out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(drop_ratio)
        self.d_model = d_model
        self.d_out = vocab_size

    def forward(self, x, encoding):
        """
        Args:
            x: (N, Lt)
            encoding: [(N, Lv, D), ] * num_hidden_layers
        """
        x = F.embedding(x, self.out.weight * math.sqrt(
            self.d_model))  # (N, Lt, D)
        x = x + positional_encodings_like(x)  # (N, Lt, D)
        x = self.dropout(x)  # (N, Lt, D)
        for layer, enc in zip(self.layers, encoding):
            x = layer(x, enc)  # (N, Lt, D)
        return x  # (N, Lt, D) at last layer


class MTransformer(nn.Module):
    def __init__(self, cfg: MartConfig):
        super().__init__()
        self.cfg = cfg
        vfeat_size = cfg.video_feature_size
        d_model = cfg.hidden_size  # 1024
        d_hidden = cfg.intermediate_size  # 2048
        n_layers = cfg.num_hidden_layers  # 6
        n_heads = cfg.num_attention_heads  # 8
        drop_ratio = cfg.hidden_dropout_prob  # 0.1
        self.vocab_size = cfg.vocab_size
        self.encoder = Encoder(vfeat_size, d_model, d_hidden, n_layers,
                               n_heads, drop_ratio)
        self.decoder = Decoder(d_model, d_hidden, self.vocab_size,
                               n_layers, n_heads, drop_ratio)
        self.loss_func = LabelSmoothingLoss(cfg.label_smoothing,
                                            cfg.vocab_size, ignore_index=-1)\
            if "label_smoothing" in cfg and cfg.label_smoothing > 0 else nn.CrossEntropyLoss(
            ignore_index=-1)

    def encode(self, video_features, video_masks):
        """
        Args:
            video_features: (N, Lv, Dv)
            video_masks: (N, Lv)  with 1 indicates valid bits
        """
        return self.encoder(video_features, video_masks)

    def decode(self, text_input_ids, _text_masks, text_input_labels,
               encoder_outputs, _video_masks):
        """
        Args:
            text_input_ids: (N, Lt)
            _text_masks: (N, Lt)  with 1 indicates valid bits, (UNUSED)
            text_input_labels: (N, Lt)  with `-1` on ignored positions
            encoder_outputs: (N, Lv, D)
            _video_masks: not used, leave here to maintain a common API with untied model (UNUSED)
        """
        # the triangular mask is generated and applied inside the attention module
        h = self.decoder(text_input_ids, encoder_outputs)  # (N, Lt, D)
        prediction_scores = self.decoder.out(h)  # (N, Lt, vocab_size)
        caption_loss = self.loss_func(
            prediction_scores.view(-1, self.cfg.vocab_size),
            text_input_labels.view(-1))  # float
        return caption_loss, prediction_scores

    def forward(self, video_features, video_masks, text_input_ids, text_masks,
                text_input_labels):
        """
        Args:
            video_features: (N, Lv, Dv)
            video_masks: (N, Lv)  with 1 indicates valid bits
            text_input_ids: (N, Lt)
            text_masks: (N, Lt)  with 1 indicates valid bits
            text_input_labels: (N, Lt)  with `-1` on ignored positions  (in some sense duplicate with text_masks)
        """
        encoder_layer_outputs = self.encode(video_features,
                                            video_masks)  # [(N, Lv, D), ] * num_hidden_layers
        caption_loss, prediction_scores = self.decode(
            text_input_ids, text_masks, text_input_labels,
            encoder_layer_outputs, None)  # float, (N, Lt, vocab_size)
        return caption_loss, prediction_scores
