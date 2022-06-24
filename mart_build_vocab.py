"""
Build vocabulary for MART.

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
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch as th
from tqdm import tqdm

from mart.recursive_caption_dataset import RecursiveCaptionDataset as RCDataset
from nntrainer.utils import TrainerPathConst


def load_glove(filename) -> Dict[str, th.FloatTensor]:
    """
    Returns:
        Dictionary of (word, embedding) pairs.
    """
    glove = {}
    with open(filename, encoding="utf-8") as f:
        for line in f.readlines():
            values = line.strip("\n").split(" ")  # space separator
            word = values[0]
            vector = np.asarray([float(e) for e in values[1:]])
            glove[word] = vector
    return glove


def extract_glove(word2idx, raw_glove_path, vocab_glove_path, glove_dim=300):
    # Make glove embedding.
    print(f"Loading glove embedding at path : {raw_glove_path}.")
    glove_full = load_glove(raw_glove_path)
    print("Glove Loaded, building word2idx, idx2word mapping.")
    idx2word = {v: k for k, v in list(word2idx.items())}

    glove_matrix = np.zeros([len(word2idx), glove_dim])
    glove_keys = list(glove_full.keys())
    for i in tqdm(list(range(len(idx2word)))):
        w = idx2word[i]
        w_embed = glove_full[w] if w in glove_keys else np.random.randn(
            glove_dim) * 0.4
        glove_matrix[i, :] = w_embed
    print("vocab embedding size is :", glove_matrix.shape)
    th.save(glove_matrix, vocab_glove_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dset_name", type=str)
    parser.add_argument("--cache", type=str, default="cache_caption")
    parser.add_argument("--annotations_dir", type=str, default=TrainerPathConst.DIR_ANNOTATIONS)
    parser.add_argument("--raw_glove_path", type=str, default="pretrained_models/glove.6B.300d.txt",
                        help="downloaded glove vectors path")
    args = parser.parse_args()

    # create cache dir
    os.makedirs(args.cache, exist_ok=True)
    # load word2idx
    word2idx_path = Path(args.annotations_dir) / args.dset_name / "mart_word2idx.json"
    print(f"Load {word2idx_path}")
    word2idx = json.load(word2idx_path.open("rt", encoding="utf8"))
    print(f"[Info] Trimmed vocabulary size = {len(word2idx)}, each with minimum occurrence = 3")

    glove_path = Path(args.cache) / f"{args.dset_name}_vocab_glove.pt"
    print(f"Extract embeddings from {glove_path}")
    extract_glove(word2idx, args.raw_glove_path, glove_path)


def _unused_build_vocab_idx(word_insts: List[List[str]], min_word_count) -> Dict[str, int]:
    """
    Build word2idx for a new dataset.

    Do not rebuild ActivityNet or YouCook2 with this function! The existing word2idx files are randomly sorted
    and cannot be reproduced. If you rebuild them, the provided models will output garbage.

    Args:
        word_insts: List of list of words.
        min_word_count:

    Returns:
        Word to index dictionary.
    """
    full_vocab = list(sorted(set(w for sent in word_insts for w in sent)))
    print(("[Info] Original Vocabulary size =", len(full_vocab)))

    word2idx = {
        RCDataset.PAD_TOKEN: RCDataset.PAD,
        RCDataset.CLS_TOKEN: RCDataset.CLS,
        RCDataset.SEP_TOKEN: RCDataset.SEP,
        RCDataset.VID_TOKEN: RCDataset.VID,
        RCDataset.BOS_TOKEN: RCDataset.BOS,
        RCDataset.EOS_TOKEN: RCDataset.EOS,
        RCDataset.UNK_TOKEN: RCDataset.UNK,
    }

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in list(word_count.items()):
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print(("[Info] Trimmed vocabulary size = {},".format(len(word2idx)),
           "each with minimum occurrence = {}".format(min_word_count)))
    print(("[Info] Ignored word count = {}".format(ignored_word_count)))
    return word2idx


if __name__ == "__main__":
    main()
