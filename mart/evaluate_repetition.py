"""
Evaluate repetition metric R@4, lower is better.

References:
    Copyright (c) 2017 Ranjay Krishna
    Licensed under The MIT License, see https://choosealicense.com/licenses/mit/
    @inproceedings{krishna2017dense,
        title={Dense-Captioning Events in Videos},
        author={Krishna, Ranjay and Hata, Kenji and Ren, Frederic and Fei-Fei, Li and Niebles, Juan Carlos},
        booktitle={ArXiv},
        year={2017}
    }

    History:
    https://github.com/ranjaykrishna/densevid_eval
    https://github.com/jamespark3922/densevid_eval
    https://github.com/jayleicn/recurrent-transformer
    Current version 2021 https://github.com/gingsi/coot-videotext
"""

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np


def get_ngrams(words_pred, unigrams, bigrams, trigrams, fourgrams):
    # N=1
    for w in words_pred:
        if w not in unigrams:
            unigrams[w] = 0
        unigrams[w] += 1
    # N=2
    for i, w in enumerate(words_pred):
        if i < len(words_pred) - 1:
            w_next = words_pred[i + 1]
            bigram = '%s_%s' % (w, w_next)
            if bigram not in bigrams:
                bigrams[bigram] = 0
            bigrams[bigram] += 1
    # N=3
    for i, w in enumerate(words_pred):
        if i < len(words_pred) - 2:
            w_next = words_pred[i + 1]
            w_next_ = words_pred[i + 2]
            tri = '%s_%s_%s' % (w, w_next, w_next_)
            if tri not in trigrams:
                trigrams[tri] = 0
            trigrams[tri] += 1
    # N=4
    for i, w in enumerate(words_pred):
        if i < len(words_pred) - 3:
            w_next = words_pred[i + 1]
            w_next_ = words_pred[i + 2]
            w_next__ = words_pred[i + 3]
            four = '%s_%s_%s_%s' % (w, w_next, w_next_, w_next__)
            if four not in fourgrams:
                fourgrams[four] = 0
            fourgrams[four] += 1
    return unigrams, bigrams, trigrams, fourgrams


def evaluate_repetition(data_predicted, data_gt, verbose=False):
    if verbose:
        print('#### Per video ####')

    num_pred = len(data_predicted)
    num_gt = len(data_gt)
    num_evaluated = 0

    re1 = []
    re2 = []
    re3 = []
    re4 = []

    for vid in data_gt:
        unigrams = {}
        bigrams = {}
        trigrams = {}
        fourgrams = {}

        # skip non-existing videos
        if vid not in data_predicted:
            continue

        num_evaluated += 1
        for e in data_predicted[vid]:
            pred_sentence = e["sentence"]

            if len(pred_sentence) > 0:
                if pred_sentence[-1] == '.':
                    pred_sentence = pred_sentence[0:-1]
                while pred_sentence[-1] == ' ':
                    pred_sentence = pred_sentence[0:-1]
                pred_sentence = pred_sentence.replace(',', ' ')
            while '  ' in pred_sentence:
                pred_sentence = pred_sentence.replace('  ', ' ')

            words_pred = pred_sentence.split(' ')
            unigrams, bigrams, trigrams, fourgrams = get_ngrams(words_pred, unigrams, bigrams, trigrams, fourgrams)

        sum_re1 = float(sum([unigrams[f] for f in unigrams]))
        sum_re2 = float(sum([bigrams[f] for f in bigrams]))
        sum_re3 = float(sum([trigrams[f] for f in trigrams]))
        sum_re4 = float(sum([fourgrams[f] for f in fourgrams]))

        vid_re1 = float(sum([max(unigrams[f] - 1, 0) for f in unigrams])) / sum_re1 if sum_re1 != 0 else 0
        vid_re2 = float(sum([max(bigrams[f] - 1, 0) for f in bigrams])) / sum_re2 if sum_re2 != 0 else 0
        vid_re3 = float(sum([max(trigrams[f] - 1, 0) for f in trigrams])) / sum_re3 if sum_re3 != 0 else 0
        vid_re4 = float(sum([max(fourgrams[f] - 1, 0) for f in fourgrams])) / sum_re4 if sum_re4 != 0 else 0

        re1.append(vid_re1)
        re2.append(vid_re2)
        re3.append(vid_re3)
        re4.append(vid_re4)

    repetition_scores = dict(
        re1=np.mean(re1),
        re2=np.mean(re2),
        re3=np.mean(re3),
        re4=np.mean(re4),
        num_pred=num_pred,
        num_gt=num_gt,
        num_evaluated=num_evaluated
    )
    return repetition_scores


def evaluate_repetition_files(submission_file: str, reference_file: str, output_file: Optional[Union[str, Path]] = None,
                              verbose: bool = False):
    # load input data
    sub_data = json.load(open(submission_file, "r"))
    ref_data = json.load(open(reference_file, "r"))
    sub_data = sub_data["results"] if "results" in sub_data else sub_data
    ref_data = ref_data["results"] if "results" in ref_data else ref_data

    # evaluate repetition
    rep_scores = evaluate_repetition(sub_data, ref_data)

    # convert result to json
    rep_scores_str = json.dumps(rep_scores, indent=4, sort_keys=True)

    # print output
    if verbose:
        print(("Repetition Metrics {}".format(rep_scores_str)))

    # save output to file
    if output_file is not None:
        with open(output_file, "w", encoding="utf8") as f:
            f.write(json.dumps(rep_scores, indent=4, sort_keys=True))

    return rep_scores
