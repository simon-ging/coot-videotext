"""
Compute Bleu, CIDEr-D, Rouge-L, METEOR metrics using package pycocoevalcap.

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
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
from pycocoevalcap.bleu.bleu import BleuScorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def parse_sent(sent):
    res = re.sub('[^a-zA-Z]', ' ', sent)
    res = res.strip().lower().split()
    return res


def parse_para(para):
    para = para.replace('..', '.')
    para = para.replace('.', ' endofsent')
    return parse_sent(para)


class CaptionEvaluator:
    """
    Evaluate model output and ground truth to get captioning stats.

    This is called ANETcaptions but also works for YouCook2.
    """

    def __init__(self, ground_truth_filenames, prediction_filename, verbose=False, all_scorer=False):
        # Check that the gt and submission files exist and load them
        self.verbose = verbose
        self.all_scorer = all_scorer
        self.ground_truths = self.import_ground_truths(ground_truth_filenames)
        self.prediction = self.import_prediction(prediction_filename)
        self.tokenizer = PTBTokenizer()

        # Set up scorers, if not verbose, we only use the one we're
        # testing on: METEOR

        # Meteor is java-based and can crash alot.
        try:
            met = Meteor()
        except (AttributeError, FileNotFoundError) as e:
            print(f"Meteor couldn't start due to {e}")
            met = None

        if self.verbose or self.all_scorer:
            self.scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (met, "METEOR"),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr")
            ]
        else:
            self.scorers = [(met, "METEOR")]

        # init some attributes
        self.easy_samples = {}
        self.hard_samples = {}
        self.n_ref_vids = set()
        self.scores = {}

    def ensure_caption_key(self, data):
        if len(data) == 0:
            return data
        if not list(data.keys())[0].startswith('v_'):
            data = {'v_' + k: data[k] for k in data}
        return data

    def import_prediction(self, prediction_filename):
        if self.verbose:
            print(("| Loading submission... {}".format(prediction_filename)))
        submission = json.load(open(prediction_filename))['results']

        # change to paragraph format
        para_submission = {}
        for idx in list(submission.keys()):
            para_submission[idx] = ''
            for info in submission[idx]:
                para_submission[idx] += info['sentence'] + '. '
        for para in list(para_submission.values()):
            assert (type(para) == str or type(para) == str)
        # Ensure that every video is limited to the correct maximum number of proposals.
        return self.ensure_caption_key(para_submission)

    def import_ground_truths(self, filenames):
        gts = []
        self.n_ref_vids = set()
        for filename in filenames:
            gt = json.load(open(filename))
            self.n_ref_vids.update(list(gt.keys()))
            gts.append(self.ensure_caption_key(gt))
        if self.verbose:
            print(("| Loading GT. #files: %d, #videos: %d" % (
                len(filenames), len(self.n_ref_vids))))
        return gts

    def check_gt_exists(self, vid_id):
        for gt in self.ground_truths:
            if vid_id in gt:
                return True
        return False

    def get_gt_vid_ids(self):
        vid_ids = set([])
        for gt in self.ground_truths:
            vid_ids |= set(gt.keys())
        return list(vid_ids)

    def evaluate(self):
        self.scores = self.evaluate_para()

    def evaluate_para(self):
        # This method averages the tIoU precision from METEOR, Bleu, etc. across videos
        gt_vid_ids = self.get_gt_vid_ids()
        vid2idx = {k: i for i, k in enumerate(gt_vid_ids)}
        gts = {vid2idx[k]: [] for k in gt_vid_ids}
        for i, gt in enumerate(self.ground_truths):
            for k in gt_vid_ids:
                if k not in gt:
                    continue
                gts[vid2idx[k]].append(' '.join(parse_sent(gt[k])))
        res = {
            vid2idx[k]: [' '.join(parse_sent(self.prediction[k]))] if k in self.prediction and len(
                self.prediction[k]) > 0 else [''] for k in gt_vid_ids}
        para_res = {
            vid2idx[k]: [' '.join(parse_para(self.prediction[k]))] if k in self.prediction and len(
                self.prediction[k]) > 0 else [''] for k in gt_vid_ids}

        # Each scorer will compute across all videos and take average score
        output = {}
        num = len(res)
        hard_samples = {}
        easy_samples = {}
        for scorer, method in self.scorers:
            if scorer is None:
                print(f"Scorer {type(scorer)} doesn't exist (probably crashed at startup).")
                score = -999
                scores = [-999] * len(gts)
            else:
                if self.verbose:
                    print(('computing %s score...' % (scorer.method())))
                if method != 'Self_Bleu':
                    try:
                        score, scores = scorer.compute_score(gts, res)
                    except (ValueError, FileNotFoundError, AttributeError) as e:
                        if isinstance(scorer, Meteor):
                            # if meteor crashes (java problems on certrain systems), report -999 score instead of dying.
                            print(f"Scorer {type(scorer)} crashed with {e}.")
                            # meteor freezes on crash, unfreeze the thread.
                            try:
                                scorer.lock.release()
                            except AttributeError:
                                pass
                        else:
                            # it's not OK for other scorers to crash.
                            raise e
                        score = -999
                        scores = [-999] * len(gts)
                else:
                    score, scores = scorer.compute_score(gts, para_res)
            scores = np.asarray(scores)

            if type(method) == list:
                for m in range(len(method)):
                    output[method[m]] = score[m]
                    if self.verbose:
                        print(("%s: %0.3f" % (method[m], output[method[m]])))
                for m, i in enumerate(scores.argmin(1)):
                    if i not in hard_samples:
                        hard_samples[i] = []
                    hard_samples[i].append(method[m])
                for m, i in enumerate(scores.argmax(1)):
                    if i not in easy_samples:
                        easy_samples[i] = []
                    easy_samples[i].append(method[m])
            else:
                output[method] = score
                if self.verbose:
                    print(("%s: %0.3f" % (method, output[method])))
                # i = scores.argmin()
                # if i not in hard_samples:
                #     hard_samples[i] = []
                # hard_samples[i].append(method)
                # i = scores.argmax()
                # if i not in easy_samples:
                #     easy_samples[i] = []
                # easy_samples[i].append(method)
        if self.verbose:
            print(f'# scored video = {num}')

        self.hard_samples = {gt_vid_ids[i]: v for i, v in
                             list(hard_samples.items())}
        self.easy_samples = {gt_vid_ids[i]: v for i, v in
                             list(easy_samples.items())}
        return output


def evaluate_language_files(submission_file, references_files, output_file: Optional[Union[str, Path]] = None,
                            verbose=False,
                            all_scorer=True):
    # print(f"Refs {references_files} sub {submission_file}")
    evaluator = CaptionEvaluator(ground_truth_filenames=references_files, prediction_filename=submission_file,
                                 verbose=verbose, all_scorer=all_scorer)
    evaluator.evaluate()
    scores = evaluator.scores
    if output_file is not None:
        with Path(output_file).open("wt", encoding="utf8") as fh:
            json.dump(scores, fh)
    return scores


class Bleu:
    """
    Custom, less verbose implementation of pycocoevalcap.bleu.bleu.Bleu
    """

    def __init__(self, n=4, verbose=0):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}
        self.verbose = verbose

    def compute_score(self, gts, res):
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for idx in imgIds:
            hypo = res[idx]
            ref = gts[idx]

            # Sanity check.
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)

        # score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='closest', verbose=self.verbose)
        # score, scores = bleu_scorer.compute_score(option='average', verbose=1)

        # return (bleu, bleu_info)
        return score, scores

    def method(self):
        return "Bleu"
