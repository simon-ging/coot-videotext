"""
Retrieval dataset.

Notes:
    Masks are bools with 0 = token, 1 = padding.


## Preprocessing

Activitynet: Switch start and stop timestamps when stop > start. Affects 2 videos.,
Convert start/stop timestamps to start/stop frames by multiplying with FPS in the features
and using floor/ceil+2 operation respectively. Expand clips to be at least 10 frames long

All: Add [CLS] at the start of each paragraph and [SEP] at the end
of each sentence before encoding with Bert model.
New updated text processing with capitalization, dots, cased models that perform better than the
`bert-base-uncased` model we used in the paper.
"""

import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch as th
from torch.utils import data as th_data

import coot.configs_retrieval
from coot.configs_retrieval import RetrievalDatasetConfig
from coot.features_loader import TextFeaturesLoader, VideoFeatureLoader
from nntrainer import data as nn_data, data_text, maths, typext, utils, utils_torch


# ---------- Single datapoint and batch definition ----------


class RetrievalDataPointTuple(typext.TypedNamedTuple):
    """
    Definition of a single datapoint.
    """
    key: str
    data_key: str
    sentences: List[str]
    vid_feat: th.Tensor  # shape (num_feat, vid_feat_dim)
    vid_feat_len: int
    par_feat: th.Tensor  # shape (num_tokens, text_feat_dim)
    par_feat_len: int
    clip_num: int
    clip_feat_list: List[th.Tensor]  # shapes (num_feat_clip, vid_feat_dim)
    clip_feat_len_list: List[int]
    sent_num: int
    sent_feat_list: List[th.Tensor]  # shapes (num_tokens_sent, text_feat_dim)
    sent_feat_len_list: List[int]

    # shape tests for tensors
    _shapes_dict = {
        "vid_feat": (None, None),
        "par_feat": (None, None),
        "clip_feat_list": (None, None),
        "sent_feat_list": (None, None)
    }


class RetrievalDataBatchTuple(typext.TypedNamedTuple):
    """
    Definition of a batch.
    """
    key: List[str]
    data_key: List[str]
    sentences: List[List[str]]
    vid_feat: th.Tensor  # shape (batch_size, max_num_feat, vid_feat_dim) dtype float
    vid_feat_mask: th.Tensor  # shape (batch_size, max_num_feat) dtype bool
    vid_feat_len: th.Tensor  # shape (batch_size) dtype long
    par_feat: th.Tensor  # shape (batch_size, max_num_tokens, text_feat_dim) dtype float
    par_feat_mask: th.Tensor  # shape (batch_size, max_num_tokens) dtype bool
    par_feat_len: th.Tensor  # shape (batch_size) dtype long
    clip_num: th.Tensor  # shape (batch_size) dtype long
    clip_feat: th.Tensor  # shapes (total_num_clips, max_num_feat_clip, vid_feat_dim) dtype float
    clip_feat_mask: th.Tensor  # shapes (total_num_clips, max_num_feat_clip) dtype bool
    clip_feat_len: th.Tensor  # shapes (total_num_clips) dtype long
    sent_num: th.Tensor  # shape (batch_size) dtype long
    sent_feat: th.Tensor  # shapes (total_num_sents, max_num_feat_sent, text_feat_dim) dtype float
    sent_feat_mask: th.Tensor  # shapes (total_num_sents, max_num_feat_sent) dtype bool
    sent_feat_len: th.Tensor  # shapes (total_num_sents) dtype long

    # shape tests for tensors
    _shapes_dict = {
        "vid_feat": (None, None, None),
        "vid_feat_mask": (None, None),
        "vid_feat_len": (None,),
        "par_feat": (None, None, None),
        "par_feat_mask": (None, None),
        "par_feat_len": (None,),
        "clip_num": (None,),
        "clip_feat": (None, None, None),
        "clip_feat_mask": (None, None),
        "clip_feat_len": (None,),
        "sent_num": (None,),
        "sent_feat": (None, None, None),
        "sent_feat_mask": (None, None),
        "sent_feat_len": (None,),
    }


class RetrievalDataset(th_data.Dataset):
    """
    Dataset for retrieval.

    Args:
        cfg: Dataset configuration class.
        path_data: Dataset base path.
        verbose: Print output (cannot use logger in multiprocessed torch Dataset class)
    """

    def __init__(self, cfg: RetrievalDatasetConfig, path_data: Union[str, Path], *,
                 verbose: bool = False):
        # store config
        self.path_data = Path(path_data)
        self.cfg = cfg
        self.split = cfg.split
        self.verbose = verbose
        self.is_train = self.split == nn_data.DataSplitConst.TRAIN

        # setup paths
        self.path_dataset = self.path_data / self.cfg.name

        # load metadata
        raw_meta_file = (self.path_dataset / f"meta_{cfg.metadata_name}.json")
        raw_meta = json.load(raw_meta_file.open("rt", encoding="utf8"))

        # get keys to load
        if self.cfg.subset == utils.DEFAULT:
            self.keys = [key for key, val in raw_meta.items() if val["split"] == self.split]
        else:
            raise NotImplementedError("Load created subsets, do cross-val, ...")

        if len(self.keys) == 0:
            available_splits = sorted(set(val["split"] for val in raw_meta.values()))
            raise ValueError(f"No datapoints found in split: {self.split}! Found splits: {available_splits} "
                             f"in file: {raw_meta_file}")

        # reduce dataset size if request
        if cfg.max_datapoints > -1:
            self.keys = self.keys[:cfg.max_datapoints]
            print(f"Reduced number of datapoints to {len(self.keys)}")

        # For each key (datapoint) get the data_key (reference to the video file).
        # A single video can appear in multiple datapoints.
        self.data_keys = [raw_meta[key]["data_key"] for key in self.keys]

        # load video features
        self.vid_feats = VideoFeatureLoader(
            self.path_dataset, self.cfg.vid_feat_name, self.cfg.vid_feat_source, self.data_keys,
            preload_vid_feat=self.cfg.preload_vid_feat)

        # build metadata for this split
        self.meta = {}
        expansions = 0
        num_segments = 0
        for key, data_key in zip(self.keys, self.data_keys):
            # add all original metadata of the video
            self.meta[key] = raw_meta[key]
            # read frames information, set video start and stop frames
            num_frames = self.vid_feats.num_frames[data_key]
            self.meta[key]["start_frame_vid"] = 0
            self.meta[key]["stop_frame_vid"] = num_frames
            self.meta[key]["num_frames_vid"] = self.meta[key]["stop_frame_vid"] - self.meta[key]["start_frame_vid"]

            # loop segments, calculate segment start and stop frames
            fps = num_frames / self.meta[key]["duration_sec"]
            for seg in self.meta[key]["segments"]:
                # calculate start and stop frame number for each segment
                time_start, time_stop = seg["start_sec"], seg["stop_sec"]
                if time_stop < time_start:
                    print(f"switch: time_stop {time_stop} > time_start {time_start}")
                    time_start, time_stop = time_stop, time_start
                    print(f"done switch: time_stop {time_stop} > time_start {time_start}")
                start_frame = int(np.floor(fps * time_start))
                # stop frame is exclusive now, so set it to ceil() + 2 to not miss stuff.
                stop_frame = int(np.ceil(fps * time_stop)) + self.cfg.add_stop_frame

                # make sure to not exceed the last frame
                if stop_frame > num_frames:
                    stop_frame = num_frames

                # expand the segment to meet minimum frames
                start_frame, stop_frame, changed = maths.expand_video_segment(
                    num_frames, self.cfg.expand_segments, start_frame, stop_frame)
                if changed:
                    expansions += 1

                seg["start_frame"] = start_frame
                seg["num_frames"] = stop_frame - start_frame
                num_segments += 1
        print(f"Built metadata for {self.split}: {len(self.keys)} datapoints, {num_segments} segments. "
              f"Expanded {expansions} segments.")

        # load text features
        self.text_feats = TextFeaturesLoader(
            self.path_dataset, f"{self.cfg.text_feat_name}", self.cfg.text_feat_source, self.keys,
            preload_text_feat=self.cfg.preload_text_feat)

        # load preprocessing function for text
        self.text_preproc_func = data_text.get_text_preprocessor(self.cfg.text_preprocessing)

    def get_vid_frames_by_indices(self, key: str, indices: List[int]) -> np.ndarray:
        """
        Load frames of a video given by indices.

        Args:
            key: Video key.
            indices: List of frame indices.

        Returns:
            Frame features with shape (len(indices), feature_dim)
        """
        data_key = self.meta[key]["data_key"]
        return self.vid_feats[data_key][indices]

    def get_vid_feat_by_amount(self, key: str, num_frames: int) -> np.ndarray:
        """
        Load a given number of frames from a video.

        Args:
            key: Video key.
            num_frames: Number of frames desired

        Returns:
            Frame features with shape (num_frames, feature_dim)
        """
        indices = maths.compute_indices(self.meta[key]["num_frames_vid"], num_frames, self.is_train)
        indices += self.meta[key]["start_frame_vid"]
        return self.get_vid_frames_by_indices(key, indices)

    def get_clip_frames_by_amount(self, key: str, seg_num: int, num_frames: int) -> np.ndarray:
        """
        Load a given number of frames from a clip.

        Args:
            key: Video key.
            seg_num: Segment number.
            num_frames: Number of frames desired.

        Returns:
            Frame features with shape (num_frames, feature_dim)
        """
        seg = self.meta[key]["segments"][seg_num]
        indices = maths.compute_indices(seg["num_frames"], num_frames, self.is_train)
        indices += seg["start_frame"]
        return self.get_vid_frames_by_indices(key, indices)

    def __len__(self) -> int:
        """
        Return dataset length.

        Returns:
            Dataset length.
        """
        return len(self.keys)

    def __getitem__(self, item: int) -> RetrievalDataPointTuple:
        """
        Return a single datapoint.

        Args:
            item: Item number.

        Returns:
            Tuple of all required data.
        """
        key = self.keys[item]
        data_key = self.meta[key]["data_key"]
        vid_dict = self.meta[key]

        # load number of clips and sentences (currently 1-1 correspondence)
        clip_num = len(vid_dict["segments"])
        sent_num = clip_num

        # ---------- load video frames ----------
        vid_feat_len = vid_dict["num_frames_vid"]
        if vid_feat_len > self.cfg.max_frames:
            vid_feat_len = self.cfg.max_frames
        vid_feat = th.Tensor(self.get_vid_feat_by_amount(key, vid_feat_len))
        assert vid_feat_len == int(vid_feat.shape[0])

        if self.cfg.frames_noise != 0:
            # add noise to frames if needed
            vid_frames_noise = utils_torch.get_truncnorm_tensor(vid_feat.shape, std=self.cfg.frames_noise)
            vid_feat += vid_frames_noise

        # ---------- load clip frames ----------
        clip_feat_list = []
        clip_feat_len_list = []
        for i, seg in enumerate(vid_dict["segments"]):
            c_num_frames = seg["num_frames"]
            if c_num_frames > self.cfg.max_frames:
                c_num_frames = self.cfg.max_frames
            c_frames = self.get_clip_frames_by_amount(key, i, c_num_frames)
            c_frames = th.Tensor(c_frames)
            if self.cfg.frames_noise != 0:
                # add noise to frames if needed
                clip_frames_noise = utils_torch.get_truncnorm_tensor(c_frames.shape, std=self.cfg.frames_noise)
                c_frames += clip_frames_noise
            clip_feat_list.append(c_frames)
            clip_feat_len_list.append(c_frames.shape[0])

        # ---------- load text as string ----------
        seg_narrations = []
        for seg in vid_dict["segments"]:
            seg_narr = seg["text"]
            if seg_narr is None:
                seg_narr = "undefined"
                print("WARNING: Undefined text tokens (no narration data, is this a test set?)")
            seg_narrations.append(seg_narr)
        sentences = self.text_preproc_func(seg_narrations)

        # ---------- load text features ----------
        par_feat, sent_feat_len_list = self.text_feats[key]
        par_feat_len = int(par_feat.shape[0])
        par_feat = th.Tensor(par_feat).float()

        # split paragraph features into sentences
        sent_feat_list = []
        pointer = 0
        for i, sent_cap_len in enumerate(sent_feat_len_list):
            sent_feat = par_feat[pointer:pointer + sent_cap_len, :]
            sent_feat_list.append(sent_feat)
            pointer += sent_cap_len

        # return single datapoint
        return RetrievalDataPointTuple(
            key, data_key, sentences, vid_feat, vid_feat_len, par_feat, par_feat_len, clip_num,
            clip_feat_list, clip_feat_len_list, sent_num, sent_feat_list, sent_feat_len_list)

    def collate_fn(self, data_batch: List[RetrievalDataPointTuple]):
        """
        Collate the single datapoints above. Custom collation needed since sequences have different length.

        Returns:
        """
        batch_size = len(data_batch)
        key: List[str] = [d.key for d in data_batch]
        data_key: List[str] = [d.data_key for d in data_batch]

        # store input text: for each video, each sentence, store each word as a string
        sentences: List[str] = [d.sentences for d in data_batch]

        # ---------- collate video features ----------

        # read video features list
        list_vid_feat = [d.vid_feat for d in data_batch]
        vid_feat_dim: int = list_vid_feat[0].shape[-1]

        # read video sequence lengths
        list_vid_feat_len = [d.vid_feat_len for d in data_batch]
        vid_feat_len = th.Tensor(list_vid_feat_len).long()
        vid_feat_max_len = int(vid_feat_len.max().numpy())

        # put all video features into a batch, masking / padding as necessary
        vid_feat = th.zeros(batch_size, vid_feat_max_len, vid_feat_dim).float()
        vid_feat_mask = th.ones(batch_size, vid_feat_max_len).bool()
        for batch, (seq_len, item) in enumerate(zip(list_vid_feat_len, list_vid_feat)):
            vid_feat[batch, :seq_len] = item
            vid_feat_mask[batch, :seq_len] = 0

        # ---------- collate paragraph features ----------

        # read paragraph features list
        list_par_feat = [d.par_feat for d in data_batch]
        par_feat_dim: int = list_par_feat[0].shape[-1]

        # read paragraph sequence lengths
        list_par_feat_len = [d.par_feat_len for d in data_batch]
        par_feat_len = th.Tensor(list_par_feat_len).long()
        par_feat_max_len = int(par_feat_len.max().numpy())

        # put all paragraph features into a batch, masking / padding as necessary
        par_feat = th.zeros(batch_size, par_feat_max_len, par_feat_dim).float()
        par_feat_mask = th.ones(batch_size, par_feat_max_len).bool()
        for batch, (seq_len, item) in enumerate(zip(list_par_feat_len, list_par_feat)):
            par_feat[batch, :seq_len, :] = item
            par_feat_mask[batch, :seq_len] = 0

        # ---------- collate clip features ----------

        # read list of list of clip features (features for each clip in each video)
        list_clip_feat_list = [d.clip_feat_list for d in data_batch]

        # read number of clips per video into tensor
        list_clip_num = [d.clip_num for d in data_batch]
        clip_num = th.Tensor(list_clip_num).long()
        total_clip_num = int(np.sum(list_clip_num))

        # read list of list of clip lengths (number of features for each clip in each video)
        list_clip_feat_len_list = [d.clip_feat_len_list for d in data_batch]

        # get length of longest clip in all videos
        clip_feat_max_len = int(np.max([np.max(len_single) for len_single in list_clip_feat_len_list]))

        # begin collation: create flattened tensor to store all clips
        clip_feat = th.zeros((total_clip_num, clip_feat_max_len, vid_feat_dim)).float()
        clip_feat_mask = th.ones((total_clip_num, clip_feat_max_len)).bool()

        # collate clips, create masks and store lengths of individual clips
        clip_feat_len_list = []
        c_num = 0
        # loop videos
        for batch, clip_feat_list in enumerate(list_clip_feat_list):
            # loop clips in the video
            for _i, clip_feat_item in enumerate(clip_feat_list):
                # get sequence length of current clip
                clip_feat_item_len = int(clip_feat_item.shape[0])
                # fill collation tensor and mask, store length
                clip_feat[c_num, :clip_feat_item_len, :] = clip_feat_item
                clip_feat_mask[c_num, :clip_feat_item_len] = 0
                clip_feat_len_list.append(clip_feat_item_len)
                # increase clip counter
                c_num += 1
        # convert stored lengths to tensor
        clip_feat_len = th.Tensor(clip_feat_len_list).long()

        # ---------- collate sentence features ----------
        # sentences will be pieced together from the paragraph features

        # read number of sentences per paragraph into tensor
        list_sent_num = [d.sent_num for d in data_batch]
        sent_num = th.Tensor(list_sent_num).long()
        total_sent_num = int(np.sum(list_sent_num))

        # read list of list of sentence lengths (number of features for each sentence in each paragraph)
        list_sent_feat_len_list = [d.sent_feat_len_list for d in data_batch]

        # get length of longest sentence in all paragraphs
        sent_feat_max_len = int(np.max([np.max(len_single) for len_single in list_sent_feat_len_list]))

        # begin collation: create flattened tensor to store all sentences
        sent_feat = th.zeros((total_sent_num, sent_feat_max_len, par_feat_dim)).float()
        sent_feat_mask = th.ones((total_sent_num, sent_feat_max_len)).bool()

        # collate sentences, create masks and store lengths of individual sentences
        sent_feat_len_list = []
        s_num = 0
        # loop paragraphs
        for batch, sent_cap_len_list in enumerate(list_sent_feat_len_list):
            # loop sentences in the video
            pointer = 0
            for sent_cap_len_item in sent_cap_len_list:
                # read sentence features out of the entire paragraph features
                single_sent_feat = par_feat[batch, pointer:pointer + sent_cap_len_item]
                # fill collation tensor and mask, store length
                sent_feat[s_num, :sent_cap_len_item] = single_sent_feat
                sent_feat_mask[s_num, :sent_cap_len_item] = 0
                sent_feat_len_list.append(sent_cap_len_item)
                # increase sentence counter and pointer in the paragraph
                s_num += 1
                pointer += sent_cap_len_item
        # convert stored lengths to tensor
        sent_feat_len = th.Tensor(sent_feat_len_list).long()

        ret = RetrievalDataBatchTuple(
            key, data_key, sentences, vid_feat, vid_feat_mask, vid_feat_len, par_feat, par_feat_mask, par_feat_len,
            clip_num, clip_feat, clip_feat_mask, clip_feat_len, sent_num, sent_feat, sent_feat_mask, sent_feat_len)
        return ret


def create_retrieval_datasets_and_loaders(cfg: coot.configs_retrieval.RetrievalConfig, path_data: Union[str, Path]) -> (
        Tuple[RetrievalDataset, RetrievalDataset, th_data.DataLoader, th_data.DataLoader]):
    """
    Create training and validation datasets and dataloaders for retrieval.

    Args:
        cfg: Experiment configuration class.
        path_data: Dataset base path.

    Returns:
        Tuple of:
            Train dataset.
            Val dataset.
            Train dataloader.
            Val dataloader.
    """
    train_set = RetrievalDataset(cfg.dataset_train, path_data)
    train_loader = nn_data.create_loader(
        train_set, cfg.dataset_train, cfg.train.batch_size, collate_fn=train_set.collate_fn)
    val_set = RetrievalDataset(cfg.dataset_val, path_data)
    val_loader = nn_data.create_loader(
        val_set, cfg.dataset_val, cfg.val.batch_size, collate_fn=val_set.collate_fn)
    return train_set, val_set, train_loader, val_loader


def run_retrieval_dataset_test(train_set: RetrievalDataset, train_loader: th_data.DataLoader) -> None:
    """
    Helper code for checking the output of this dataset.

    Args:
        train_set: Training dataset.
        train_loader: Training dataloader.
    """
    print("---------- Testing dataset ----------")
    print(f"Length {len(train_set)}")

    # print one batch of data and exit
    for i, batch in enumerate(train_loader):  # type: RetrievalDataBatchTuple
        print("batch number:", i)
        for field, value in batch.dict().items():
            print(f"{field}:", end=" ")
            if isinstance(value, th.Tensor):
                print(value.shape, value.dtype)
            else:
                print(str(value)[:70], "..." if len(str(value)) > 70 else "")
        break
