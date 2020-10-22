import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

import utils


def _load_features_acitivitynet(file):
    return np.load(str(file))["frame_scores"].squeeze(1).squeeze(
        2).squeeze(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataroot", type=str, default="data",
        help="change datasets root path")
    args = parser.parse_args()

    dataset_path = Path(args.dataroot) / "activitynet"
    captions_path = dataset_path / "captions"
    features_path = (dataset_path / "features" /
                     "ICEP_V3_global_pool_skip_8_direct_resize")
    frame_lens_file = dataset_path / "features_data_shapes.json"
    meta_file = dataset_path / "meta_default.json"
    val_split = "val_1"

    # check which features exists
    feature_keys = []
    for file_name in sorted(os.listdir(str(features_path))):
        id_ = file_name.split(".npz")[0]
        feature_keys.append(id_)
    print(f"found features for {len(feature_keys)} videos")
    assert len(feature_keys) == 19994

    # read frame lengths if not exists
    if not frame_lens_file.is_file():
        print("check frame lengths...")
        frame_lens = {}
        for id_ in tqdm(feature_keys):
            file = features_path / f"{id_}.npz"
            data = _load_features_acitivitynet(file)
            frame_lens[id_] = data.shape[0]
        json.dump(frame_lens, frame_lens_file.open("wt", encoding="utf8"))
    frame_lens = json.load(frame_lens_file.open("rt", encoding="utf8"))
    print(f"Read {len(frame_lens)} frame lengths")

    # build metadata
    meta = {}
    expand_seg = 10
    regex_replace_newlines = re.compile(r"\s+")
    expanded = 0
    n_seg = 0
    for split in ["train", val_split]:
        with (captions_path / f"{split}.json").open(
                "rt", encoding="utf8") as fh:
            raw_data = json.load(fh)
        for key, val in raw_data.items():
            # load video information
            timestamps = val["timestamps"]
            sentences = val["sentences"]
            duration_sec = val["duration"]
            num_frames = frame_lens[key]
            # build segments
            segments = []
            for num_seg in range(len(timestamps)):
                # load narration sentence and preprocess line separators
                sentence = sentences[num_seg]
                sentence = regex_replace_newlines.sub(" ", sentence)
                # load start and stop timestamps
                start_sec = timestamps[num_seg][0]
                stop_sec = timestamps[num_seg][1]
                if stop_sec < start_sec:
                    print(
                        f"switch: stop_sec {stop_sec} > start_sec {start_sec}")
                    temp_ms = start_sec
                    start_sec = stop_sec
                    stop_sec = temp_ms
                # calculate start and stop frame
                start_frame = int(
                    np.floor(start_sec / duration_sec * num_frames))
                stop_frame = int(
                    np.ceil(stop_sec / duration_sec * num_frames))
                for i in range(2):
                    if stop_frame >= num_frames:
                        stop_frame -= 1
                start_frame, stop_frame, changes = utils.expand_segment(
                    num_frames, expand_seg, start_frame, stop_frame)
                if changes:
                    expanded += 1
                n_seg += 1
                num_frames_seg = stop_frame - start_frame + 1
                # save segment
                segment = {
                    "narration": sentence,
                    "start_frame": start_frame,
                    "num_frames": num_frames_seg
                }
                segments.append(segment)
            # save video
            meta[key] = {
                "split": split,
                "segments": segments,
                "num_frames": num_frames
            }
    print(f"expanded {expanded} segments to be at least {expand_seg} "
          f"frames long. {n_seg} total segments.")
    # write metadata to file
    json.dump(meta, meta_file.open("wt", encoding="utf8"), sort_keys=True)
    print(f"wrote metadata for {len(meta)} videos to {meta_file}")


if __name__ == '__main__':
    main()
