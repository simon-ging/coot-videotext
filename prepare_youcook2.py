import argparse
import json
from collections import OrderedDict as odict
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataroot", type=str, default="data",
        help="change datasets root path")
    parser.add_argument(
        "--howto100m", action="store_true")
    args = parser.parse_args()
    dataset_path = Path(args.dataroot) / "youcook2"
    features_file = (dataset_path / (
        "video_feat_100m.h5" if args.howto100m else "video_feat_2d3d.h5"))
    meta_file = dataset_path / "meta_{}.json".format(
        "100m" if args.howto100m else "2d3d")
    meta_in_file = (dataset_path / "captions" /
                    "youcookii_annotations_trainval.json")
    with meta_in_file.open("rt", encoding="utf8") as fh:
        meta_raw = json.load(fh)["database"]
    vid_h5 = h5py.File(features_file, "r")
    split_map = {
        "training": "train",
        "validation": "val"
    }
    meta_dict = odict()
    for idx, meta in meta_raw.items():
        duration_sec = meta["duration"]
        split = split_map[meta["subset"]]
        num_frames = int(vid_h5[idx].shape[0])
        fps = num_frames / duration_sec
        segs = []
        for seg in meta["annotations"]:
            time_start, time_stop = seg["segment"]
            assert time_stop > time_start
            start_frame = int(np.floor(fps * time_start))
            stop_frame = int(np.ceil(fps * time_stop)) + 1
            if stop_frame >= num_frames:
                stop_frame = num_frames - 1
            num_frames_seg = stop_frame - start_frame + 1
            narration = seg["sentence"]
            seg_new = {
                "narration": narration,
                "start_frame": start_frame,
                "num_frames": num_frames_seg
            }
            segs.append(seg_new)
        meta_dict[idx] = odict([
            ("split", split),
            ("data_id", idx),
            ("num_frames", num_frames),
            ("segments", segs),
        ])
    json.dump(meta_dict, meta_file.open("wt", encoding="utf8"), sort_keys=True)
    print(f"wrote {meta_file}")


if __name__ == '__main__':
    main()
