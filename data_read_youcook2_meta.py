"""
Read youcook2 metadata.
"""
import json
from pathlib import Path

import repo_config
from nntrainer import arguments, utils


# map original youcook2 splits to our syntax
SPLIT_MAP = {
    "training": "train",
    "validation": "val"
}

# FIXES = {"tomoatoes": "tomatoes"}
FIXES = {}


def main():
    # argparser
    parser = utils.ArgParser(description=__doc__)
    arguments.add_path_args(parser)
    args = parser.parse_args()

    # setup dataset path
    path_data = args.data_path if args.data_path is not None else repo_config.DATA_PATH
    path_dataset = Path(path_data) / "youcook2"
    captions_path = Path("annotations") / "youcook2"
    print(f"Working on dataset path {path_dataset} captions from {captions_path}")

    # setup other paths
    meta_file = path_dataset / "meta_all.json"

    # load input meta
    meta_in_file = (captions_path / "youcookii_annotations_trainval.json")
    with meta_in_file.open("rt", encoding="utf8") as fh:
        meta_raw = json.load(fh)["database"]

    # loop all videos in the dataset
    meta_dict = {}
    for key, meta in meta_raw.items():
        # load relevant meta fields
        duration_sec = meta["duration"]
        split = SPLIT_MAP[meta["subset"]]

        # loop video segments
        segs = []
        for seg in meta["annotations"]:
            # read segment fields
            time_start, time_stop = seg["segment"]
            assert time_stop > time_start, f"Negative duration"
            narration = seg["sentence"]
            for fix_from, fix_to in FIXES.items():
                narration = narration.replace(fix_from, fix_to)

            # create segment meta
            seg_new = {"text": narration, "start_sec": float(time_start), "stop_sec": float(time_stop)}
            segs.append(seg_new)

        # create video meta
        meta_dict[key] = {"data_key": key, "duration_sec": duration_sec, "split": split, "segments": segs}

    # write meta to file
    json.dump(meta_dict, meta_file.open("wt", encoding="utf8"), sort_keys=True)
    print(f"wrote {meta_file}")


if __name__ == "__main__":
    main()
