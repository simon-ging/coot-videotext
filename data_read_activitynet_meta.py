"""
Read activitynet metadata.
"""
import json
import re
from pathlib import Path

import repo_config
from nntrainer import arguments, utils


RE_SPACELIKES = re.compile(r"\s+")


def main():
    # argparser
    parser = utils.ArgParser(description=__doc__)
    arguments.add_path_args(parser)
    args = parser.parse_args()

    # setup dataset path
    path_data = args.data_path if args.data_path is not None else repo_config.DATA_PATH
    path_dataset = Path(path_data) / "activitynet"
    captions_path = Path("annotations") / "activitynet"
    print(f"Working on dataset path {path_dataset} captions from {captions_path}")

    # setup other paths
    meta_file = path_dataset / "meta_all.json"

    meta_dict = {}
    for split in ["train", "val_1", "val_2"]:
        raw_data = json.load((captions_path / f"{split}.json").open("rt", encoding="utf8"))
        for key, val in raw_data.items():
            # load video information
            timestamps = val["timestamps"]
            sentences = val["sentences"]
            duration_sec = val["duration"]

            # build segments
            segments = []
            for num_seg in range(len(timestamps)):
                # load narration sentence and preprocess line separators
                sentence = sentences[num_seg]
                sentence = RE_SPACELIKES.sub(" ", sentence)
                # load start and stop timestamps
                start_sec = timestamps[num_seg][0]
                stop_sec = timestamps[num_seg][1]
                # switch them in case stop < start
                if stop_sec < start_sec:
                    print(f"switch: stop_sec {stop_sec} > start_sec {start_sec}")
                    temp_ms = start_sec
                    start_sec = stop_sec
                    stop_sec = temp_ms
                segments.append({"text": sentence, "start_sec": start_sec, "stop_sec": stop_sec})
            # shorten video key to 11 youtube letters for consistency
            assert key[:2] == "v_"
            short_key = key[2:]
            # multiple datapoints with different annotations point to the same video. add split to the key
            item_key = f"{short_key}_{split}"
            meta_dict[item_key] = {
                "data_key": short_key,
                "split": split,
                "segments": segments,
                "duration_sec": duration_sec
            }

    # write meta to file
    json.dump(meta_dict, meta_file.open("wt", encoding="utf8"), sort_keys=True)
    print(f"wrote {meta_file}")


if __name__ == "__main__":
    main()
