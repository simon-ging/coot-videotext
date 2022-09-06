"""
Extract and crop frames from videos using ffmpeg
"""
# # BEGIN PRIVATE

import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import time
import traceback
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Sequence, Tuple, Union

import ffmpeg
import numpy as np
import tqdm

FRAME_FILE = "frame_%010d.jpg"
FILETYPES = ["mp4", "mkv", "webm"]
FFPROBE_INFO_FILE = "ffprobe_videos.json"
FFPROBE_ANALYSIS_FILE = "ffprobe_results.txt"
TQDM_WID = 90


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, help="input path to videos",
                        default=None, required=True)
    parser.add_argument("-l", "--input_list", type=str,
                        help="list of video file names to process less",
                        default=None)
    parser.add_argument("output_path", type=str, help="output path")
    parser.add_argument("--write", action="store_true", help="Start the crop.")
    parser.add_argument("--fps", type=float, default=16, help="Frames per second.")
    parser.add_argument(
            "--reload", action="store_true",
            help="reload all video information from the files using ffmpeg")
    parser.add_argument(
            "-x", "--width", type=int, default=256,
            help="target width of extracted frames")
    parser.add_argument(
            "-y", "--height", type=int, default=256,
            help="target height of extracted frames")
    parser.add_argument(
            "-q", "--quality", type=int, default=2,
            help="frame jpeg quality (2=best, 31=worst)")
    parser.add_argument(
            "--num_workers", type=int, default=0,
            help="how many processes to spawn, 0 means use cpu_count, ")
    parser.add_argument(
            "--max_videos", type=int, default=-1,
            help="how many videos to process in one run, default -1 = all")
    parser.add_argument(
            "--disable_progressbar", action="store_true",
            help="multiprocessing without progressbar")
    parser.add_argument(
            "--verbose", action="store_true", help="more output")
    args = parser.parse_args()

    # Get list of videos
    input_path = Path(args.input_path)
    if not args.input_list:
        files = os.listdir(input_path)
    else:
        # if a list of files is given make sure all files are found
        all_files = os.listdir(input_path)
        files = [a for a in
                 map(str.strip, Path(args.input_list).read_text("utf8").splitlines(keepends=False))
                 if
                 a != ""]
        fail = 0
        for file in files:
            if file not in all_files:
                # try adding "v_" to match
                file = f"v_{file}"
                if file not in all_files:
                    print(f"WARN: {file} not found in {all_files[:min(10, len(all_files))]}")
                    fail += 1
        if fail:
            raise ValueError(f"Missing {fail} videos out of {len(files)}")
    file_keys = []
    file_formats = []

    # if max_videos is specified (e.g. for testing) only keep this amount of files
    if args.max_videos > 0:
        files = files[:args.max_videos]
        print(f"Only process first {args.max_videos} videos.")

    # Loop videos and determine file format
    for file in files:
        file = str(file)
        file_split = file.split(".")
        file_name = ".".join(file_split[:-1])
        file_type = file_split[-1]
        if (input_path / file).is_dir():
            print(f"SKIP: directory {file}")
            continue
        if file_type not in FILETYPES:
            print(f"SKIP: don't understand filetype of {file}")
            continue
        key = file_name
        file_keys.append(key)
        file_formats.append(file_type)

    # Make sure there is only one video per ID
    if len(list(set(file_keys))) != len(file_keys):
        # now there are multiple formats of the same video, keep only the first one
        new_file_keys, new_file_formats = [], []
        for key, formt in zip(file_keys, file_formats):
            if key in new_file_keys:
                continue
            new_file_keys.append(key)
            new_file_formats.append(formt)
        file_keys, file_formats = new_file_keys, new_file_formats
        print(f"Reduced to {len(file_keys)} videos.")

    # Create output path
    output_path = Path(args.output_path)
    os.makedirs(output_path, exist_ok=True)

    # Load video ffprobe info
    num_tasks = len(file_keys)
    print("reading ffprobe info of {} videos...".format(num_tasks))
    ffprobe_file = output_path / FFPROBE_INFO_FILE
    if not ffprobe_file.exists() or args.reload:
        # # single process for debugging
        # ffprobe_infos = {}
        # for file_key, file_format in zip(file_keys, file_formats):
        #     file_video = input_path / f"{file_key}.{file_format}"
        #     try:
        #         vid_id, ffprobe_info = ffmpeg.probe(str(file_video))
        #     except ffmpeg.Error as e:
        #         print("OUT", e.stdout)
        #         print("ERR", e.stderr)
        #         raise e
        #     ffprobe_infos[vid_id] = ffprobe_info
        #     print(ffprobe_info)
        #     # num_tasks -= 1

        # setup multiprocessing
        mp = MultiProcessor(num_workers=args.num_workers, progressbar=not args.disable_progressbar)

        # enqueue tasks and run them
        num_tasks = len(file_keys)
        for file_key, file_format in zip(file_keys, file_formats):
            file_video = input_path / f"{file_key}.{file_format}"
            mp.add_task(TaskFfprobe(file_key, file_video))
        results = mp.run()
        mp.close()

        # read results
        ffprobe_infos = OrderedDict()
        for r in results:
            vid_id, ffprobe_info = r
            ffprobe_infos[vid_id] = ffprobe_info
            num_tasks -= 1

        # store to file
        with ffprobe_file.open("wt", encoding="utf8") as fh:
            json.dump(ffprobe_infos, fh, indent=4, sort_keys=True)
        print("wrote ffprobe info to: {:s}".format(str(ffprobe_file)))
    else:
        # reload from file
        with ffprobe_file.open("rt", encoding="utf8") as fh:
            ffprobe_infos = json.load(fh)
            print(f"Reloaded {len(ffprobe_infos)} videos from ffprobe results")
    print(f"{len(ffprobe_infos)} videos in ffprobe infos. {len(file_keys)} files to process.")
    assert all(file in ffprobe_infos for file in file_keys), (
            f"{ffprobe_infos.keys()}, {file_keys}. FFPROBE info seems incorrect, try reloading with --reload")

    # analyze ffprobe info
    format_list, ratio_list, fps_list, duration_list = [], [], [], []
    for vid_id, ffprobe_info in ffprobe_infos.items():
        width, height, fps, duration = get_video_info_from_ffprobe_result(ffprobe_info)
        format_list.append((width, height))
        ratio_list.append(width / height)
        fps_list.append(fps)
        duration_list.append(duration)

    # print ffprobe info
    duration_list = np.array(duration_list)
    print(f"Durations (sec): min {np.min(duration_list):.3f}, max {np.max(duration_list):.3f}, "
          f"avg {duration_list.mean():.3f}, std {duration_list.std():.3f}")
    ffprobe_analysis_file = (output_path / FFPROBE_ANALYSIS_FILE)
    with ffprobe_analysis_file.open("wt", encoding="utf8") as fh:
        print()
        format_list, ratio_list, fps_list = (
                sorted(list(set(a))) for a in (format_list, ratio_list, fps_list))
        ratio_list = [float("{:.3f}".format(a)) for a in ratio_list]
        fps_list = [float("{:.3f}".format(float(Fraction(a)))) for a in
                    fps_list]
        for file_h in [fh]:
            print("formats: {}\nratios (w/h): {}\nframerates: {}".format(
                    format_list, ratio_list, fps_list), file=file_h)
        formats_x, formats_y = zip(*format_list)
        for item, name in zip([formats_x, formats_y, ratio_list, fps_list],
                              "formats_x, formats_y, ratio_list, fps_list".split(", ")):
            hist, bin_edges = np.histogram(item, bins=30)
            bin_means = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
            print("-" * 20, name)
            print_table = list((f"{m:5.0f}", f"{h:5.0f}") for h, m in zip(hist, bin_means))
            print(" ".join(t[0] for t in print_table))
            print(" ".join(t[1] for t in print_table))

    # see how many videos still need converting to frames
    done_file = output_path / "done.txt"
    done_keys = []
    file_keys_process = file_keys
    file_formats_process = file_formats
    if done_file.is_file():
        done_keys = done_file.read_text().splitlines()
        file_keys_process = []
        file_formats_process = []
        for file_key, file_format in zip(file_keys, file_formats):
            if file_key in done_keys:
                continue
            file_keys_process.append(file_key)
            file_formats_process.append(file_format)

    print(f"{len(done_keys)} already done, {len(file_keys_process)} left.")

    # for test only, exit here
    if not args.write:
        return

    # start multiprocessing
    mp = MultiProcessor(num_workers=args.num_workers,
                        progressbar=not args.disable_progressbar)

    # enqueue tasks and run
    num_tasks = len(file_keys_process)
    for file_key, file_format in zip(file_keys_process, file_formats_process):
        # determine video file path and frame folder

        file_video_full = input_path / f"{file_key}.{file_format}"
        path_frames_full = output_path / file_key

        # read ffprobe info
        ffprobe_info = ffprobe_infos[file_key]

        # create task and enqueue it
        task = TaskExtractFrames(
                file_key, str(file_video_full), path_frames_full,
                ffprobe_info, args.width, args.height, args.fps, args.quality,
                verbose=args.verbose)
        mp.add_task(task)
    results = mp.run()
    # mp.close()

    # read results
    done_fh = done_file.open("wt")
    done_fh.write("\n".join(done_keys))
    print("analyzing results")
    for result in results:
        if result is None:
            # do not update data for failed videos
            num_tasks -= 1
            continue
        else:
            num_tasks -= 1
            vid_id, retcode, w, h, fps, num_frames = result
            done_fh.write(f"{vid_id}\n")

    # systemcall("stty sane")
    os.system("stty sane")


class TaskFfprobe(object):
    """
    Video analysis with ffprobe
    """

    def __init__(self, vid_id, file_video):
        self.vid_id = vid_id
        self.file_video = file_video

    def __call__(self):
        probe_info = get_video_ffprobe_info(self.file_video)
        # print(f"Duration is {probe_info['duration']}")
        return self.vid_id, probe_info

    def __str__(self):
        return "ffmpeg.probe on video {:s}".format(self.vid_id)


class TaskExtractFrames(object):
    """
    Frame extraction with ffmpeg
    """

    def __init__(self, vid_id, file_video, folder_frames, ffprobe_info, tw, th,
                 fps, quality, verbose=False):
        self.vid_id = vid_id
        self.file_video = file_video
        self.folder_frames = folder_frames
        self.ffprobe_info = ffprobe_info
        self.target_w = tw
        self.target_h = th
        self.target_fps = fps
        self.quality = quality
        self.verbose = verbose

    def __call__(self):
        # get width and height from ffprobe info
        w, h, fps, duration = get_video_info_from_ffprobe_result(self.ffprobe_info)
        target_w, target_h = self.target_w, self.target_h

        # prepare empty frame directory
        shutil.rmtree(str(self.folder_frames), ignore_errors=True)
        os.makedirs(str(self.folder_frames))

        # get scaled crop
        crop_y, crop_x, crop_h, crop_w = get_scaled_crop(h, w, target_h, target_w)
        ffmpeg_filter = "crop={:d}:{:d}:{:d}:{:d},scale={:d}:{:d}".format(
                crop_w, crop_h, crop_x, crop_y, target_w, target_h)
        # print(f"Scaled crop filter: {ffmpeg_filter}")

        target_fps = self.target_fps

        # define the ffmpeg command with filters
        file_frames = str(self.folder_frames / FRAME_FILE)
        cmd = "ffmpeg -i {:s} -hide_banner -vf \"{:s},fps={:f}\" -qscale:v {:d} {:s} ".format(
                self.file_video, ffmpeg_filter, target_fps, self.quality, file_frames)

        if self.verbose:
            print("command:", cmd)

        # run command
        out, err, retcode = systemcall(cmd)
        if retcode != 0:
            print()
            print("WARNING: video {} failed with return code {}".format(
                    self.vid_id, retcode))
            print("command was: {}".format(cmd))
            print("stdout:", out)
            print("stderr:", err)
            raise RuntimeError(
                    "video processing for {} failed, see stdout".format(
                            self.vid_id))

        # check how many frames where created
        num_frames = len(os.listdir(str(self.folder_frames)))

        return self.vid_id, retcode, w, h, fps, num_frames


# ---------- Image and video utility code ----------


def get_video_ffprobe_info(file_video: Union[str, Path]):
    """
    Return dictionary with info about the video given the input file
    Args:
        file_video:

    Returns:

    """
    # regular probe
    probe_info = ffmpeg.probe(str(file_video))
    # additional duration probe, otherwise duration is missing for some videos
    duration_call = (
            "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 -i "
            f"{file_video}")
    out, err, retcode = systemcall(duration_call)
    if retcode != 0:
        raise RuntimeError(
                f"Call {duration_call} failed with OUT={out} ERR={err} RETCODE={retcode}")
    duration = float(out)
    probe_info["duration"] = duration
    return probe_info


def get_video_info_from_ffprobe_result(ffprobe_info):
    """

    Args:
        ffprobe_info:   dict with ffprobe _results

    Returns:
        width, height, framerate as fraction string (e.g. 30/1)

    """
    video_stream, audio_stream = get_ffprobe_streams(ffprobe_info)
    height, width = video_stream["height"], video_stream["width"]
    fps = video_stream["r_frame_rate"]
    duration = ffprobe_info[
        "duration"]  # this was set by running an additional ffprobe system command
    return width, height, fps, duration


def get_ffprobe_streams(info):
    """

    Args:
        info: info dictionary returned from ffprobe

    Returns: video_stream, audio_stream

    """
    streams = info['streams']
    assert len(streams) == 2, streams
    video_stream, audio_stream = None, None
    for s in streams:
        if s["codec_type"] == "video":
            video_stream = s
        elif s["codec_type"] == "audio":
            audio_stream = s
        else:
            raise ValueError("unknown stream: {}".format(s))
    assert video_stream["codec_type"] == "video", streams
    assert audio_stream["codec_type"] == "audio", streams
    if video_stream is None:
        raise ValueError("video stream not found")
    if audio_stream is None:
        raise ValueError("audio stream not found")
    return video_stream, audio_stream

def floor(num):
    return int(np.floor(num).astype(int))


def rnd(num):
    return int(np.round(num).astype(int))

def get_scaled_crop(h: int, w: int, target_h: int, target_w: int) -> Tuple[int, int, int, int]:
    """
    Calculate the position and size of a cropping rectangle, such that the input image will
    be cropped to the same aspect ratio as the target rectangle.

    After applying this crop, the output image can be scaled directly to the target rectangle
    without distortion (i.e. without changing its aspect ratio).

    Args:
        h: Input height
        w: Input width
        target_h: Target height
        target_w: Target width

    Returns:
        Crop position y, x, crop height, crop width
    """
    ratio_in = w / h
    ratio_out = target_w / target_h
    if ratio_in < ratio_out:
        # video too narrow
        crop_w = w
        crop_h = rnd(w / ratio_out)
    elif ratio_in > ratio_out:
        # video too wide
        crop_w = rnd(h * ratio_out)
        crop_h = h
    else:
        # video has correct ratio
        crop_w = w
        crop_h = h

    crop_x = floor((w - crop_w) / 2)
    crop_y = floor((h - crop_h) / 2)

    return crop_y, crop_x, crop_h, crop_w


# ---------- Multiprocessing utility code ----------


def systemcall(call: Union[str, Sequence[str]]):
    pipe = subprocess.PIPE
    process = subprocess.Popen(call, stdout=pipe, stderr=pipe, shell=True)
    out, err = process.communicate()
    retcode = process.poll()
    charset = 'utf-8'
    out = out.decode(charset)
    err = err.decode(charset)
    return out, err, retcode


class Worker(multiprocessing.Process):
    def __init__(self, task_q, result_q, error_q, verbose=False):
        super().__init__()
        self.task_q = task_q
        self.result_q = result_q
        self.error_q = error_q
        self.verbose = verbose

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_q.get()
            if next_task is None:
                # poison pill means shutdown
                if self.verbose:
                    print("{:s}: exiting".format(proc_name))
                self.task_q.task_done()
                break
            if self.verbose:
                print(str(next_task))
            try:
                result = next_task()
                pass
            except Exception as e:
                self.error_q.put((e, traceback.format_exc()))
                result = None
            self.task_q.task_done()
            self.result_q.put(result)


class MultiProcessor(object):
    """
    Convenience class for multiprocessing jobs.
    """

    def __init__(self, num_workers=0, verbose=True, progressbar=True):
        self._num_workers = num_workers
        self.verbose = verbose
        self.progressbar = progressbar
        if self._num_workers == 0:
            self._num_workers = multiprocessing.cpu_count()
        self._tasks = multiprocessing.JoinableQueue()
        self._results = multiprocessing.Queue()
        self._errors = multiprocessing.Queue()
        self._workers: List[Worker] = []
        self._num_tasks = 0
        self.total_time = 0

    def add_task(self, task):
        self._tasks.put(task)
        self._num_tasks += 1

    def close(self):
        self._results.close()
        self._errors.close()
        for w in self._workers:
            w.terminate()

    def run(self, read_results=True):
        # start N _workers
        start = timer()
        if self.verbose:
            print('Creating {:d} workers'.format(self._num_workers))
        self._workers = [Worker(self._tasks, self._results, self._errors)
                         for _ in range(self._num_workers)]
        for w in self._workers:
            w.start()

        # add poison pills for _workers
        for i in range(self._num_workers):
            self._tasks.put(None)

        # write start message
        if self.verbose:
            print("Running {:d} enqueued tasks and {:d} stop signals".format(
                    self._num_tasks, self._num_workers))

        # check info on the queue, with a nice (somewhat stable) progressbar
        if self.progressbar:
            if self.verbose:
                print("waiting for the task queue to be filled...")
            num_wait = 0
            while self._tasks.empty():
                time.sleep(1)
                num_wait += 1
                if num_wait >= 5:
                    break
            tasks_now = self._num_tasks + self._num_workers
            pbar = tqdm.tqdm(total=tasks_now, ncols=TQDM_WID)
            while not self._tasks.empty():
                time.sleep(1)
                tasks_before = tasks_now
                tasks_now = self._tasks.qsize()
                resolved = tasks_before - tasks_now
                pbar.set_description(
                        "~{:7d} tasks remaining...".format(tasks_now))
                pbar.update(resolved)
            pbar.close()

        # join _tasks
        if self.verbose:
            print("waiting for all tasks to finish...")
        self._tasks.join()

        # check _errors
        if self.verbose:
            print("reading error queue... ")
        num_err = 0
        while not self._errors.empty():
            e, stacktrace = self._errors.get()
            num_err += 1
            print()
            print(stacktrace)
        if num_err >= 0:
            print("{} errors, check the log.".format(num_err))
        elif self.verbose:
            print("no errors found.")

        if not read_results:
            return self._results, self._num_tasks

        # read _results and return them
        if self.verbose:
            print("reading results...")
        results = []
        # # this can lead to some results missing
        # while not self._results.empty():
        while self._num_tasks > 0:
            result = self._results.get()
            results.append(result)
            self._num_tasks -= 1
        stop = timer()
        self.total_time = stop - start
        if self.verbose:
            print("Operation took {:.3f}s".format(self.total_time))
        return results


if __name__ == '__main__':
    main()
