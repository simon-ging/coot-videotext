"""
Extract howto100m features given a directory of frames.
"""
import argparse
import os
from pathlib import Path

import h5py
import torch as th
from tqdm import tqdm
from PIL import Image


class FramesLoader:
    def __init__(self, input_path):
        print(f"Checking {input_path} for frames")
        input_files = sorted(os.listdir(input_path))
        num_frames = {}
        for frame_dir in tqdm(input_files, desc="Checking frames"):
            full_frame_dir = Path(input_path) / frame_dir
            if not full_frame_dir.is_dir():
                continue
            n_frames = 0
            for frame_file in sorted(os.listdir(full_frame_dir)):
                if not frame_file.endswith(".jpg"):
                    continue
                n_frames += 1
            num_frames[frame_dir] = n_frames
        print(f"Found {len(num_frames)} videos with {sum(num_frames.values())} frames total")

        self.input_path = input_path
        self.num_frames = num_frames

    def get_frames(self, video_id):
        # returns a stack of all frames for the video, float32, range [0, 1]
        frames = []
        for n_frame in range(self.num_frames[video_id]):
            frame_file = Path(self.input_path) / video_id / f"frame_{n_frame + 1:010d}.jpg"

            # noinspection PyTypeChecker
            decoded_arr = np.array(Image.open(str(frame_file)))
            if decoded_arr.ndim == 2:
                # grayscale image, repeat to get 3 channels
                decoded_arr = np.stack([decoded_arr] * 3, axis=-1)
            decoded_arr = decoded_arr.astype(np.float32) / 255  # shape (h, w, 3)
            tensor = th.permute(th.from_numpy(decoded_arr), (2, 0, 1))  # shape (3, h, w)
            frames.append(tensor)
        stacked_frames = th.stack(frames)  # shape (len_video, 3, h, w)
        frames = th.permute(stacked_frames, (1, 0, 2, 3)) # shape (3, len_video, h, w)
        return frames


@th.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames_path", type=str, help="Path to video frames")
    parser.add_argument("output_file", type=str, help="Path to output features")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--num_cuda", type=int, default=1)
    parser.add_argument("--kernel", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--layer", type=str, default="video_embedding",
                        help="Options 'video_embedding', 'mixed_5c', "
                             "'video_embedding,mixed_5c' or 'before_mean'")
    args = parser.parse_args()
    path = Path(args.frames_path)
    batch_size = args.batch_size
    kernel = args.kernel
    stride = args.stride

    # load model
    print("Load model...")
    net = S3D("pretrained_models/s3d_dict.npy", 512)
    net.load_state_dict(th.load("pretrained_models/s3d_howto100m.pth"))
    net = net.eval()
    if args.cuda:
        net = net.cuda()
    output_names = [args.layer]
    if "," in args.layer:
        output_names = args.layer.split(",")

    # load frames
    print(f"Assuming frames at 16 FPS and 256x256 resolution in {path}. Load frames")
    frames_loader = FramesLoader(path)
    video_keys = list(frames_loader.num_frames.keys())

    # make frame features
    vid_h5_file = Path(args.output_file)
    os.makedirs(vid_h5_file.parent, exist_ok=True)
    print(f"Appending to file {vid_h5_file}")
    vid_h5 = h5py.File(vid_h5_file, "a")
    pbar = tqdm(total=len(video_keys))
    last_shape = None
    for key in video_keys:
        data_id = key
        if data_id not in vid_h5:
            # preload all frames for this video
            num_frames = frames_loader.num_frames[data_id]
            frames = frames_loader.get_frames(data_id)

            # input frames: shape (3, len_video, 256, 256)
            def feed_batch(input_list):
                batch_input = th.stack(input_list, dim=0)
                try:
                    if args.cuda:
                        if args.num_cuda > 1:
                            res = nn.parallel.data_parallel(net, batch_input, range(args.num_cuda))
                        else:
                            res = net(batch_input.cuda())
                    else:
                        res = net(batch_input)
                except RuntimeError as e:
                    raise RuntimeError(
                            f"Input {batch_input.shape} failed! Video {key} frames {frames.shape}") from e
                # print(f"result {result.shape}")
                results_list = []
                for layer in output_names:
                    results_list.append(res[layer])
                res = th.cat(results_list, dim=-1)
                return res

            # given some number of frames, get 32 frames with stride 16
            frames_collector = []
            results_collector = []
            for pointer in range(0, num_frames, stride):
                frames_single = frames[:, pointer:pointer + kernel, :, :]
                if frames_single.shape[1] < kernel:
                    # last frames are less than kernel size.
                    if num_frames > kernel:
                        # if total video is longer than kernel, add the last frames
                        frames_single = frames[:, -kernel:, :, :]
                    else:
                        # otherwise ignore the too short video, will be handled below
                        frames_single = None
                if frames_single is not None:
                    frames_collector.append(frames_single)
                if len(frames_collector) == batch_size:
                    results_batch = feed_batch(frames_collector)
                    results_collector.append(results_batch)
                    frames_collector = []
            # feed trailing batch if exists
            if len(frames_collector) > 0:
                results_collector.append(feed_batch(frames_collector))
            if len(results_collector) == 0:
                # video was too short, less than kernel size. simply input entire video
                print(f"WARNING: Low amount of data for {data_id} only {frames.shape[1]} frames.")
                # must be multiple of 16 otherwise it will not work
                frames = frames[:, :16, :, :]
                results_collector.append(feed_batch([frames]))
            results = th.cat(results_collector, dim=0)
            last_shape = results.shape
            # shape (new_num_frames, 512)

            # write to h5
            vid_h5[data_id] = results.detach().cpu().numpy()
            del results
            del frames_collector
            del results_collector
            del frames
        pbar.set_description(refresh=False, desc=f"shape {last_shape}")
        pbar.update()

    vid_h5.close()
    pbar.close()

    print(f"File {vid_h5_file} shapes:")
    vid_h5 = h5py.File(vid_h5_file, "r")
    for key, data in vid_h5.items():
        print(key, data.shape)
        break
    print("Done!")


"""
Source: https://github.com/ArrowLuo/VideoFeatureExtractor/blob/master/videocnn/models/s3dg.py

Contains a PyTorch definition for Gated Separable 3D network (S3D-G)
with a text module for computing joint text-video embedding from raw text
and video input. The following code will enable you to load the HowTo100M
pretrained S3D Text-Video model from:
  A. Miech, J.-B. Alayrac, L. Smaira, I. Laptev, J. Sivic and A. Zisserman,
  End-to-End Learning of Visual Representations from Uncurated Instructional Videos.
  https://arxiv.org/abs/1912.06430.

S3D-G was proposed by:
  S. Xie, C. Sun, J. Huang, Z. Tu and K. Murphy,
  Rethinking Spatiotemporal Feature Learning For Video Understanding.
  https://arxiv.org/abs/1712.04851.
  Tensorflow code: https://github.com/tensorflow/models/blob/master/research/slim/nets/s3dg.py

The S3D architecture was slightly modified with a space to depth trick for TPU
optimization.
"""
# # BEGIN PRIVATE
import re

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    def __init__(
            self,
            inpu_dim,
            num_outputs_0_0a,
            num_outputs_1_0a,
            num_outputs_1_0b,
            num_outputs_2_0a,
            num_outputs_2_0b,
            num_outputs_3_0b,
            gating=True,
    ):
        super(InceptionBlock, self).__init__()
        self.conv_b0 = STConv3D(inpu_dim, num_outputs_0_0a, [1, 1, 1])
        self.conv_b1_a = STConv3D(inpu_dim, num_outputs_1_0a, [1, 1, 1])
        self.conv_b1_b = STConv3D(
                num_outputs_1_0a, num_outputs_1_0b, [3, 3, 3], padding=1, separable=True
        )
        self.conv_b2_a = STConv3D(inpu_dim, num_outputs_2_0a, [1, 1, 1])
        self.conv_b2_b = STConv3D(
                num_outputs_2_0a, num_outputs_2_0b, [3, 3, 3], padding=1, separable=True
        )
        self.maxpool_b3 = th.nn.MaxPool3d((3, 3, 3), stride=1, padding=1)
        self.conv_b3_b = STConv3D(inpu_dim, num_outputs_3_0b, [1, 1, 1])
        self.gating = gating
        self.output_dim = (
                num_outputs_0_0a + num_outputs_1_0b + num_outputs_2_0b + num_outputs_3_0b
        )
        if gating:
            self.gating_b0 = SelfGating(num_outputs_0_0a)
            self.gating_b1 = SelfGating(num_outputs_1_0b)
            self.gating_b2 = SelfGating(num_outputs_2_0b)
            self.gating_b3 = SelfGating(num_outputs_3_0b)

    def forward(self, inpu):
        """
        Inception block.
        """
        b0 = self.conv_b0(inpu)
        b1 = self.conv_b1_a(inpu)
        b1 = self.conv_b1_b(b1)
        b2 = self.conv_b2_a(inpu)
        b2 = self.conv_b2_b(b2)
        b3 = self.maxpool_b3(inpu)
        b3 = self.conv_b3_b(b3)
        if self.gating:
            b0 = self.gating_b0(b0)
            b1 = self.gating_b1(b1)
            b2 = self.gating_b2(b2)
            b3 = self.gating_b3(b3)
        return th.cat((b0, b1, b2, b3), dim=1)


class SelfGating(nn.Module):
    def __init__(self, inpu_dim):
        super(SelfGating, self).__init__()
        self.fc = nn.Linear(inpu_dim, inpu_dim)

    def forward(self, inpu_tensor):
        """
        Feature gating as used in S3D-G.
        """
        spatiotemporal_average = th.mean(inpu_tensor, dim=[2, 3, 4])
        weights = self.fc(spatiotemporal_average)
        weights = th.sigmoid(weights)
        return weights[:, :, None, None, None] * inpu_tensor


# noinspection PyUnboundLocalVariable
class STConv3D(nn.Module):
    def __init__(
            self, inpu_dim, output_dim, kernel_size, stride=1, padding=0, separable=False
    ):
        super(STConv3D, self).__init__()
        self.separable = separable
        self.relu = nn.ReLU(inplace=True)
        assert len(kernel_size) == 3
        if separable and kernel_size[0] != 1:
            spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
            temporal_kernel_size = [kernel_size[0], 1, 1]
            if isinstance(stride, list) and len(stride) == 3:
                spatial_stride = [1, stride[1], stride[2]]
                temporal_stride = [stride[0], 1, 1]
            else:
                spatial_stride = [1, stride, stride]
                temporal_stride = [stride, 1, 1]
            if isinstance(padding, list) and len(padding) == 3:
                spatial_padding = [0, padding[1], padding[2]]
                temporal_padding = [padding[0], 0, 0]
            else:
                spatial_padding = [0, padding, padding]
                temporal_padding = [padding, 0, 0]
        if separable:
            self.conv1 = nn.Conv3d(
                    inpu_dim,
                    output_dim,
                    kernel_size=spatial_kernel_size,
                    stride=spatial_stride,
                    padding=spatial_padding,
                    bias=False,
            )
            self.bn1 = nn.BatchNorm3d(output_dim)
            self.conv2 = nn.Conv3d(
                    output_dim,
                    output_dim,
                    kernel_size=temporal_kernel_size,
                    stride=temporal_stride,
                    padding=temporal_padding,
                    bias=False,
            )
            self.bn2 = nn.BatchNorm3d(output_dim)
        else:
            self.conv1 = nn.Conv3d(
                    inpu_dim,
                    output_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
            )
            self.bn1 = nn.BatchNorm3d(output_dim)

    def forward(self, inpu):
        out = self.relu(self.bn1(self.conv1(inpu)))
        if self.separable:
            out = self.relu(self.bn2(self.conv2(out)))
        return out


class MaxPool3dTFPadding(th.nn.Module):
    def __init__(self, kernel_size, stride=None, padding="SAME"):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == "SAME":
            padding_shape = self._get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = th.nn.ConstantPad3d(padding_shape, 0)
        self.pool = th.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def _get_padding_shape(self, filter_shape, stride):
        def _pad_top_bottom(filter_dim_, stride_val_):
            pad_along = max(filter_dim_ - stride_val_, 0)
            pad_top_ = pad_along // 2
            pad_bottom_ = pad_along - pad_top_
            return pad_top_, pad_bottom_

        padding_shape = []
        for filter_dim, stride_val in zip(filter_shape, stride):
            pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
            padding_shape.append(pad_top)
            padding_shape.append(pad_bottom)
        depth_top = padding_shape.pop(0)
        depth_bottom = padding_shape.pop(0)
        padding_shape.append(depth_top)
        padding_shape.append(depth_bottom)
        return tuple(padding_shape)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out


class Sentence_Embedding(nn.Module):
    def __init__(
            self,
            embd_dim,
            num_embeddings=66250,
            word_embedding_dim=300,
            token_to_word_path="dict.npy",
            max_words=16,
            output_dim=2048,
    ):
        super(Sentence_Embedding, self).__init__()
        self.word_embd = nn.Embedding(num_embeddings, word_embedding_dim)
        self.fc1 = nn.Linear(word_embedding_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, embd_dim)
        self.word_to_token = {}
        self.max_words = max_words
        token_to_word = np.load(token_to_word_path)
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [
                self.word_to_token[word] for word in words if word in self.word_to_token
        ]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we
        else:
            return th.zeros(self.max_words).long()

    def _words_to_ids(self, x):
        split_x = [self._words_to_token(self._split_text(sent.lower())) for sent in x]
        return th.stack(split_x, dim=0)

    def forward(self, x):
        x = self._words_to_ids(x)
        x = self.word_embd(x)
        x = F.relu(self.fc1(x))
        x = th.max(x, dim=1)[0]
        x = self.fc2(x)
        return {'text_embedding': x}


class S3D(nn.Module):
    def __init__(self, dict_path, num_classes=512, gating=True, space_to_depth=True):
        super(S3D, self).__init__()
        self.num_classes = num_classes
        self.gating = gating
        self.space_to_depth = space_to_depth
        if space_to_depth:
            self.conv1 = STConv3D(
                    24, 64, [2, 4, 4], stride=1, padding=(1, 2, 2), separable=False
            )
        else:
            self.conv1 = STConv3D(
                    3, 64, [3, 7, 7], stride=2, padding=(1, 3, 3), separable=False
            )
        self.conv_2b = STConv3D(64, 64, [1, 1, 1], separable=False)
        self.conv_2c = STConv3D(64, 192, [3, 3, 3], padding=1, separable=True)
        self.gating = SelfGating(192)
        self.maxpool_2a = MaxPool3dTFPadding(
                kernel_size=(1, 3, 3), stride=(1, 2, 2), padding="SAME"
        )
        self.maxpool_3a = MaxPool3dTFPadding(
                kernel_size=(1, 3, 3), stride=(1, 2, 2), padding="SAME"
        )
        self.mixed_3b = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.mixed_3c = InceptionBlock(
                self.mixed_3b.output_dim, 128, 128, 192, 32, 96, 64
        )
        self.maxpool_4a = MaxPool3dTFPadding(
                kernel_size=(3, 3, 3), stride=(2, 2, 2), padding="SAME"
        )
        self.mixed_4b = InceptionBlock(
                self.mixed_3c.output_dim, 192, 96, 208, 16, 48, 64
        )
        self.mixed_4c = InceptionBlock(
                self.mixed_4b.output_dim, 160, 112, 224, 24, 64, 64
        )
        self.mixed_4d = InceptionBlock(
                self.mixed_4c.output_dim, 128, 128, 256, 24, 64, 64
        )
        self.mixed_4e = InceptionBlock(
                self.mixed_4d.output_dim, 112, 144, 288, 32, 64, 64
        )
        self.mixed_4f = InceptionBlock(
                self.mixed_4e.output_dim, 256, 160, 320, 32, 128, 128
        )
        self.maxpool_5a = self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
                kernel_size=(2, 2, 2), stride=(2, 2, 2), padding="SAME"
        )
        self.mixed_5b = InceptionBlock(
                self.mixed_4f.output_dim, 256, 160, 320, 32, 128, 128
        )
        self.mixed_5c = InceptionBlock(
                self.mixed_5b.output_dim, 384, 192, 384, 48, 128, 128
        )
        self.fc = nn.Linear(self.mixed_5c.output_dim, num_classes)
        self.text_module = Sentence_Embedding(num_classes,
                                              token_to_word_path=dict_path)

    def _space_to_depth(self, inpu):
        """
        3D space to depth trick for TPU optimization.
        """
        B, C, T, H, W = inpu.shape
        inpu = inpu.view(B, C, T // 2, 2, H // 2, 2, W // 2, 2)
        inpu = inpu.permute(0, 3, 5, 7, 1, 2, 4, 6)
        inpu = inpu.contiguous().view(B, 8 * C, T // 2, H // 2, W // 2)
        return inpu

    def forward(self, inpus):
        """
        Defines the S3DG base architecture.
        """
        if self.space_to_depth:
            inpus = self._space_to_depth(inpus)
        net = self.conv1(inpus)
        if self.space_to_depth:
            # we need to replicate 'SAME' tensorflow padding
            net = net[:, :, 1:, 1:, 1:]
        net = self.maxpool_2a(net)
        net = self.conv_2b(net)
        net = self.conv_2c(net)
        if self.gating:
            net = self.gating(net)
        net = self.maxpool_3a(net)
        net = self.mixed_3b(net)
        net = self.mixed_3c(net)
        net = self.maxpool_4a(net)
        net = self.mixed_4b(net)
        net = self.mixed_4c(net)
        net = self.mixed_4d(net)
        net = self.mixed_4e(net)
        net = self.mixed_4f(net)
        net = self.maxpool_5a(net)
        net = self.mixed_5b(net)
        net_all = self.mixed_5c(net)
        net = th.mean(net_all, dim=[2, 3, 4])
        return {'video_embedding': self.fc(net), 'mixed_5c': net, 'before_mean': net_all}


if __name__ == "__main__":
    main()
