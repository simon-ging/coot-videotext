# COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning

This repository is the official PyTorch implementation of our [paper](https://arxiv.org/abs/2011.00597) which will be published at NeurIPS 2020.

<!-- Check our [slides](assets/slides_coot.pdf) or [poster](assets/poster_coot.pdf) for a short overview. -->

<p align="center"><img src="assets/logo.png" alt="Logo" title="Logo" /></p>

<!-- ![Logo](assets/logo.png) -->

## Model Outline

<p align="center"><img src="assets/thumbnail.png" alt="Method" title="Method" /></p>

<!-- ![Method](assets/thumbnail.png) -->

## Development Roadmap

### Current version features

- Reproduce the evaluation results on Video-Text Retrieval either with the provided models or by training them from scratch. Configurations and weights for the COOT models described in tables 2 and 3 of the paper are provided.

### Planned features

- Upload COOT feature output. See [this issue](https://github.com/gingsi/coot-videotext/issues/4) for an explanation on how to extract them yourself with this version.
- Reproduce the results on Video Captioning described in tables 4 and 5.
- Improve code to make it easier to input a custom dataset.

## Prerequisites

Requires `Python>=3.6`, `PyTorch>=1.4`. Tested on Ubuntu. At least 8GB of free RAM are needed to load the text features into memory. GPU training is recommended for speed (requires 2x11GB GPU memory).

## Installation

1. Install Python and PyTorch
1. Clone repository: `git clone https://github.com/gingsi/coot-videotext`
1. Set working directory to be inside the repository: `cd coot-videotext`
1. Install other requirements `pip install -r requirements.txt`
1. All future commands in this Readme assume that the current working directory is the root of this repository

## Prepare datasets

### ActivityNet Captions

**Download**: Please download the file via P2P using [this torrent](https://academictorrents.com/details/0c824440c94cc18ace1cb2c77423919b728d703e) and kindly keep seeding after you are done. See Troubleshoot / Downloading torrents below for help.

**Alternative google drive download:** [Download Link](https://drive.google.com/file/d/13ZnIfBRShld8KOKJV3kpOLJPFkl12V9J/view?usp=sharing) or [Mirror Link](https://drive.google.com/file/d/1Gir-cRLhVpqjyADq55r5VF9Cs9YdOOz9/view?usp=sharing)

~~~bash
# 1) download ~52GB zipped features to data/activitynet/
# 2) unzip
# after extraction, the folder structure should look like this:
# data/activitynet/features/ICEP_V3_global_pool_skip_8_direct_resize/v_XXXXXXXXXXX.npz
tar -C data/activitynet/features -xvzf data/activitynet/ICEP_V3_global_pool_skip_8_direct_resize.tar.gz
# 3) preprocess dataset and compute bert features
python prepare_activitynet.py
python run_bert.py activitynet --cuda
~~~

### Youcook2 with ImageNet/Kinetics Features

**Download**: Please download the file via P2P using [this torrent](https://academictorrents.com/details/3ae97c261ed32d3bd5326d3bf6991c9e2ea3dc17) and kindly keep seeding after you are done. See Troubleshoot / Downloading torrents below for help.

**Alternative google drive download:** [Download Link](https://drive.google.com/file/d/1q7QocJq3mDJU0VxqJRZhSbqdtPerC4PS/view?usp=sharing) 

~~~bash
# 1) download ~13GB zipped features to data/youcook2/
# 2) unzip
tar -C data/youcook2 -xzvf data/youcook2/video_feat_2d3d.tar.gz
# after extraction, the folder structure should look like this:
# data/youcook2/video_feat_2d3d.h5
# 2) preprocess dataset and compute bert features
python prepare_youcook2.py
python run_bert.py youcook2 --cuda --metadata_name 2d3d
~~~

### Youcook2 with Howto100m features

**Download**: Please download the file via P2P using [this torrent](https://academictorrents.com/details/70417e3793dbbb03ca68981307860254766d5a1d) and kindly keep seeding after you are done. See Troubleshoot / Downloading torrents below for help.

**Alternative google drive download:** [Download Link](https://drive.google.com/file/d/1oWSg7mvZE2kP_Ig4-OdNjRPAMqDghwag/view?usp=sharing)

~~~bash
# 1) download ~623MB zipped features to data/youcook2/
# 2) unzip
tar -C data/youcook2 -xzvf data/youcook2/video_feat_100m.tar.gz
# after extraction, the folder structure should look like this:
# data/youcook2/video_feat_100m.h5
# 3) preprocess dataset and compute bert features
python prepare_youcook2.py --howto100m
python run_bert.py youcook2 --cuda --metadata_name 100m
~~~

## Download provided models for evaluation

**Google drive download:** [Download Link](https://drive.google.com/file/d/1JPN8v3sz4rRvqo5CB76lrOdCh6kSDkg4/view?usp=sharing)

~~~bash
# 1) download ~100mb zipped models
# 2) unzip
tar -xzvf provided_models.tar.gz
# after extraction, the folder structure should look like this:
# provided_models/MODEL_NAME.pth
~~~

## Run

###  Script flags

~~~bash
--preload_vid  # preload video features to RAM (~110GB RAM needed for activitynet, 60GB for youcook2 resnet/resnext, 20GB for youcook2 howto100m)
--workers N    # change number of parallel dataloader workers, default: min(10, N_CPU - 1)
--cuda         # run on GPU
--single_gpu   # run on only one GPU
~~~

### Notes for training

- We use early stopping. Models are evaluated automatically and results are output during training. To evaluate a model again after training it, check the end of the script output or the logfile in path `runs/MODEL_NAME/log_DATE_TIME.log` to find the best epoch. Then run `python eval.py config/MODEL_NAME.yaml runs/MODEL_NAME/ckpt_ep##.pth`
- When training from scratch, actual results may vary due to randomness (no fixed seeds).
- Described train time assumes data preloading with `--preload_vid` and varies due to early stopping.

### Table 2: Video-paragraph retrieval results on AcitvityNet-captions dataset (val1).

~~~bash
# train from scratch
python train.py config/anet_coot.yaml --cuda --log_dir runs/anet_coot

# evaluate provided model
python eval.py config/anet_coot.yaml provided_models/anet_coot.pth --cuda --workers 10
~~~

| Model | Paragraph->Video R@1 | R@5  | R@50 | Video->Paragraph R@1 | R@5  | R@50 | Train time |
| ----- | -------------------- | ---- | ---- | -------------------- | ---- | ---- | ---------- |
| COOT  | 61.3                 | 86.7 | 98.7 | 60.6                 | 87.9 | 98.7 | ~70min     |

### Table 3: Retrieval Results on Youcook2 dataset

~~~bash
# train from scratch (row 1, model with ResNet/ResNext features)
python train.py config/yc2_2d3d_coot.yaml --cuda --log_dir runs/yc2_2d3d_coot

# evaluate provided model (row 1)
python eval.py config/yc2_2d3d_coot.yaml provided_models/yc2_2d3d_coot.pth --cuda

# train from scratch (row 2, model with HowTo100m features)
python train.py config/yc2_100m_coot.yaml --cuda --log_dir runs/yc2_100m_coot

# evaluate provided model (row 2)
python eval.py config/yc2_100m_coot.yaml provided_models/yc2_100m_coot.pth --cuda
~~~

| Model                             | Paragraph->Video R@1 | R@5  | R@10  | MR   | Sentence->Clip R@1 | R@5  | R@50 | MR   | Train time |
| --------------------------------- | -------------------- | ---- | ----- | ---- | ------------------ | ---- | ---- | ---- | ---------- |
| COOT with ResNet/ResNeXt features | 51.2                 | 79.9 | 88.20 | 1    | 6.6                | 17.3 | 25.1 | 48   | ~180min    |
| COOT with HowTo100m features      | 78.3                 | 96.2 | 97.8  | 1    | 16.9               | 40.5 | 52.5 | 9    | ~16 min    |

## Additional information

The default datasets folder is `data/`. To use a different folder, supply all python scripts with flag `--dataroot new_path` and change the commands for dataset preprocessing accordingly.

### Preprocessing steps, done automatically

- Activitynet
    - Switch start and stop timestamps when stop > start. Affects 2 videos.
    - Convert start/stop timestamps to start/stop frames by multiplying with FPS in the features and using floor/ceil operation respectively.
    - Captions: Replace all newlines/tabs/multiple spaces with a single space.
    - Cut too long captions (>512 bert tokens in the paragraph) by retaining at least 4 tokens and the [SEP] token for each sentence. Affects 1 video.
    - Expand clips to be at least 10 frames long
        - train/val_1:  2823 changed / 54926 total
        - train/val_2: 2750 changed / 54452 total
- Activitynet and Youcook2
    - Add [CLS] at the start of each paragraph and [SEP] at the end of each sentence before encoding with Bert model.

## Troubleshoot

### Downloading Torrents

If you have problems downloading our torrents, try following this tutorial:

1. Download and install the torrent client qBittorrent.
1. Download the torrent files from the links and open them with qBittorrent.
1. Options -> Advanced, check the fields "Always announce to all trackers in a tier" and "Always announce to all tiers".
1. Options -> BitTorrent, disable "Torrent Queueing"
1. Options -> Connection, disable "Use UPnp..." and everything under "Connection Limits" and set Proxy Server to "(None)"
1. Options -> Speed, make sure speed is unlimited.
1. Right click your torrent and "Force reannounce"
1. Right click your torrent and "Force resume"
1. Let it run for at least 24 hours.
1. If it still doesn't download after waiting for an hour, feel free to open an issue.
1. Once you are done, please keep seeding.

## Acknowledgements

For the full references see our [paper](https://arxiv.org/abs/2011.00597). We especially thank the creators of the following github repositories for providing helpful code:

- Zhang et al. for retrieval code and activitynet-densecaptions features: [CMHSE](https://github.com/zbwglory/CMHSE) 
- Wang et al. for their captioning model and code: [MART](https://github.com/jayleicn/recurrent-transformer) 
- Miech et al. for their [Video feature extractor](https://github.com/antoine77340/video_feature_extractor) and their [HowTo100M Model](https://github.com/antoine77340/MIL-NCE_HowTo100M)

We also thank the authors of all packages in the `requirements.txt` file.

Credit of the bird image to [Laurie Boyle](https://www.flickr.com/photos/92384235@N02/10551357354/) - Australia.

## License

Code is licensed under Apache2 (Copyright 2020 S. Ging). Dataset features are licensed under Apache2 (Copyright to the respective owners).

## Citation

If you find our work or code useful, please consider citing our paper:

~~~
@inproceedings{ging2020coot,
  title={COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning},
  author={Simon Ging and Mohammadreza Zolfaghari and Hamed Pirsiavash and Thomas Brox},
  booktitle={Conference on Neural Information Processing Systems},
  year={2020}
}
~~~

