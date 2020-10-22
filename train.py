import argparse
from multiprocessing import cpu_count
from pathlib import Path

import utils
from dataset import create_datasets, create_loaders
from trainer import TrainerVideoText


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'config', type=str, help='Experiment to run')
    parser.add_argument(
        '--workers', type=int, default=None,
        help='set number of workers (default #CPUs - 1)')
    parser.add_argument(
        '--log_dir', type=str, default="runs/run1",
        help='directory to save/load the runs and logs')
    parser.add_argument(
        "--dataroot", type=str, default="data",
        help="change datasets root path")
    parser.add_argument(
        "--cuda", action="store_true", help="train on GPUs")
    parser.add_argument(
        "--single_gpu", action="store_true", help="Disable multi-GPU")
    parser.add_argument(
        "--preload_vid", action="store_true",
        help="Load video features into RAM")
    parser.add_argument(
        "--no_preload_text", action="store_true",
        help="Do not load text features into RAM")
    args = parser.parse_args()

    cfg = utils.load_config(args.config)
    if cfg.training.random_seed is not None:
        print('Random seed: {:d}'.format(cfg.training.random_seed))
        utils.set_seed(cfg.training.random_seed)
    num_workers = min(
        10, cpu_count() - 1) if args.workers is None else args.workers
    print(f"{num_workers} parallel dataloader workers")

    dataset_path = Path(args.dataroot) / cfg.dataset.name
    train_set, val_set = create_datasets(
        dataset_path, cfg, args.preload_vid, not args.no_preload_text)
    train_loader, val_loader = create_loaders(
        train_set, val_set, cfg.training.batch_size, num_workers)

    trainer = TrainerVideoText(
        args.log_dir, cfg, args.cuda, args.cuda and not args.single_gpu)
    trainer.train_loop(train_loader, val_loader)
    trainer.logger.info("---------- Results ----------")
    utils.print_csv_results(
        trainer.log_dir / "train_metrics.csv", cfg,
        print_fn=trainer.logger.info)
    trainer.close()


if __name__ == '__main__':
    main()
