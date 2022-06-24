"""
Script to run the MLP MNIST example.


Examples:
    python -m nntrainer.examples.run_mlp_mnist -e mnist
    python -m nntrainer.examples.run_mlp_mnist -c config/mlp/default/mnist.yaml
    python -m nntrainer.examples.run_mlp_mnist -e mnist -n 3

Notes:
    Training workflow will look like this:

    1. Load configuration YAML file as a dictionary from `config/mlpmnist/default/example.yaml`
    2. Create a configuration object with class `MLPMnistConfig` from the configuration dictionary
    3. Create PyTorch datasets and dataloaders for train and val set
    4. Create the `MLPModelManager` which only holds one model in this case and defines the forward pass.
    5. Create the `MLPMnistTrainer`.
    6. `trainer.train_model()` contains the training loop over epochs, including validation at the end of epochs.
    7. `trainer.validate_epoch()` Evalues a single epoch.
"""

import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from nntrainer import arguments, utils
from nntrainer.data import create_loader
from nntrainer.examples.mlp_mnist import MLPMNISTExperimentConfig, MLPMNISTTrainer, MLPModelManager, MNISTExperimentType
from nntrainer.utils_torch import set_seed
from nntrainer.utils_yaml import load_yaml_config_file


EXP_TYPE = MNISTExperimentType


def main():
    # setup arguments
    parser = utils.ArgParser(description=__doc__)
    arguments.add_default_args(parser)
    arguments.add_exp_identifier_args(parser)
    arguments.add_trainer_args(parser)
    arguments.add_dataset_test_arg(parser)
    args = parser.parse_args()

    # load repository config yaml file to dict
    exp_group, exp_name, config_file = arguments.setup_experiment_identifier_from_args(args, EXP_TYPE)
    config = load_yaml_config_file(config_file)

    # update experiment config and dataset path given the script arguments
    config = arguments.update_config_from_args(config, args)
    dataset_path = arguments.update_path_from_args(args)

    # read experiment config dict
    cfg = MLPMNISTExperimentConfig(config)
    if args.print_config:
        print(cfg)

    # set seed
    verb = "Set seed"
    if cfg.random_seed is None:
        cfg.random_seed = np.random.randint(0, 2 ** 15, dtype=np.int32)
        verb = "Randomly generated seed"
    print(f"{verb} {cfg.random_seed} deterministic {cfg.cudnn_deterministic} "
          f"benchmark {cfg.cudnn_benchmark}")
    set_seed(cfg.random_seed, cudnn_deterministic=cfg.cudnn_deterministic, cudnn_benchmark=cfg.cudnn_benchmark)

    # create datasets
    train_set = MNIST(str(dataset_path), train=True, download=True, transform=ToTensor())
    val_set = MNIST(str(dataset_path), train=False, download=True, transform=ToTensor())

    # make datasets smaller if requested in config
    if cfg.dataset_train.max_datapoints > -1:
        train_set.data = train_set.data[:cfg.dataset_train.max_datapoints]
    if cfg.dataset_val.max_datapoints > -1:
        val_set.data = val_set.data[:cfg.dataset_val.max_datapoints]

    # create dataloaders
    train_loader = create_loader(train_set, cfg.dataset_train, batch_size=cfg.train.batch_size)
    val_loader = create_loader(val_set, cfg.dataset_val, batch_size=cfg.val.batch_size)

    if args.test_dataset:
        # run dataset test and exit
        run_mlpmnist_dataset_test(train_set, train_loader)
        return
    print("---------- Setup done!")

    for run_number in range(1, args.num_runs + 1):
        run_name = f"{args.run_name}{run_number}"

        # create model
        model_mgr = MLPModelManager(cfg)

        # always load best epoch during validation
        load_best = args.load_best or args.validate

        # create trainer
        trainer = MLPMNISTTrainer(
            cfg, model_mgr, exp_group, exp_name, run_name, len(train_loader), log_dir=args.log_dir,
            log_level=args.log_level, logger=None, print_graph=args.print_graph, reset=args.reset,
            load_best=load_best, load_epoch=args.load_epoch, inference_only=args.validate)

        if args.validate:
            # run validation
            trainer.validate_epoch(val_loader)
        else:
            # run training
            trainer.train_model(train_loader, val_loader)

        # done with this round
        trainer.close()
        del model_mgr
        del trainer


def run_mlpmnist_dataset_test(train_set: MNIST, train_loader: data.DataLoader) -> None:
    """
    Test MNIST dataset.

    Args:
        train_set: Dataset.
        train_loader: Dataloader.
    """
    print("---------- Testing dataset ----------")
    print(f"Length {len(train_set)}")
    for i, (inputs, labels) in enumerate(train_loader):
        print(f"batch number: {i}, inputs: {inputs.shape}, labels: {labels.shape}")
        break


if __name__ == "__main__":
    main()
