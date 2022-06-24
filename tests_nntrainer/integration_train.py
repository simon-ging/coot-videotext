"""
Integration test: Train on MNIST, save and reload from checkpoints.

- Components all initialize correctly (dataset, model, trainer).
- Training and validation runs.
- Restarts work between different setups (with/without CUDA/nn.DataParallel).
"""

from copy import deepcopy

from torch import cuda
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from nntrainer.arguments import setup_config_file_from_experiment_identifier
from nntrainer.data import create_loader
from nntrainer.examples.mlp_mnist import MLPMNISTExperimentConfig, MLPMNISTTrainer, MLPModelManager, MNISTExperimentType
from nntrainer.utils import TrainerPathConst
from nntrainer.utils_torch import set_seed
from nntrainer.utils_yaml import load_yaml_config_file


def test_save_load():
    # load repository and experiment config
    exp_group, exp_name, run_name = "default", "mnist", "run1"
    config_file = setup_config_file_from_experiment_identifier(MNISTExperimentType, exp_group, exp_name,
                                                               config_dir=TrainerPathConst.DIR_CONFIG)
    config = load_yaml_config_file(config_file)
    dataset_path = "data"
    cfg = MLPMNISTExperimentConfig(config)
    cfg.dataset_train.num_workers = 0
    cfg.dataset_val.num_workers = 0
    set_seed(0, cudnn_deterministic=True, cudnn_benchmark=False)

    if not cuda.is_available():
        # make this test safe in cpu environment
        cfg.use_multi_gpu = False
        cfg.use_cuda = False

    # create dataset and dataloader
    train_set = MNIST(str(dataset_path), train=True, download=True, transform=ToTensor())
    val_set = MNIST(str(dataset_path), train=False, download=True, transform=ToTensor())
    train_loader = create_loader(train_set, cfg.dataset_train, batch_size=cfg.train.batch_size)
    val_loader = create_loader(val_set, cfg.dataset_val, batch_size=cfg.val.batch_size)

    # create model
    model_mgr = MLPModelManager(cfg)

    # create trainer
    trainer = MLPMNISTTrainer(cfg, model_mgr, exp_group, exp_name, run_name, len(train_loader), reset=True)

    # run training for some epochs
    cfg.train.num_epochs = 2
    trainer.train_model(train_loader, val_loader)

    # reload without cuda and train an epoch on the CPU
    del trainer
    del model_mgr
    cfg.train.num_epochs = 3
    cfg.use_multi_gpu = False
    cfg.use_cuda = False

    model_mgr = MLPModelManager(cfg)
    trainer = MLPMNISTTrainer(cfg, model_mgr, exp_group, exp_name, run_name, len(train_loader))
    trainer.train_model(train_loader, val_loader)

    # one more epoch with cuda only (no multi_gpu)
    del trainer
    del model_mgr
    cfg.train.num_epochs = 4
    cfg.use_multi_gpu = False
    cfg.use_cuda = True
    if not cuda.is_available():
        # make this test safe in cpu environment
        cfg.use_multi_gpu = False
        cfg.use_cuda = False

    model_mgr = MLPModelManager(cfg)
    trainer = MLPMNISTTrainer(cfg, model_mgr, exp_group, exp_name, run_name, len(train_loader))
    trainer.train_model(train_loader, val_loader)
    old_metrics = deepcopy(trainer.metrics.storage_epoch)
    print(old_metrics)

    # test inference from loaded model
    del trainer
    del model_mgr
    cfg.train.num_epochs = 4
    cfg.use_multi_gpu = False
    cfg.use_cuda = True
    if not cuda.is_available():
        # make this test safe in cpu environment
        cfg.use_cuda = False

    model_mgr = MLPModelManager(cfg)
    trainer = MLPMNISTTrainer(cfg, model_mgr, exp_group, exp_name, run_name, len(train_loader), inference_only=True)
    loss, acc, _, _ = trainer.validate_epoch(val_loader)
    print(loss, acc)


if __name__ == "__main__":
    test_save_load()
