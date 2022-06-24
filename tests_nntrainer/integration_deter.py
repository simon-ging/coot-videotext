"""
Integration test: Check if training can be set to deterministic.
"""

from copy import deepcopy

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch import cuda
from nntrainer.arguments import setup_config_file_from_experiment_identifier
from nntrainer.data import create_loader
from nntrainer.examples.mlp_mnist import MLPMNISTExperimentConfig, MLPMNISTTrainer, MLPModelManager, MNISTExperimentType
from nntrainer.utils import TrainerPathConst
from nntrainer.utils_torch import set_seed
from nntrainer.utils_yaml import load_yaml_config_file


def test_deterministic():
    # load repository and experiment config
    exp_group, exp_name, run_name = "default", "test_determ", "test1"
    config_file = setup_config_file_from_experiment_identifier(MNISTExperimentType, exp_group, exp_name,
                                                               config_dir=TrainerPathConst.DIR_CONFIG)
    config = load_yaml_config_file(config_file)
    dataset_path = "data"
    cfg = MLPMNISTExperimentConfig(config)
    cfg.dataset_train.num_workers = 0
    cfg.dataset_val.num_workers = 0

    if not cuda.is_available():
        # make this test safe in cpu environment
        cfg.use_multi_gpu = False
        cfg.use_cuda = False

    def setup_test():
        # reset the seed
        set_seed(0)
        # create dataset
        _train_set = MNIST(str(dataset_path), train=True, download=True, transform=ToTensor())
        _val_set = MNIST(str(dataset_path), train=False, download=True, transform=ToTensor())
        # make datasets smaller if requested in config
        if cfg.dataset_train.max_datapoints > -1:
            _train_set.data = _train_set.data[:cfg.dataset_train.max_datapoints]
        if cfg.dataset_val.max_datapoints > -1:
            _val_set.data = _val_set.data[:cfg.dataset_val.max_datapoints]
        # create dataloader
        _train_loader = create_loader(_train_set, cfg.dataset_train, batch_size=cfg.train.batch_size)
        _val_loader = create_loader(_val_set, cfg.dataset_val, batch_size=cfg.val.batch_size)

        # create model and trainer
        _model_mgr = MLPModelManager(cfg)
        _trainer = MLPMNISTTrainer(cfg, _model_mgr, exp_group, exp_name, run_name, len(_train_loader), reset=True)
        return _trainer, _model_mgr, _train_loader, _val_loader

    # run training for some epochs
    trainer, model_mgr, train_loader, val_loader = setup_test()
    trainer.train_model(train_loader, val_loader)
    results = deepcopy(trainer.validate_epoch(val_loader))

    # reload and train again
    del trainer, model_mgr
    trainer, model_mgr, train_loader, val_loader = setup_test()
    trainer.train_model(train_loader, val_loader)
    results_again = deepcopy(trainer.validate_epoch(val_loader))

    # results must be exactly the same
    assert results == results_again, f"Run1 {results} Run2 {results_again}"


if __name__ == "__main__":
    test_deterministic()
