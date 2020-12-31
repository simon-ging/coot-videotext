"""
Dataset utilities.
"""

from typing import Any, Callable, Dict, List, Optional

from torch.utils import data

from nntrainer import typext
from nntrainer.typext import ConfigClass


class DataSplitConst(typext.ConstantHolder):
    """
    Store dataset splits.
    """
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class BaseDatasetConfig(ConfigClass):
    """
    Base Dataset Configuration class

    Args:
        config: Configuration dictionary to be loaded, dataset part.
    """

    def __init__(self, config: Dict) -> None:
        # general dataset info
        self.name: str = config.pop("name")
        self.data_type: str = config.pop("data_type")
        self.subset: str = config.pop("subset")
        self.split: str = config.pop("split")
        self.max_datapoints: int = config.pop("max_datapoints")
        self.shuffle: bool = config.pop("shuffle")
        # general dataloader configuration
        self.pin_memory: bool = config.pop("pin_memory")
        self.num_workers: int = config.pop("num_workers")
        self.drop_last: bool = config.pop("drop_last")


def create_loader(dataset: data.Dataset, cfg: BaseDatasetConfig, batch_size: int, *,
                  collate_fn: Optional[Callable[[List[Any]], Any]] = None) -> data.DataLoader:
    """
    Create torch dataloader from torch dataset.

    Args:
        dataset: Dataset.
        cfg: Dataset configuration.
        batch_size: Batch size.
        collate_fn: Collation function to be used to stack the data into batches.

    Returns:
    """
    return data.DataLoader(
        dataset, batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last, collate_fn=collate_fn)  # type: ignore
