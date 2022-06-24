"""
Dataset utilities.
"""

from typing import Any, Callable, List, Optional

from torch.utils import data

from nntrainer import trainer_configs, typext


class DataSplitConst(typext.ConstantHolder):
    """
    Store dataset splits.
    """
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def create_loader(dataset: data.Dataset, cfg: trainer_configs.BaseDatasetConfig, batch_size: int, *,
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
