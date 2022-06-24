"""
Centralize some of the MART evaluation toolset.
"""

from pathlib import Path
from typing import Dict, List, Union

from nntrainer.utils import TrainerPathConst


def get_reference_files(dset_name: str, annotations_dir: Union[str, Path] = TrainerPathConst.DIR_ANNOTATIONS
                        ) -> Dict[str, List[Path]]:
    """
    Given dataset name, load the ground truth annotations for captioning.

    Args:
        dset_name: Dataset name.
        annotations_dir: Folder with annotations.

    Returns:
        Dictionary with key: evaluation_mode (val, test) and value: list of annotation files.
    """
    annotations_dir = Path(annotations_dir) / dset_name
    if dset_name == "activitynet":
        return {
            "val": [annotations_dir / "captioning_val_1_para.json", annotations_dir / "captioning_val_2_para.json"],
            "test": [annotations_dir / "captioning_test_1_para.json", annotations_dir / "captioning_test_2_para.json"]}
    if dset_name == "youcook2":
        return {"val": [annotations_dir / "captioning_val_para.json"]}
    raise ValueError(f"Dataset unknown {dset_name}")
