"""Find photos on disk. No Qt — used by the dataset and the file dialogs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from app.constants import IMAGE_EXTENSIONS, LABEL_EXTENSIONS


def is_image_path(path: str) -> bool:
    name = Path(path).name
    if name.startswith("."):
        return False
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_label_path(path: str) -> bool:
    return Path(path).suffix.lower() in LABEL_EXTENSIONS


def collect_images(root: Union[str, Path]) -> List[str]:
    """Image paths under ``root``, nested folders included, sorted."""
    directory = Path(root)
    if not directory.is_dir():
        return []
    paths: List[str] = []
    try:
        for entry in directory.rglob("*"):
            if entry.is_file() and is_image_path(str(entry)):
                paths.append(str(entry))
    except OSError:
        return []
    paths.sort()
    return paths
