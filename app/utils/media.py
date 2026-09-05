"""Walks a folder for photos. No Qt, so the dataset can use this without pulling in the UI."""

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
    """Lists every image under the given folder, including nested ones, sorted. Pass a directory path. You get a list of file paths, or an empty list if it is not a folder."""
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
