"""Qt file dialogs and drag-and-drop helpers for photos and label tables."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import QFileDialog, QWidget

from app.utils.media import collect_images, is_image_path, is_label_path

DropEvent = Union[QDragEnterEvent, QDropEvent]

IMAGE_DIALOG_FILTER = (
    "Images (*.jpg *.jpeg *.png *.tif *.tiff *.JPG *.JPEG *.PNG *.TIF "
    "*.TIFF);;"
    "All files (*)"
)
LABEL_DIALOG_FILTER = (
    "Label tables (*.csv *.txt *.xlsx *.xls);;"
    "CSV (*.csv);;Excel (*.xlsx *.xls);;Text (*.txt);;All files (*)"
)


def collect_from_urls(
    urls, *, extensions: Sequence[str], recurse_dirs: bool
) -> List[str]:
    """Turns a drop's URLs into local file paths. Pass the mime URLs, the suffixes you accept, and whether dropped folders should be walked. You get a list of matching paths."""
    suffixes = tuple(ext.lower() for ext in extensions)
    paths: List[str] = []
    seen: set[str] = set()
    for url in urls:
        if not url.isLocalFile():
            continue
        local = url.toLocalFile()
        candidate = Path(local)
        if recurse_dirs and candidate.is_dir():
            for image_path in collect_images(candidate):
                if image_path not in seen:
                    paths.append(image_path)
                    seen.add(image_path)
            continue
        if (
            candidate.is_file()
            and local.lower().endswith(suffixes)
            and local not in seen
        ):
            if candidate.name.startswith("."):
                continue
            paths.append(local)
            seen.add(local)
    return paths


def urls_have_accepted_files(
    urls, *, extensions: Sequence[str], recurse_dirs: bool
) -> bool:
    """True if the drop has at least one matching file, or a folder when recurse_dirs is on. Same arguments as collect_from_urls, but you get a yes or no instead of the paths."""
    suffixes = tuple(ext.lower() for ext in extensions)
    for url in urls:
        if not url.isLocalFile():
            continue
        local = url.toLocalFile()
        candidate = Path(local)
        if recurse_dirs and candidate.is_dir():
            return True
        if candidate.is_file() and local.lower().endswith(suffixes):
            return True
    return False


def pick_image_files(parent: Optional[QWidget] = None) -> List[str]:
    paths, _ = QFileDialog.getOpenFileNames(
        parent, "Select photos", "", IMAGE_DIALOG_FILTER
    )
    return [path for path in paths if is_image_path(path)]


def pick_image_folder(parent: Optional[QWidget] = None) -> Optional[str]:
    folder = QFileDialog.getExistingDirectory(
        parent,
        "Select photo folder",
        "",
        QFileDialog.Option.ShowDirsOnly,
    )
    return folder or None


def pick_labels_file(parent: Optional[QWidget] = None) -> Optional[str]:
    path, _ = QFileDialog.getOpenFileName(
        parent, "Select label table", "", LABEL_DIALOG_FILTER
    )
    if path and is_label_path(path):
        return path
    return None


def unique_paths(
    existing: Iterable[str], incoming: Iterable[str]
) -> List[str]:
    known = set(existing)
    added: List[str] = []
    for path in incoming:
        if path in known:
            continue
        known.add(path)
        added.append(path)
    return added


def dropped_local_paths(
    event: DropEvent,
    extensions: Sequence[str],
    *,
    recurse_dirs: bool = False,
) -> List[str]:
    """Pulls matching local files out of a drop event. Pass the event, the extensions you want, and set recurse_dirs if folders should be searched. You get a list of paths."""
    mime = event.mimeData()
    if mime is None or not mime.hasUrls():
        return []
    return collect_from_urls(
        mime.urls(), extensions=extensions, recurse_dirs=recurse_dirs
    )


def drop_has_accepted_files(
    event: DropEvent,
    extensions: Sequence[str],
    *,
    recurse_dirs: bool = False,
) -> bool:
    mime = event.mimeData()
    if mime is None or not mime.hasUrls():
        return False
    return urls_have_accepted_files(
        mime.urls(), extensions=extensions, recurse_dirs=recurse_dirs
    )
