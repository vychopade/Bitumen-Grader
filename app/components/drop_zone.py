"""Shared drag-and-drop helpers for file drop zones."""
from __future__ import annotations

from typing import Sequence, Union

from PyQt6.QtGui import QDragEnterEvent, QDropEvent

DropEvent = Union[QDragEnterEvent, QDropEvent]


def dropped_local_paths(event: DropEvent, extensions: Sequence[str]) -> list[str]:
    """Local file paths from a drag/drop event that match ``extensions``."""
    mime = event.mimeData()
    if mime is None or not mime.hasUrls():
        return []
    suffixes = tuple(extensions)
    paths: list[str] = []
    for url in mime.urls():
        if not url.isLocalFile():
            continue
        path = url.toLocalFile()
        if path.lower().endswith(suffixes):
            paths.append(path)
    return paths
