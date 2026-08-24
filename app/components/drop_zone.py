"""Shared drag-and-drop helpers for file drop zones."""
from __future__ import annotations

from typing import Sequence, Union

from PyQt6.QtGui import QDragEnterEvent, QDropEvent

from app.utils.files import collect_from_urls, urls_have_accepted_files

DropEvent = Union[QDragEnterEvent, QDropEvent]


def dropped_local_paths(
    event: DropEvent,
    extensions: Sequence[str],
    *,
    recurse_dirs: bool = False,
) -> list[str]:
    """Local file paths from a drag/drop that match ``extensions``.

    When ``recurse_dirs`` is True, dropped folders are walked for matching files.
    """
    mime = event.mimeData()
    if mime is None or not mime.hasUrls():
        return []
    return collect_from_urls(mime.urls(), extensions=extensions, recurse_dirs=recurse_dirs)


def drop_has_accepted_files(
    event: DropEvent,
    extensions: Sequence[str],
    *,
    recurse_dirs: bool = False,
) -> bool:
    mime = event.mimeData()
    if mime is None or not mime.hasUrls():
        return False
    return urls_have_accepted_files(mime.urls(), extensions=extensions, recurse_dirs=recurse_dirs)
