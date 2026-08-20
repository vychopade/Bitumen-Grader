"""Alt+key shortcuts that only apply while a page is visible.

Qt button shortcuts are window-wide, so if every page bound Alt+S all the
time they'd collide. Bind on show, unbind on hide. Don't reuse Alt+I/T/G/L
(those are the sidebar nav keys).
"""
from __future__ import annotations

from typing import Iterable, Tuple

from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import QAbstractButton

ShortcutBinding = Tuple[QAbstractButton, str]


def bind_page_shortcuts(bindings: Iterable[ShortcutBinding]) -> None:
    """Set Alt+<letter> on each (button, letter) pair."""
    for button, letter in bindings:
        button.setShortcut(QKeySequence(f"Alt+{letter}"))


def unbind_page_shortcuts(bindings: Iterable[ShortcutBinding]) -> None:
    """Clear shortcuts set by bind_page_shortcuts."""
    for button, _letter in bindings:
        button.setShortcut(QKeySequence())


def shortcut_tooltip(description: str, letter: str) -> str:
    """Tooltip text that mentions the Alt+letter shortcut."""
    return f"{description} (Alt+{letter})"
