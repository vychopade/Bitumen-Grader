"""
Page-scoped Alt+key keyboard shortcut helper.

All pages live simultaneously inside a single QStackedWidget (only one is
visible at a time), but QAbstractButton.setShortcut() uses Qt's
WindowShortcut context, which stays active for the whole top-level window
regardless of whether the owning page is currently the visible one. Binding
every button's shortcut permanently would therefore risk cross-page Alt+key
collisions (e.g. two different pages both wanting "Alt+S").

``bind_page_shortcuts``/``unbind_page_shortcuts`` let each page register its
Alt+key bindings only while it is the visible page (call from
``showEvent``/``hideEvent``), so every page is free to choose whichever
letters make sense for its own buttons -- as long as they don't collide with
the sidebar's permanently-active navigation shortcuts (Alt+I/T/G/L).
"""
from __future__ import annotations

from typing import Iterable, Tuple

from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import QAbstractButton

ShortcutBinding = Tuple[QAbstractButton, str]


def bind_page_shortcuts(bindings: Iterable[ShortcutBinding]) -> None:
    """Assign an Alt+<letter> shortcut to each ``(button, letter)`` pair."""
    for button, letter in bindings:
        button.setShortcut(QKeySequence(f"Alt+{letter}"))


def unbind_page_shortcuts(bindings: Iterable[ShortcutBinding]) -> None:
    """Clear the shortcuts previously assigned by ``bind_page_shortcuts``."""
    for button, _letter in bindings:
        button.setShortcut(QKeySequence())


def shortcut_tooltip(description: str, letter: str) -> str:
    """Build a tooltip string documenting a button's Alt+<letter> shortcut."""
    return f"{description} (Alt+{letter})"
