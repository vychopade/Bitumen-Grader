"""Dialogs built to match the app's dark surface instead of the stock Qt look."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QWidget,
)

from app.theme import (
    ACCENT_COLOR,
    DANGER_COLOR,
    SURFACE_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    accent_button_qss,
    danger_outline_button_qss,
    ghost_button_qss,
)

DIALOG_MIN_WIDTH = 380

# macOS hides message-box titles, so the headline text has to read on its own.
# Colour carries the tone instead of an OS icon.
_TONE_COLORS = {
    "error": DANGER_COLOR,
    "warning": ACCENT_COLOR,
    "info": TEXT_PRIMARY,
}


def _dialog_qss(headline_color: str = TEXT_PRIMARY) -> str:
    # QMessageBox names its two text labels internally. If a future Qt renames
    # them, the plain QLabel rule still gives sensible colours.
    return f"""
QMessageBox {{ background-color: {SURFACE_COLOR}; }}
QMessageBox QLabel {{ color: {TEXT_PRIMARY}; background: transparent; }}
QMessageBox QLabel#qt_msgbox_label {{
    color: {headline_color}; font-size: 13px;
}}
QMessageBox QLabel#qt_msgbox_informativelabel {{
    color: {TEXT_SECONDARY}; font-size: 12px;
}}
"""


def _widen(box: QMessageBox) -> None:
    """QMessageBox ignores setMinimumWidth, so pad its grid instead."""
    layout = box.layout()
    if isinstance(layout, QGridLayout):
        layout.addItem(
            QSpacerItem(
                DIALOG_MIN_WIDTH,
                0,
                QSizePolicy.Policy.Minimum,
                QSizePolicy.Policy.Expanding,
            ),
            layout.rowCount(),
            0,
            1,
            layout.columnCount(),
        )


def _dress(button: QPushButton, qss: str) -> None:
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setStyleSheet(qss)
    button.setMinimumWidth(96)


def confirm_destructive(
    parent: Optional[QWidget],
    *,
    title: str,
    message: str,
    detail: str = "",
    confirm_text: str = "Delete",
    cancel_text: str = "Cancel",
) -> bool:
    """Ask before something irreversible. Cancel is the default so Enter or Escape backs out."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.NoIcon)
    box.setWindowTitle(title)
    box.setText(message)
    if detail:
        box.setInformativeText(detail)
    box.setStyleSheet(_dialog_qss())

    confirm = box.addButton(
        confirm_text, QMessageBox.ButtonRole.DestructiveRole
    )
    cancel = box.addButton(cancel_text, QMessageBox.ButtonRole.RejectRole)
    _dress(confirm, danger_outline_button_qss())
    _dress(cancel, ghost_button_qss())
    box.setDefaultButton(cancel)
    box.setEscapeButton(cancel)

    _widen(box)
    box.exec()
    return box.clickedButton() is confirm


def alert(
    parent: Optional[QWidget],
    *,
    title: str,
    message: str,
    detail: str = "",
    details_text: str = "",
    tone: str = "error",
    button_text: str = "OK",
) -> None:
    """Tell the user something went wrong. tone picks the headline colour: error, warning, or info. details_text goes behind a Show Details button."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.NoIcon)
    box.setWindowTitle(title)
    box.setText(message)
    if detail:
        box.setInformativeText(detail)
    if details_text:
        box.setDetailedText(details_text)
    box.setStyleSheet(
        _dialog_qss(_TONE_COLORS.get(tone, TEXT_PRIMARY))
    )

    acknowledge = box.addButton(
        button_text, QMessageBox.ButtonRole.AcceptRole
    )
    _dress(acknowledge, accent_button_qss())
    box.setDefaultButton(acknowledge)
    box.setEscapeButton(acknowledge)

    _widen(box)
    box.exec()
