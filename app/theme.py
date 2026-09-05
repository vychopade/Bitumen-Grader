"""Colours and button styles shared by the pages. Kept simple on purpose so it feels like a lab tool, not a website."""

from __future__ import annotations

from app.constants import SUM_DEVIATION_OK, SUM_DEVIATION_TIGHT

# Same colours as assets/style.qss so Python-built widgets match the stylesheet.
BACKGROUND_COLOR = "#1C1E22"
SURFACE_COLOR = "#24262B"
SURFACE_HOVER_COLOR = "#2C2F35"
BORDER_COLOR = "#3A3D44"
SEPARATOR_COLOR = "#2E3136"
NAV_HOVER_COLOR = "#222429"

ACCENT_COLOR = "#C9A227"
ACCENT_HOVER_COLOR = "#B08E1F"
ACCENT_PRESSED_COLOR = "#9A7B1A"
ACCENT_DISABLED_BG = "#3F3A2C"
ACCENT_DISABLED_FG = "#8A8068"
TEXT_INVERSE = "#1C1E22"

TEXT_PRIMARY = "#E4E4E4"
TEXT_SECONDARY = "#9A9A9A"

DANGER_COLOR = "#C4544C"
DANGER_HOVER_BG = "rgba(196, 84, 76, 36)"
DANGER_DISABLED_FG = "#6B5050"
DANGER_DISABLED_BORDER = "#4A3838"

SUCCESS_COLOR = "#4A9A6A"
SUCCESS_BG = "#1E2A22"
WARNING_BG = "#2A2618"

# Line colours for the water, solids, bitumen, and validation series.
WATER_LINE_COLOR = "#6A8FB0"
SOLIDS_LINE_COLOR = "#4A9A6A"
BITUMEN_LINE_COLOR = "#C9A227"
VAL_LINE_COLOR = "#6A8FB0"

RADIUS_SM = 3

PAGE_MARGINS = (22, 18, 22, 16)
PAGE_SPACING = 12


def accent_button_qss() -> str:
    return (
        f"QPushButton {{ background-color: {ACCENT_COLOR};"
        f" color: {TEXT_INVERSE};"
        f"border: 1px solid {ACCENT_COLOR}; border-radius: {RADIUS_SM}px;"
        f"padding: 6px 12px; min-height: 24px; }}"
        f"QPushButton:hover {{ background-color: {ACCENT_HOVER_COLOR}; }}"
        f"QPushButton:pressed {{ background-color: {ACCENT_PRESSED_COLOR}; }}"
        f"QPushButton:disabled {{ background-color: {ACCENT_DISABLED_BG};"
        f" color: {ACCENT_DISABLED_FG};"
        f"border: 1px solid {ACCENT_DISABLED_BG}; }}"
        f"QPushButton:focus {{ outline: none; }}"
    )


def ghost_button_qss() -> str:
    return (
        f"QPushButton {{ background-color: transparent; color: {TEXT_PRIMARY};"
        f"border: 1px solid {BORDER_COLOR}; border-radius: {RADIUS_SM}px;"
        f"padding: 6px 12px; min-height: 24px; }}"
        f"QPushButton:hover {{ background-color: {SURFACE_HOVER_COLOR}; }}"
        f"QPushButton:pressed {{ background-color: {NAV_HOVER_COLOR}; }}"
        f"QPushButton:disabled {{ color: {TEXT_SECONDARY};"
        f" border: 1px solid {SEPARATOR_COLOR}; }}"
        f"QPushButton:focus {{ outline: none; }}"
    )


def danger_outline_button_qss() -> str:
    return (
        f"QPushButton {{ background-color: transparent; color: {DANGER_COLOR};"
        f"border: 1px solid {DANGER_COLOR}; border-radius: {RADIUS_SM}px;"
        f"padding: 6px 12px; min-height: 24px; }}"
        f"QPushButton:hover {{ background-color: {DANGER_HOVER_BG}; }}"
        f"QPushButton:pressed {{ background-color: {DANGER_HOVER_BG}; }}"
        f"QPushButton:disabled {{ color: {DANGER_DISABLED_FG};"
        f" border: 1px solid {DANGER_DISABLED_BORDER}; }}"
        f"QPushButton:focus {{ outline: none; }}"
    )


def link_button_qss(*, color: str = TEXT_SECONDARY) -> str:
    """Styles a button that looks like a text link so we do not stack a second outlined button. Pass a colour if you want something other than the default grey."""
    return (
        f"QPushButton {{ background: transparent; color: {color};"
        f" border: none;"
        f"padding: 4px 2px; text-align: left; font-size: 12px;"
        f" min-height: 20px; }}"
        f"QPushButton:hover {{ color: {TEXT_PRIMARY}; }}"
        f"QPushButton:disabled {{ color: {SEPARATOR_COLOR}; }}"
        f"QPushButton:focus {{ outline: none; }}"
    )


# QLabel inherits QFrame, so a blanket "QFrame { border }" rule would box every
# label. Stick this after those rules so labels stay plain text.
LABEL_RESET_QSS = "QLabel { border: none; background: transparent; }"


def drop_zone_qss(object_name: str, *, active: bool) -> str:
    border = ACCENT_COLOR if active else BORDER_COLOR
    background = SURFACE_COLOR if active else "transparent"
    return (
        f"QFrame#{object_name} {{ background-color: {background};"
        f" border: 1px dashed {border};"
        f"border-radius: {RADIUS_SM}px; }}"
        f"{LABEL_RESET_QSS}"
    )


def card_qss(*, inset: bool = False) -> str:
    fill = BACKGROUND_COLOR if inset else SURFACE_COLOR
    return (
        f"QFrame {{ background-color: {fill};"
        f" border: 1px solid {BORDER_COLOR};"
        f"border-radius: {RADIUS_SM}px; }}"
        f"{LABEL_RESET_QSS}"
    )


def sum_deviation_color(deviation: float) -> str:
    """Picks a colour for how far the three grades are from 100 percent. Green is tight, amber is still usable, red means treat the photo carefully. Pass the absolute deviation and you get a hex colour back."""
    if deviation < SUM_DEVIATION_TIGHT:
        return SUCCESS_COLOR
    if deviation <= SUM_DEVIATION_OK:
        return ACCENT_COLOR
    return DANGER_COLOR
