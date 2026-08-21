"""Shared color tokens and widget stylesheets.

Pages used to copy these hex values (and near-identical button QSS) in
every file. Keep visual behavior the same; use these names instead of
new literals.
"""
from __future__ import annotations

from app.constants import SUM_DEVIATION_OK, SUM_DEVIATION_TIGHT

# Palette — matches assets/style.qss
BACKGROUND_COLOR = "#1A1C20"
SURFACE_COLOR = "#22252C"
SURFACE_HOVER_COLOR = "#2A2E36"
BORDER_COLOR = "#33373F"
SEPARATOR_COLOR = "#2A2D34"
NAV_HOVER_COLOR = "#1D2027"

ACCENT_COLOR = "#E8A838"
ACCENT_HOVER_COLOR = "#C98A20"
ACCENT_PRESSED_COLOR = "#B37A1C"
ACCENT_DISABLED_BG = "#4A4230"
ACCENT_DISABLED_FG = "#8B8168"
TEXT_INVERSE = "#13151A"

TEXT_PRIMARY = "#E8E9EC"
TEXT_SECONDARY = "#8B909A"

DANGER_COLOR = "#E5484D"
DANGER_HOVER_BG = "rgba(229, 72, 77, 40)"
DANGER_DISABLED_FG = "#6B5050"
DANGER_DISABLED_BORDER = "#4A3838"

SUCCESS_COLOR = "#3CB878"
SUCCESS_HOVER_COLOR = "#58D492"
SUCCESS_BG = "#1E3327"
WARNING_COLOR = "#F5C518"
WARNING_BG = "#332B18"

# Chart series (MAE / loss plots)
WATER_LINE_COLOR = "#5B9BD5"
SOLIDS_LINE_COLOR = "#3CB878"
BITUMEN_LINE_COLOR = "#E8A838"
VAL_LINE_COLOR = "#5B9BD5"

PILL_BACKGROUND = "#2A2E36"
REGRESSION_PILL_COLOR = "#4A7FC1"

BUTTON_COLOR = SURFACE_HOVER_COLOR
BUTTON_HOVER_COLOR = BORDER_COLOR

RADIUS_SM = 6
RADIUS_MD = 8

PAGE_MARGINS = (32, 28, 32, 24)
PAGE_SPACING = 16

ACCENT_TINT_HOVER = "rgba(232, 168, 56, 30)"


def accent_button_qss(*, extra: str = "") -> str:
    extra_rule = f" {extra}" if extra else ""
    return (
        f"QPushButton {{ background-color: {ACCENT_COLOR}; color: {TEXT_INVERSE}; font-weight: 600;"
        f"border: none; border-radius: {RADIUS_SM}px; padding: 8px 16px;{extra_rule} }}"
        f"QPushButton:hover {{ background-color: {ACCENT_HOVER_COLOR}; }}"
        f"QPushButton:pressed {{ background-color: {ACCENT_PRESSED_COLOR}; }}"
        f"QPushButton:disabled {{ background-color: {ACCENT_DISABLED_BG}; color: {ACCENT_DISABLED_FG}; }}"
        f"QPushButton:focus {{ outline: none; border: 1px solid {ACCENT_COLOR}; }}"
    )


def ghost_button_qss(*, extra: str = "") -> str:
    """Amber outline on a transparent fill (secondary actions)."""
    extra_rule = f" {extra}" if extra else ""
    return (
        f"QPushButton {{ background-color: transparent; color: {ACCENT_COLOR}; font-weight: 600;"
        f"border: 1px solid {ACCENT_COLOR}; border-radius: {RADIUS_SM}px; padding: 8px 16px;{extra_rule} }}"
        f"QPushButton:hover {{ background-color: {ACCENT_TINT_HOVER}; }}"
        f"QPushButton:pressed {{ background-color: {ACCENT_TINT_HOVER}; }}"
        f"QPushButton:disabled {{ color: {TEXT_SECONDARY}; border: 1px solid {BORDER_COLOR}; }}"
        f"QPushButton:focus {{ outline: none; border: 1px solid {ACCENT_HOVER_COLOR}; }}"
    )


def secondary_button_qss() -> str:
    """Dark filled button with a hairline border."""
    return (
        f"QPushButton {{ background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};"
        f"border: 1px solid {BORDER_COLOR}; border-radius: {RADIUS_SM}px; padding: 8px 12px; font-size: 12px; }}"
        f"QPushButton:hover {{ background-color: {SURFACE_HOVER_COLOR}; }}"
        f"QPushButton:pressed {{ background-color: {NAV_HOVER_COLOR}; }}"
        f"QPushButton:disabled {{ color: {TEXT_SECONDARY}; border: 1px solid {SEPARATOR_COLOR}; }}"
        f"QPushButton:focus {{ outline: none; border: 1px solid {ACCENT_COLOR}; }}"
    )


def danger_outline_button_qss() -> str:
    return (
        f"QPushButton {{ background-color: transparent; color: {DANGER_COLOR}; font-weight: 600;"
        f"border: 1px solid {DANGER_COLOR}; border-radius: {RADIUS_SM}px; }}"
        f"QPushButton:hover {{ background-color: {DANGER_HOVER_BG}; }}"
        f"QPushButton:pressed {{ background-color: {DANGER_HOVER_BG}; }}"
        f"QPushButton:disabled {{ color: {DANGER_DISABLED_FG}; border: 1px solid {DANGER_DISABLED_BORDER}; }}"
        f"QPushButton:focus {{ outline: none; border: 1px solid {DANGER_COLOR}; }}"
    )


def drop_zone_qss(object_name: str, *, active: bool) -> str:
    border = ACCENT_HOVER_COLOR if active else ACCENT_COLOR
    background = SURFACE_HOVER_COLOR if active else SURFACE_COLOR
    return (
        f"QFrame#{object_name} {{ background-color: {background}; border: 2px dashed {border};"
        f"border-radius: {RADIUS_MD}px; }}"
    )


def card_qss(*, inset: bool = False) -> str:
    fill = BACKGROUND_COLOR if inset else SURFACE_COLOR
    return f"QFrame {{ background-color: {fill}; border-radius: {RADIUS_MD if not inset else 6}px; }}"


def sum_deviation_color(deviation: float) -> str:
    """Green if the composition is tight, amber if usable, red if off."""
    if deviation < SUM_DEVIATION_TIGHT:
        return SUCCESS_COLOR
    if deviation <= SUM_DEVIATION_OK:
        return ACCENT_COLOR
    return DANGER_COLOR
