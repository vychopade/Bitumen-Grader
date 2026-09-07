"""Paints matplotlib axes in the same dark colours as the rest of the UI."""

from __future__ import annotations

from typing import Optional

from app.theme import BORDER_COLOR, TEXT_SECONDARY


def style_axes(
    axes,
    ylabel: str,
    *,
    facecolor: Optional[str] = None,
    grid_alpha: float = 0.5,
    clear: bool = False,
) -> None:
    if clear:
        axes.clear()
    if facecolor is not None:
        axes.set_facecolor(facecolor)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    for spine in axes.spines.values():
        spine.set_color(BORDER_COLOR)
    axes.set_xlabel("Epoch", color=TEXT_SECONDARY, fontsize=8)
    axes.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=8)
    axes.grid(True, color=BORDER_COLOR, linewidth=0.5, alpha=grid_alpha)
