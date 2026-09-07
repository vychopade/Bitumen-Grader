"""Force Water + Solids + Bitumen to 100% by predicting the two strongest heads and filling the third."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch

from app.constants import OUTPUT_NAMES

DEFAULT_RESIDUAL = "Solids"
# Bitumen is the grade this app is meant to report, so it is never filled in
# as 100 minus the other two. That would dump water and solids error onto
# bitumen. Water is the leftover instead.
PROTECTED_OUTPUTS = ("Bitumen",)


def predicted_outputs(residual_output: str) -> list:
    """The two grades the networks actually predict, in Water, Solids, Bitumen order."""
    return [name for name in OUTPUT_NAMES if name != residual_output]


def choose_residual_output(
    r2_by_name: Optional[Mapping[str, float]] = None,
    mae_by_name: Optional[Mapping[str, float]] = None,
) -> str:
    """Pick the leftover grade: lowest validation R², or highest MAE if R² is missing. Bitumen is never leftover. Solids is the fallback."""
    r2_by_name = r2_by_name or {}
    mae_by_name = mae_by_name or {}
    candidates = [
        name for name in OUTPUT_NAMES if name not in PROTECTED_OUTPUTS
    ] or list(OUTPUT_NAMES)
    ranked = []
    for name in candidates:
        if name not in r2_by_name:
            continue
        try:
            r2 = float(r2_by_name[name])
        except (TypeError, ValueError):
            continue
        mae = 0.0
        if name in mae_by_name:
            try:
                mae = float(mae_by_name[name])
            except (TypeError, ValueError):
                mae = 0.0
        ranked.append((r2, -mae, name))
    if ranked:
        ranked.sort()
        return ranked[0][2]

    mae_ranked = []
    for name in candidates:
        if name not in mae_by_name:
            continue
        try:
            mae_ranked.append((float(mae_by_name[name]), name))
        except (TypeError, ValueError):
            continue
    if mae_ranked:
        mae_ranked.sort(reverse=True)
        return mae_ranked[0][1]
    return DEFAULT_RESIDUAL


def close_composition(
    values: torch.Tensor,
    residual_output: str,
    names: Sequence[str] = OUTPUT_NAMES,
) -> torch.Tensor:
    """Keep the two predicted columns (clipped to 0–100) and set the leftover so each row sums to 100."""
    if residual_output not in names:
        residual_output = DEFAULT_RESIDUAL
    residual_index = list(names).index(residual_output)
    predicted = [index for index in range(len(names)) if index != residual_index]
    closed = values.detach().float().clone()
    first = closed[:, predicted[0]].clamp(min=0.0)
    second = closed[:, predicted[1]].clamp(min=0.0)
    pair_sum = first + second
    scale = torch.ones_like(pair_sum)
    overflow = pair_sum > 100.0
    if overflow.any():
        scale = torch.where(
            overflow, 100.0 / pair_sum.clamp(min=1e-8), scale
        )
    first = first * scale
    second = second * scale
    leftover = (100.0 - first - second).clamp(min=0.0)
    closed[:, predicted[0]] = first
    closed[:, predicted[1]] = second
    closed[:, residual_index] = leftover
    return closed
