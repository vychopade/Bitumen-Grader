"""Save/load trained models and their metadata JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch.nn as nn

from app.constants import OUTPUT_NAMES

if TYPE_CHECKING:
    from app.ml.predictor import RegressionPredictor

_R2_HISTORY_KEYS = {
    "Water": "water_r2",
    "Solids": "solids_r2",
    "Bitumen": "bitumen_r2",
}


def save_model(
    model: nn.Module,
    name: str,
    output_stats: Dict[str, Dict[str, float]],
    result: Any,
    save_dir: Union[str, Path],
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Write ``{name}_{timestamp}.pt`` and a matching ``.json`` into
    ``save_dir``.

    Returns ``{"model_path": ..., "metadata_path": ...}``.
    """
    try:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Could not create model save directory '{save_dir}': {exc}"
        ) from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = save_dir / f"{name}_{timestamp}.pt"
    metadata_path = save_dir / f"{name}_{timestamp}.json"

    try:
        model.save(model_path)
    except OSError as exc:
        raise OSError(
            f"Could not save model weights to '{model_path}': {exc}"
        ) from exc

    metadata = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "regression",
        "output_names": list(OUTPUT_NAMES),
        "output_stats": output_stats,
        "num_outputs": len(OUTPUT_NAMES),
        "best_val_loss": result.best_val_loss,
        "best_val_mae": result.best_val_mae,
        "best_val_r2": getattr(result, "best_val_r2", None),
        "test_loss": result.test_loss,
        "test_mae": result.test_mae,
        "test_r2": getattr(result, "test_r2", None),
        "best_val_cls_acc": getattr(result, "best_val_cls_acc", None),
        "test_cls_acc": getattr(result, "test_cls_acc", None),
        "test_sum_deviation": result.test_sum_deviation,
        "final_epoch": result.final_epoch,
        "stopped_early": result.stopped_early,
        "normalise_targets": result.normalise_targets,
        "training_history": result.training_history,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    if hasattr(model, "config_dict"):
        for key, value in model.config_dict().items():
            metadata.setdefault(key, value)

    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except OSError as exc:
        raise OSError(
            f"Could not save model metadata to '{metadata_path}': {exc}"
        ) from exc

    return {"model_path": model_path, "metadata_path": metadata_path}


def load_model_metadata(json_path: Union[str, Path]) -> Dict[str, Any]:
    """Read the metadata ``.json`` next to a saved model."""
    json_path = Path(json_path)
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except OSError as exc:
        raise OSError(
            f"Could not read model metadata file '{json_path}': {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model metadata file '{json_path}' is not valid JSON: {exc}"
        ) from exc


def load_model(json_path: Union[str, Path]) -> RegressionPredictor:
    """Load metadata + sibling ``.pt`` and return a RegressionPredictor."""
    # Local import so app.utils doesn't hard-depend on torch-heavy app.ml.
    from app.ml.predictor import RegressionPredictor

    json_path = Path(json_path)
    metadata = load_model_metadata(json_path)

    model_path = json_path.with_suffix(".pt")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights file '{model_path}' was not found alongside "
            f"metadata '{json_path}'"
        )

    try:
        return RegressionPredictor(model_path, metadata)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model from '{model_path}': {exc}"
        ) from exc


def list_saved_models(save_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """Scan ``save_dir`` for model metadata; newest first. Empty if missing."""
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return []

    try:
        json_paths = sorted(save_dir.glob("*.json"))
    except OSError as exc:
        raise OSError(
            f"Could not scan model directory '{save_dir}': {exc}"
        ) from exc

    entries: List[Dict[str, Any]] = []
    for json_path in json_paths:
        try:
            metadata = load_model_metadata(json_path)
        except (OSError, ValueError):
            continue

        metadata["model_path"] = str(json_path.with_suffix(".pt"))
        metadata["metadata_path"] = str(json_path)
        entries.append(metadata)

    entries.sort(key=lambda entry: entry.get("created_at", ""), reverse=True)
    return entries


def format_created_at(
    created_at: Optional[str], fmt: str = "%b %d, %Y"
) -> str:
    """Format a model metadata ISO timestamp, or '' if missing/invalid."""
    if not created_at:
        return ""
    try:
        parsed = datetime.fromisoformat(created_at)
    except ValueError:
        return ""
    return parsed.strftime(fmt)


def _complete_r2(scores: Any) -> Optional[Dict[str, float]]:
    if not isinstance(scores, dict):
        return None
    try:
        return {name: float(scores[name]) for name in OUTPUT_NAMES}
    except (KeyError, TypeError, ValueError):
        return None


def mean_r2(scores: Dict[str, float]) -> float:
    """Mean of Water / Solids / Bitumen R² — same figure used to pick
    checkpoints."""
    return sum(float(scores[name]) for name in OUTPUT_NAMES) / len(
        OUTPUT_NAMES
    )


def _best_r2_from_history(
    history: List[Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    best_scores: Optional[Dict[str, float]] = None
    best_mean = float("-inf")
    for entry in history or []:
        try:
            scores = {
                name: float(entry[key])
                for name, key in _R2_HISTORY_KEYS.items()
            }
        except (KeyError, TypeError, ValueError):
            continue
        current_mean = mean_r2(scores)
        if current_mean > best_mean:
            best_mean = current_mean
            best_scores = scores
    return best_scores


def model_r2_splits(
    metadata: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Validation and test R² dicts when they were recorded.

    ``val`` comes from ``best_val_r2``, or the best-mean epoch in
    ``training_history`` for older files that only stored per-epoch R².
    """
    metadata = metadata or {}
    splits: Dict[str, Dict[str, float]] = {}
    test = _complete_r2(metadata.get("test_r2"))
    if test:
        splits["test"] = test
    val = _complete_r2(metadata.get("best_val_r2")) or _best_r2_from_history(
        metadata.get("training_history") or []
    )
    if val:
        splits["val"] = val
    return splits


def resolve_model_r2(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Headline accuracy: test R² if saved, otherwise validation R².

    Returns ``{"split": "test"|"val", "scores": {...}}`` or ``{}``.
    """
    splits = model_r2_splits(metadata)
    if "test" in splits:
        return {"split": "test", "scores": splits["test"]}
    if "val" in splits:
        return {"split": "val", "scores": splits["val"]}
    return {}


def format_r2_headline(metadata: Optional[Dict[str, Any]]) -> str:
    """``R² 0.42`` from the headline split, or '' if none was recorded."""
    resolved = resolve_model_r2(metadata)
    if not resolved:
        return ""
    return f"R² {mean_r2(resolved['scores']):.2f}"
