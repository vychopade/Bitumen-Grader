"""
Model save/load utilities.

Handles persisting trained regression models (``BitumenRegressor`` weights as
``.pt`` files) to the ``models/`` directory alongside a metadata JSON sidecar
(output stats, training history, validation metrics), and loading them back
as ready-to-use ``RegressionPredictor`` instances for the Model Manager and
Grade pages.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING, Union

import torch.nn as nn

if TYPE_CHECKING:
    from app.ml.predictor import RegressionPredictor

OUTPUT_NAMES = ["Water", "Solids", "Bitumen"]


def save_model(
    model: nn.Module,
    name: str,
    output_stats: Dict[str, Dict[str, float]],
    result: Any,
    save_dir: Union[str, Path],
) -> Dict[str, Path]:
    """Save a trained regression model's weights and metadata to ``save_dir``.

    Writes two timestamped files into ``save_dir`` so repeated training runs
    of the same model name never collide:
        - ``{name}_{YYYYMMDD_HHMM}.pt``: the model's raw ``state_dict``,
          loadable via ``BitumenRegressor.from_pretrained``.
        - ``{name}_{YYYYMMDD_HHMM}.json``: sidecar metadata describing the
          output normalisation stats and training outcome.

    Args:
        model: The trained ``BitumenRegressor`` whose weights should be saved.
        name: Base filename, without extension or timestamp.
        output_stats: Per-output normalisation stats, e.g.
            ``{"Water": {"mean": x, "std": x}, "Solids": {...}, "Bitumen": {...}}``
            (typically from ``RegressionDataset.get_output_stats()``).
        result: The ``RegressionTrainingResult`` returned by
            ``RegressionTrainer`` once training finishes. Must expose
            ``best_val_loss``, ``best_val_mae``, ``final_epoch``,
            ``stopped_early``, and ``training_history``.
        save_dir: Directory to write the ``.pt``/``.json`` files into. It is
            created if it does not already exist.

    Returns:
        A dict with keys ``"model_path"`` and ``"metadata_path"`` pointing
        to the files that were written.

    Raises:
        OSError: If ``save_dir`` cannot be created, or either file cannot be
            written, with a descriptive message.
    """
    try:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Could not create model save directory '{save_dir}': {exc}") from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = save_dir / f"{name}_{timestamp}.pt"
    metadata_path = save_dir / f"{name}_{timestamp}.json"

    try:
        model.save(model_path)
    except OSError as exc:
        raise OSError(f"Could not save model weights to '{model_path}': {exc}") from exc

    metadata = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "regression",
        "output_names": list(OUTPUT_NAMES),
        "output_stats": output_stats,
        "num_outputs": len(OUTPUT_NAMES),
        "best_val_loss": result.best_val_loss,
        "best_val_mae": result.best_val_mae,
        "final_epoch": result.final_epoch,
        "stopped_early": result.stopped_early,
        "normalise_targets": result.normalise_targets,
        "training_history": result.training_history,
    }

    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except OSError as exc:
        raise OSError(f"Could not save model metadata to '{metadata_path}': {exc}") from exc

    return {"model_path": model_path, "metadata_path": metadata_path}


def load_model_metadata(json_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a saved model's sidecar metadata JSON file.

    Args:
        json_path: Path to the ``.json`` metadata file written by
            ``save_model``.

    Returns:
        The parsed metadata dict.

    Raises:
        OSError: If the file cannot be read, with a descriptive message.
        ValueError: If the file does not contain valid JSON.
    """
    json_path = Path(json_path)
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except OSError as exc:
        raise OSError(f"Could not read model metadata file '{json_path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model metadata file '{json_path}' is not valid JSON: {exc}") from exc


def load_model(json_path: Union[str, Path]) -> RegressionPredictor:
    """Load a saved regression model as a ready-to-use ``RegressionPredictor``.

    Reads the metadata JSON at ``json_path``, locates the sibling ``.pt``
    weights file (same stem, ``.pt`` extension), and instantiates a
    ``RegressionPredictor`` from both.

    Args:
        json_path: Path to the ``.json`` metadata file written by
            ``save_model``.

    Returns:
        A ``RegressionPredictor`` ready to call ``.predict()`` on.

    Raises:
        OSError: If the metadata file cannot be read.
        ValueError: If the metadata file is not valid JSON.
        FileNotFoundError: If the corresponding ``.pt`` weights file is missing.
        RuntimeError: If the model weights fail to load.
    """
    # Imported locally to avoid a hard import-time dependency between
    # app.utils (I/O helpers) and app.ml (torch-heavy model code).
    from app.ml.predictor import RegressionPredictor

    json_path = Path(json_path)
    metadata = load_model_metadata(json_path)

    model_path = json_path.with_suffix(".pt")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights file '{model_path}' was not found alongside metadata '{json_path}'"
        )

    try:
        return RegressionPredictor(model_path, metadata)
    except Exception as exc:
        raise RuntimeError(f"Could not load model from '{model_path}': {exc}") from exc


def list_saved_models(save_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """List metadata for every saved regression model in ``save_dir``, newest first.

    Args:
        save_dir: Directory to scan for ``.json`` metadata files.

    Returns:
        A list of metadata dicts (see ``load_model_metadata``), each
        augmented with ``"model_path"`` and ``"metadata_path"`` string keys
        pointing to the corresponding files. Sorted by ``created_at``
        descending (newest first). Returns an empty list if ``save_dir``
        does not exist, or if it contains no valid metadata files.

    Raises:
        OSError: If ``save_dir`` exists but cannot be scanned, with a
            descriptive message.
    """
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return []

    try:
        json_paths = sorted(save_dir.glob("*.json"))
    except OSError as exc:
        raise OSError(f"Could not scan model directory '{save_dir}': {exc}") from exc

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
