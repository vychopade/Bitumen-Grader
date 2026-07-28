"""
Model save/load utilities.

Handles persisting trained PyTorch models (.pt files) to the models/
directory alongside a metadata JSON file (e.g. architecture info, training
hyperparameters, class labels, metrics), and loading them back for use in
the Model Manager and Predict pages.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


def save_model(
    model: nn.Module,
    name: str,
    num_classes: int,
    grade_labels: List[str],
    training_history: List[Dict[str, Any]],
    save_dir: Union[str, Path],
    best_val_accuracy: Optional[float] = None,
) -> Dict[str, Path]:
    """Save a trained model's weights and metadata to ``save_dir``.

    Writes two files into ``save_dir``:
        - ``{name}.pt``: a checkpoint dict containing the model's
          ``state_dict``, ``num_classes``, and ``grade_labels``, loadable via
          ``BitumenCNN.from_pretrained``.
        - ``{name}.json``: sidecar metadata containing ``name``,
          ``created_at`` (ISO timestamp), ``num_classes``, ``grade_labels``,
          ``best_val_accuracy``, and ``training_history``.

    Args:
        model: The trained model whose weights should be saved.
        name: Base filename (without extension) used for both output files.
        num_classes: Number of output classes the model was trained for.
        grade_labels: Human-readable grade label for each output class,
            ordered to match the model's output logits.
        training_history: List of per-epoch training log dicts (as produced
            by ``ModelTrainer``).
        save_dir: Directory to write the ``.pt``/``.json`` files into. It is
            created if it does not already exist.
        best_val_accuracy: Best validation accuracy achieved during
            training. If omitted, it is inferred from ``training_history``.

    Returns:
        A dict with keys ``"model_path"`` and ``"metadata_path"`` pointing
        to the files that were written.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"{name}.pt"
    metadata_path = save_dir / f"{name}.json"

    if best_val_accuracy is None:
        accuracies = [entry.get("val_accuracy", 0.0) for entry in training_history]
        best_val_accuracy = max(accuracies) if accuracies else 0.0

    checkpoint = {
        "state_dict": model.state_dict(),
        "num_classes": num_classes,
        "grade_labels": grade_labels,
    }
    torch.save(checkpoint, model_path)

    metadata = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_classes": num_classes,
        "grade_labels": grade_labels,
        "best_val_accuracy": best_val_accuracy,
        "training_history": training_history,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {"model_path": model_path, "metadata_path": metadata_path}


def load_model_metadata(json_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a saved model's sidecar metadata JSON file.

    Args:
        json_path: Path to the ``.json`` metadata file written by
            ``save_model``.

    Returns:
        The parsed metadata dict.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_saved_models(save_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """List metadata for every saved model in ``save_dir``, newest first.

    Args:
        save_dir: Directory to scan for ``.json`` metadata files.

    Returns:
        A list of metadata dicts (see ``load_model_metadata``), each
        augmented with ``"model_path"`` and ``"metadata_path"`` string keys
        pointing to the corresponding files. Sorted by ``created_at``
        descending (newest first). Returns an empty list if ``save_dir``
        does not exist, or if it contains no valid metadata files.
    """
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return []

    entries: List[Dict[str, Any]] = []
    for json_path in sorted(save_dir.glob("*.json")):
        try:
            metadata = load_model_metadata(json_path)
        except (json.JSONDecodeError, OSError):
            continue

        metadata["model_path"] = str(json_path.with_suffix(".pt"))
        metadata["metadata_path"] = str(json_path)
        entries.append(metadata)

    entries.sort(key=lambda entry: entry.get("created_at", ""), reverse=True)
    return entries
