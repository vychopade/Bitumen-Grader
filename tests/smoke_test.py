"""
Smoke tests for BitumenGrader's ML backend.

These are fast, offline sanity checks -- not a full test suite -- meant to
catch integration breakage between the CNN architecture, the predictor, and
the model save/load utilities. None of them touch PyQt6, so they run quickly
and without a display.

Run with:
    python -m pytest tests/smoke_test.py -v
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from app.ml.cnn_model import DEFAULT_GRADE_LABELS, BitumenCNN
from app.ml.predictor import ModelPredictor
from app.utils.model_io import load_model_metadata, save_model


def test_bitumen_cnn_forward_pass_shape() -> None:
    """A BitumenCNN with 5 classes should map a single image to 5 logits."""
    # pretrained=False keeps this test fast and network-independent (no
    # ImageNet weight download); architecture/shape behavior is identical.
    model = BitumenCNN(num_classes=5, pretrained=False)
    model.eval()

    dummy_input = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 5)


def test_model_predictor_predict_returns_expected_keys(tmp_path: Path) -> None:
    """ModelPredictor.predict() should return grade/confidence/probabilities."""
    grade_labels = list(DEFAULT_GRADE_LABELS)
    model = BitumenCNN(num_classes=len(grade_labels), pretrained=False)

    paths = save_model(
        model=model,
        name="smoke_test_dummy_model",
        num_classes=len(grade_labels),
        grade_labels=grade_labels,
        training_history=[],
        save_dir=tmp_path,
    )

    predictor = ModelPredictor(paths["model_path"], grade_labels=grade_labels)

    blank_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    result = predictor.predict(blank_image)

    assert set(result.keys()) == {"grade", "confidence", "all_probabilities"}
    assert result["grade"] in grade_labels
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["all_probabilities"].keys()) == set(grade_labels)


def test_save_and_load_model_metadata_round_trip(tmp_path: Path) -> None:
    """save_model()'s metadata sidecar should load back with the same fields."""
    grade_labels = ["PG 52-28", "PG 58-28"]
    model = BitumenCNN(num_classes=len(grade_labels), pretrained=False)
    training_history = [
        {"epoch": 1, "train_loss": 1.2, "val_loss": 1.1, "val_accuracy": 0.4},
        {"epoch": 2, "train_loss": 0.9, "val_loss": 0.8, "val_accuracy": 0.6},
    ]

    paths = save_model(
        model=model,
        name="roundtrip_model",
        num_classes=len(grade_labels),
        grade_labels=grade_labels,
        training_history=training_history,
        save_dir=tmp_path,
        best_val_accuracy=0.6,
    )

    assert paths["model_path"].exists()
    assert paths["metadata_path"].exists()

    metadata = load_model_metadata(paths["metadata_path"])

    assert metadata["name"] == "roundtrip_model"
    assert metadata["num_classes"] == len(grade_labels)
    assert metadata["grade_labels"] == grade_labels
    assert metadata["best_val_accuracy"] == 0.6
    assert metadata["training_history"] == training_history
    assert "created_at" in metadata
