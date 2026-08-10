"""
Smoke tests for BitumenGrader's regression ML backend.

These are fast, offline sanity checks -- not a full test suite -- meant to
catch integration breakage between the CNN regressor, the dataset loader,
the predictor, and the model save/load utilities. None of them touch
PyQt6, so they run quickly and without a display.

Run with:
    python -m pytest tests/smoke_test.py -v
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from app.ml.cnn_model import BitumenRegressor
from app.ml.dataset import RegressionDataset
from app.ml.predictor import RegressionPredictor
from app.ml.trainer import RegressionTrainingResult
from app.utils.model_io import list_saved_models, load_model, save_model

_DUMMY_OUTPUT_STATS = {
    "Water": {"mean": 8.0, "std": 1.5},
    "Solids": {"mean": 12.0, "std": 2.0},
    "Bitumen": {"mean": 88.0, "std": 3.0},
}


def _make_dummy_result(output_stats=None, normalise_targets: bool = True) -> RegressionTrainingResult:
    """Build a minimal ``RegressionTrainingResult`` for save_model() tests."""
    return RegressionTrainingResult(
        best_val_loss=0.01,
        best_val_mae={"Water": 0.34, "Solids": 0.21, "Bitumen": 0.58},
        final_epoch=3,
        stopped_early=False,
        training_history=[
            {
                "epoch": 1,
                "train_loss": 0.05,
                "val_loss": 0.06,
                "water_mae": 1.2,
                "solids_mae": 0.8,
                "bitumen_mae": 1.5,
                "sum_deviation": 3.0,
            }
        ],
        output_stats=output_stats or dict(_DUMMY_OUTPUT_STATS),
        normalise_targets=normalise_targets,
    )


def test_bitumen_regressor_forward_pass_shape() -> None:
    """A BitumenRegressor should map a single image to 3 raw outputs (Water, Solids, Bitumen)."""
    model = BitumenRegressor(pretrained=False)
    model.eval()

    dummy_input = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 3)


def test_bitumen_regressor_from_scratch_flag() -> None:
    """pretrained=False should build ResNet-18 without ImageNet weights."""
    model = BitumenRegressor(pretrained=False)
    # Random init still produces a valid forward shape.
    with torch.no_grad():
        output = model(torch.rand(1, 3, 224, 224))
    assert output.shape == (1, 3)


def test_predictor_round_trip(tmp_path: Path) -> None:
    """save_model() -> load_model() -> predict() should return a well-formed result dict."""
    model = BitumenRegressor(pretrained=False)
    result = _make_dummy_result()

    paths = save_model(
        model=model,
        name="smoke_test_dummy_model",
        output_stats=result.output_stats,
        result=result,
        save_dir=tmp_path,
    )

    predictor = load_model(paths["metadata_path"])

    blank_image = Image.new("RGB", (224, 224), color=(255, 255, 255))
    prediction = predictor.predict(blank_image)

    assert set(prediction.keys()) == {"Water", "Solids", "Bitumen", "sum", "sum_deviation", "sum_ok"}
    for label in ("Water", "Solids", "Bitumen"):
        assert isinstance(prediction[label]["value"], float)


def test_predictor_skips_denorm_when_targets_not_normalised(tmp_path: Path) -> None:
    """When normalise_targets is False, raw network outputs are treated as percentages."""
    model = BitumenRegressor(pretrained=False)
    result = _make_dummy_result(normalise_targets=False)

    paths = save_model(
        model=model,
        name="no_norm_model",
        output_stats=result.output_stats,
        result=result,
        save_dir=tmp_path,
    )

    metadata = {
        "output_stats": result.output_stats,
        "normalise_targets": False,
    }
    predictor = RegressionPredictor(paths["model_path"], metadata)

    # Force known raw outputs so we can assert no z-score inversion happened.
    class _FixedModel:
        def eval(self):
            return self

        def __call__(self, tensor):
            return torch.tensor([[10.0, 20.0, 70.0]])

    predictor.model = _FixedModel()
    prediction = predictor.predict(Image.new("RGB", (224, 224), color=(0, 0, 0)))

    assert prediction["Water"]["value"] == 10.0
    assert prediction["Solids"]["value"] == 20.0
    assert prediction["Bitumen"]["value"] == 70.0


def test_save_model_persists_normalise_targets_flag(tmp_path: Path) -> None:
    """Metadata sidecar must record whether training z-scored targets."""
    model = BitumenRegressor(pretrained=False)
    result = _make_dummy_result(normalise_targets=False)

    paths = save_model(
        model=model,
        name="flag_model",
        output_stats=result.output_stats,
        result=result,
        save_dir=tmp_path,
    )

    saved = list_saved_models(tmp_path)
    assert len(saved) == 1
    assert saved[0]["normalise_targets"] is False
    assert paths["metadata_path"].exists()


def test_regression_dataset_filename_matching(tmp_path: Path) -> None:
    """RegressionDataset should match every CSV row against its corresponding image file."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    filenames = [f"sample_{index}.jpg" for index in range(5)]
    for filename in filenames:
        Image.new("RGB", (32, 32), color=(200, 200, 200)).save(image_dir / filename)

    rows = [
        {
            "Image": filename,
            "Pan": 3 + (index % 4),
            "Water": 5.0 + index,
            "Solids": 10.0 + index,
            "Bitumen": 85.0 - index,
        }
        for index, filename in enumerate(filenames)
    ]
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    dataset = RegressionDataset(str(csv_path), str(image_dir), split="train", val_fraction=0.2, seed=42)

    match_summary = dataset.get_match_summary()
    assert match_summary["matched"] == 5

    output_stats = dataset.get_output_stats()
    assert set(output_stats.keys()) == {"Water", "Solids", "Bitumen"}

    # Gentler crop used for lab photos (not torchvision's default 0.08–1.0).
    crop = dataset.train_transforms.transforms[2]
    assert isinstance(crop, type(dataset.train_transforms.transforms[2]))
    assert getattr(crop, "scale", None) == (0.8, 1.0)


def test_save_model_and_list_saved_models_round_trip(tmp_path: Path) -> None:
    """save_model()'s metadata sidecar should round-trip through list_saved_models()."""
    model = BitumenRegressor(pretrained=False)
    result = _make_dummy_result()

    save_model(
        model=model,
        name="roundtrip_model",
        output_stats=result.output_stats,
        result=result,
        save_dir=tmp_path,
    )

    saved_models = list_saved_models(tmp_path)
    assert len(saved_models) == 1

    metadata = saved_models[0]
    assert metadata["model_type"] == "regression"
    assert metadata["output_names"] == ["Water", "Solids", "Bitumen"]
    assert set(metadata["best_val_mae"].keys()) == {"Water", "Solids", "Bitumen"}
    assert metadata["normalise_targets"] is True
