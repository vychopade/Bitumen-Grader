"""Quick offline checks for the ML backend (no PyQt).

    python -m pytest tests/smoke_test.py -v
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from app.ml.cnn_model import BitumenRegressor
from app.ml.dataset import RegressionDataset
from app.ml.predictor import RegressionPredictor
from app.ml.trainer import RegressionTrainer, RegressionTrainingResult
from app.utils.model_io import (
    format_r2_headline,
    list_saved_models,
    load_model,
    resolve_model_r2,
    save_model,
)

_DUMMY_OUTPUT_STATS = {
    "Water": {"mean": 8.0, "std": 1.5},
    "Solids": {"mean": 12.0, "std": 2.0},
    "Bitumen": {"mean": 88.0, "std": 3.0},
}


def _make_dummy_result(output_stats=None, normalise_targets: bool = False) -> RegressionTrainingResult:
    """Minimal RegressionTrainingResult for save_model tests."""
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
                "water_r2": 0.4,
                "solids_r2": 0.5,
                "bitumen_r2": 0.6,
                "sum_deviation": 3.0,
            }
        ],
        output_stats=output_stats or dict(_DUMMY_OUTPUT_STATS),
        normalise_targets=normalise_targets,
        test_mae={"Water": 0.40, "Solids": 0.25, "Bitumen": 0.60},
        test_loss=0.02,
        test_sum_deviation=1.5,
        best_val_r2={"Water": 0.50, "Solids": 0.55, "Bitumen": 0.70},
        test_r2={"Water": 0.48, "Solids": 0.52, "Bitumen": 0.66},
        best_val_cls_acc={"Water": 0.6, "Solids": 0.7, "Bitumen": 0.8},
        test_cls_acc={"Water": 0.55, "Solids": 0.65, "Bitumen": 0.75},
    )


def _write_tiny_dataset(tmp_path: Path, count: int = 20):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    filenames = [f"sample_{index}.jpg" for index in range(count)]
    for filename in filenames:
        Image.new("RGB", (64, 64), color=(200, 200, 200)).save(image_dir / filename)
    rows = [
        {
            "Image": filename,
            "Pan": 3 + (index % 4),
            "Water": 50.0 + index,
            "Solids": 20.0 + (index % 5),
            "Bitumen": 30.0 - (index % 5),
        }
        for index, filename in enumerate(filenames)
    ]
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, image_dir


def test_bitumen_regressor_forward_pass_shape() -> None:
    """One image in → 3 outputs (Water, Solids, Bitumen)."""
    model = BitumenRegressor(pretrained=False)
    model.eval()

    dummy_input = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 3)
    assert model.architecture == "baseline"


def test_bitumen_regressor_from_scratch_flag() -> None:
    """Default architecture is the compact baseline, not an ImageNet backbone."""
    model = BitumenRegressor(pretrained=False)
    assert model.architecture == "baseline"
    with torch.no_grad():
        output = model(torch.rand(1, 3, 256, 256))
    assert output.shape == (1, 3)


def test_resnet50_and_vgg16_c2_heads() -> None:
    """Transfer variants keep a 3-d output; C2 is a 2-layer head."""
    resnet = BitumenRegressor(architecture="resnet50", pretrained=False, head="c2")
    vgg = BitumenRegressor(architecture="vgg16", pretrained=False, head="native")
    dummy = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        assert resnet(dummy).shape == (1, 3)
        assert vgg(dummy).shape == (1, 3)
    assert resnet.head_type == "c2"
    assert isinstance(resnet.head, torch.nn.Sequential)
    assert isinstance(vgg.head, torch.nn.Linear)


def test_backbone_freeze_helpers() -> None:
    model = BitumenRegressor(pretrained=False)
    model.freeze_backbone()
    assert all(not parameter.requires_grad for parameter in model.backbone_parameters())
    assert all(parameter.requires_grad for parameter in model.head_parameters())
    model.unfreeze_backbone()
    assert all(parameter.requires_grad for parameter in model.backbone_parameters())


def test_init_output_bias_predicts_train_means() -> None:
    """Fresh heads should emit the label means before any training."""
    model = BitumenRegressor(pretrained=False)
    stats = {
        "Water": {"mean": 70.0, "std": 5.0},
        "Solids": {"mean": 18.0, "std": 3.0},
        "Bitumen": {"mean": 12.0, "std": 4.0},
    }
    model.init_output_bias(stats)
    model.eval()
    with torch.no_grad():
        output = model(torch.rand(4, 3, 256, 256))
    expected = torch.tensor([70.0, 18.0, 12.0])
    assert torch.allclose(output, expected.expand_as(output), atol=1e-4)


def test_predictor_round_trip(tmp_path: Path) -> None:
    """save → load → predict returns the expected keys."""
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
    """If normalise_targets is False, treat raw outputs as percentages."""
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
    """Metadata JSON should store normalise_targets."""
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
    assert saved[0]["test_mae"]["Water"] == 0.40
    assert paths["metadata_path"].exists()


def test_regression_dataset_filename_matching(tmp_path: Path) -> None:
    """Every CSV row should match its image file."""
    csv_path, image_dir = _write_tiny_dataset(tmp_path, count=5)

    dataset = RegressionDataset(
        str(csv_path), str(image_dir), split="train", val_fraction=0.2, test_fraction=0.2, seed=42
    )

    match_summary = dataset.get_match_summary()
    assert match_summary["matched"] == 5

    output_stats = dataset.get_output_stats()
    assert set(output_stats.keys()) == {"Water", "Solids", "Bitumen"}

    crop_names = [type(step).__name__ for step in dataset.train_transforms.transforms]
    assert "RandomResizedCrop" not in crop_names
    assert "ColorJitter" not in crop_names
    assert "RandomHorizontalFlip" in crop_names
    resize = dataset.train_transforms.transforms[0]
    assert getattr(resize, "size", None) == (256, 256) or getattr(resize, "size", None) == 256

    tensor, _target = dataset[0]
    assert tuple(tensor.shape) == (3, 256, 256)


def test_dataset_matches_images_in_nested_folders(tmp_path: Path) -> None:
    """Labels can name a file that lives in a subfolder of the photo directory."""
    from app.utils.media import collect_images

    nested = tmp_path / "photos" / "run_a"
    nested.mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(nested / "sample_0.jpg")
    Image.new("RGB", (32, 32), color=(40, 50, 60)).save(nested / "sample_1.jpg")
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {"Image": "sample_0.jpg", "Pan": 3, "Water": 10.0, "Solids": 20.0, "Bitumen": 70.0},
            {"Image": "sample_1", "Pan": 4, "Water": 11.0, "Solids": 21.0, "Bitumen": 68.0},
        ]
    ).to_csv(csv_path, index=False)

    found = collect_images(tmp_path / "photos")
    assert len(found) == 2

    dataset = RegressionDataset(
        str(csv_path), str(tmp_path / "photos"), split="train", val_fraction=0.0, test_fraction=0.0, seed=42
    )
    assert dataset.get_match_summary()["matched"] == 2
    assert dataset.get_match_summary()["unmatched"] == 0


def test_regression_dataset_train_val_test_split(tmp_path: Path) -> None:
    csv_path, image_dir = _write_tiny_dataset(tmp_path, count=20)
    kwargs = dict(csv_path=str(csv_path), image_dir=str(image_dir), val_fraction=0.2, test_fraction=0.15, seed=42)
    train = RegressionDataset(split="train", **kwargs)
    val = RegressionDataset(split="val", **kwargs)
    test = RegressionDataset(split="test", **kwargs)

    assert len(train) + len(val) + len(test) == 20
    assert len(train) > 0 and len(val) > 0 and len(test) > 0

    train_paths = {item["image_path"] for item in train.data}
    val_paths = {item["image_path"] for item in val.data}
    test_paths = {item["image_path"] for item in test.data}
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)


def test_mixed_source_resolutions_standardize_to_model_size(tmp_path: Path) -> None:
    """Phone, crop, and thumbnail photos all become 3×256×256 tensors."""
    from app.utils.image_utils import build_eval_transforms, cap_long_edge, prepare_image

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    sizes = [(64, 64), (800, 600), (1920, 1080), (120, 400), (4000, 3000)]
    rows = []
    for index, size in enumerate(sizes):
        filename = f"sample_{index}.jpg"
        Image.new("RGB", size, color=(180, 90, 40)).save(image_dir / filename)
        rows.append(
            {
                "Image": filename,
                "Pan": 4,
                "Water": 10.0 + index,
                "Solids": 20.0 + index,
                "Bitumen": 70.0 - index,
            }
        )
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    dataset = RegressionDataset(
        str(csv_path),
        str(image_dir),
        split="train",
        val_fraction=0.0,
        test_fraction=0.0,
        seed=42,
    )
    assert len(dataset) == len(sizes)
    for index in range(len(dataset)):
        tensor, _target = dataset[index]
        assert tuple(tensor.shape) == (3, 256, 256)

    eval_transform = build_eval_transforms(256)
    for size in sizes:
        prepared = prepare_image(Image.new("RGB", size, color=(10, 20, 30)), image_size=256)
        assert prepared.size == (256, 256)
        tensor = eval_transform(prepared)
        assert tuple(tensor.shape) == (3, 256, 256)

    huge = Image.new("RGB", (4000, 3000), color=(0, 0, 0))
    capped = cap_long_edge(huge, 512)
    assert capped.size == (512, 384)

    from app.utils.image_utils import standardize_to_model_size

    for size in sizes:
        squared = standardize_to_model_size(Image.new("RGB", size, color=(1, 2, 3)), 256)
        assert squared.size == (256, 256)
        assert squared.mode == "RGB"


def test_trainer_paper_recipe_runs(tmp_path: Path) -> None:
    """Short Adam + MSE run reports R², 3-bin accuracy, and a test eval."""
    csv_path, image_dir = _write_tiny_dataset(tmp_path, count=12)
    common = dict(
        csv_path=str(csv_path),
        image_dir=str(image_dir),
        val_fraction=0.25,
        test_fraction=0.25,
        normalise=False,
        seed=42,
    )
    train_ds = RegressionDataset(split="train", **common)
    val_ds = RegressionDataset(split="val", **common)
    test_ds = RegressionDataset(split="test", **common)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    model = BitumenRegressor(pretrained=False)
    trainer = RegressionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        learning_rate=0.001,
        num_epochs=2,
        output_stats=train_ds.get_output_stats(),
        normalise_targets=False,
        patience=0,
        test_loader=test_loader,
        adaptation="scratch",
        bin_edges=train_ds.get_bin_edges(),
        init_output_bias=True,
    )

    finished = []
    errors = []
    trainer.finished.connect(finished.append)
    trainer.error.connect(errors.append)
    trainer.run()

    assert not errors, errors
    assert len(finished) == 1
    result = finished[0]
    assert result.test_mae is not None
    assert set(result.test_mae.keys()) == {"Water", "Solids", "Bitumen"}
    assert result.test_r2 is not None
    assert set(result.test_r2.keys()) == {"Water", "Solids", "Bitumen"}
    assert result.test_cls_acc is not None
    assert result.final_epoch == 2
    assert result.stopped_early is False
    assert result.normalise_targets is False
    history = result.training_history[0]
    assert "bitumen_r2" in history
    assert "bitumen_cls_acc" in history


def test_save_model_and_list_saved_models_round_trip(tmp_path: Path) -> None:
    """save_model metadata shows up in list_saved_models."""
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
    assert metadata["normalise_targets"] is False
    assert metadata["test_loss"] == 0.02
    assert metadata["architecture"] == "baseline"
    assert metadata["head"] == "native"
    assert metadata["image_size"] == 256
    assert resolve_model_r2(metadata)["split"] == "test"
    assert format_r2_headline(metadata) == "R² 0.55"


def test_resolve_model_r2_prefers_test_then_history() -> None:
    test_scores = {"Water": 0.48, "Solids": 0.52, "Bitumen": 0.66}
    val_scores = {"Water": 0.50, "Solids": 0.55, "Bitumen": 0.70}
    resolved = resolve_model_r2({"test_r2": test_scores, "best_val_r2": val_scores})
    assert resolved["split"] == "test"
    assert resolved["scores"]["Bitumen"] == 0.66
    assert format_r2_headline({"test_r2": test_scores}) == "R² 0.55"

    from_history = resolve_model_r2(
        {
            "training_history": [
                {"water_r2": 0.10, "solids_r2": 0.10, "bitumen_r2": 0.10},
                {"water_r2": 0.40, "solids_r2": 0.50, "bitumen_r2": 0.60},
            ]
        }
    )
    assert from_history["split"] == "val"
    assert from_history["scores"]["Bitumen"] == 0.60
    assert resolve_model_r2({"best_val_mae": {"Water": 1.0}}) == {}
    assert format_r2_headline({}) == ""


def test_parse_campaign_id_from_filename() -> None:
    from app.ml.dataset import parse_campaign_id

    assert parse_campaign_id("IMG_0032_N12.jpg") == "N12"
    assert parse_campaign_id("IMG_0027_11.jpg") == "N11"
    assert parse_campaign_id("IMG_0039_N21 (repeated N9).jpg") == "N21"
    assert parse_campaign_id("plain.jpg") == "unknown"


def test_experiment_split_holds_out_campaigns(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    for campaign in (1, 2, 3, 4, 5):
        for index in range(4):
            filename = f"IMG_{campaign:04d}_N{campaign}_{index}.jpg"
            Image.new("RGB", (32, 32), color=(100, 100, 100)).save(image_dir / filename)
            rows.append(
                {
                    "Image": filename,
                    "Pan": 3,
                    "Water": 10.0 + campaign,
                    "Solids": 20.0,
                    "Bitumen": 70.0 - campaign,
                }
            )
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    kwargs = dict(
        csv_path=str(csv_path),
        image_dir=str(image_dir),
        val_fraction=0.2,
        test_fraction=0.2,
        seed=42,
        split_mode="experiment",
    )
    train = RegressionDataset(split="train", **kwargs)
    val = RegressionDataset(split="val", **kwargs)
    test = RegressionDataset(split="test", **kwargs)

    assert train.split_mode == "experiment"
    train_c = set(train.split_campaigns["train"])
    val_c = set(train.split_campaigns["val"])
    test_c = set(train.split_campaigns["test"])
    assert train_c.isdisjoint(val_c)
    assert train_c.isdisjoint(test_c)
    assert val_c.isdisjoint(test_c)
    assert train_c and val_c  # at least train + val campaigns

    train_paths = {item["image_path"] for item in train.data}
    val_paths = {item["image_path"] for item in val.data}
    test_paths = {item["image_path"] for item in test.data}
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)


def test_continue_training_from_checkpoint(tmp_path: Path) -> None:
    """A saved baseline can be loaded and trained further on a new split."""
    csv_path, image_dir = _write_tiny_dataset(tmp_path, count=12)
    common = dict(
        csv_path=str(csv_path),
        image_dir=str(image_dir),
        val_fraction=0.25,
        test_fraction=0.25,
        normalise=True,
        seed=42,
    )
    train_ds = RegressionDataset(split="train", **common)
    val_ds = RegressionDataset(split="val", **common)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    model = BitumenRegressor(architecture="baseline", pretrained=False, head="native")
    trainer = RegressionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        learning_rate=0.001,
        num_epochs=1,
        weight_decay=0.0001,
        output_stats=train_ds.get_output_stats(),
        normalise_targets=True,
        patience=5,
        adaptation="scratch",
    )
    finished = []
    trainer.finished.connect(finished.append)
    trainer.error.connect(lambda message: finished.append(message))
    trainer.run()
    assert finished and not isinstance(finished[0], str), finished

    paths = save_model(
        model=model,
        name="parent",
        output_stats=finished[0].output_stats,
        result=finished[0],
        save_dir=tmp_path,
    )
    continued = BitumenRegressor.from_checkpoint(paths["model_path"], {"architecture": "baseline", "head": "native"})
    continued.train()
    trainer2 = RegressionTrainer(
        model=continued,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        learning_rate=0.0001,
        num_epochs=1,
        weight_decay=0.0001,
        output_stats=train_ds.get_output_stats(),
        normalise_targets=True,
        patience=5,
        adaptation="ft",
    )
    finished2 = []
    errors2 = []
    trainer2.finished.connect(finished2.append)
    trainer2.error.connect(errors2.append)
    trainer2.run()
    assert not errors2, errors2
    assert len(finished2) == 1
    assert finished2[0].final_epoch == 1


def test_feature_extraction_keeps_backbone_frozen(tmp_path: Path) -> None:
    csv_path, image_dir = _write_tiny_dataset(tmp_path, count=8)
    common = dict(
        csv_path=str(csv_path),
        image_dir=str(image_dir),
        val_fraction=0.25,
        test_fraction=0.25,
        normalise=True,
        seed=1,
    )
    train_ds = RegressionDataset(split="train", **common)
    val_ds = RegressionDataset(split="val", **common)
    model = BitumenRegressor(architecture="baseline", pretrained=False)
    trainer = RegressionTrainer(
        model=model,
        train_loader=DataLoader(train_ds, batch_size=4, shuffle=True),
        val_loader=DataLoader(val_ds, batch_size=4, shuffle=False),
        device=torch.device("cpu"),
        learning_rate=0.001,
        num_epochs=2,
        weight_decay=0.0,
        output_stats=train_ds.get_output_stats(),
        normalise_targets=True,
        patience=5,
        adaptation="fe",
    )
    trainer.run()
    assert all(not parameter.requires_grad for parameter in model.backbone_parameters())
    assert all(parameter.requires_grad for parameter in model.head_parameters())


def test_paper_recipe_constants() -> None:
    from app.ml.recipe import (
        BATCH_SIZE,
        CLS_BINS,
        DEFAULT_SPLIT_MODE,
        LEARNING_RATE_FE,
        LEARNING_RATE_FT,
        NUM_EPOCHS,
        TEST_FRACTION,
        VAL_FRACTION,
        WEIGHT_DECAY,
        learning_rate_for_adaptation,
    )

    assert BATCH_SIZE == 32
    assert NUM_EPOCHS == 100
    assert LEARNING_RATE_FT == 1e-4
    assert LEARNING_RATE_FE == 1e-3
    assert WEIGHT_DECAY == 0.0
    assert DEFAULT_SPLIT_MODE == "experiment"
    assert CLS_BINS == 3
    assert abs(TEST_FRACTION - 0.20) < 1e-9
    assert abs(VAL_FRACTION - 0.16) < 1e-9
    assert abs(1.0 - VAL_FRACTION - TEST_FRACTION - 0.64) < 1e-9
    assert learning_rate_for_adaptation("fe") == LEARNING_RATE_FE
    assert learning_rate_for_adaptation("ft") == LEARNING_RATE_FT
    assert learning_rate_for_adaptation("scratch") == LEARNING_RATE_FT


def test_dataset_raw_percentages_and_equal_frequency_bins(tmp_path: Path) -> None:
    csv_path, image_dir = _write_tiny_dataset(tmp_path, count=12)
    dataset = RegressionDataset(
        str(csv_path), str(image_dir), split="train", val_fraction=0.25, test_fraction=0.25, seed=42
    )
    assert dataset.normalise is False
    _, target = dataset[0]
    assert float(target.min()) >= 0.0
    edges = dataset.get_bin_edges()
    assert set(edges) == {"Water", "Solids", "Bitumen"}
    for name in edges:
        assert len(edges[name]) == 2
        assert edges[name][0] < edges[name][1]


def test_legacy_resnet18_round_trip(tmp_path: Path) -> None:
    model = BitumenRegressor(architecture="resnet18", pretrained=False, head="native")
    result = _make_dummy_result()
    paths = save_model(
        model=model,
        name="legacy18",
        output_stats=result.output_stats,
        result=result,
        save_dir=tmp_path,
        extra_metadata={"architecture": "resnet18", "head": "native", "image_size": 224},
    )
    loaded = BitumenRegressor.from_checkpoint(paths["model_path"], {"architecture": "resnet18"})
    with torch.no_grad():
        output = loaded(torch.rand(1, 3, 224, 224))
    assert output.shape == (1, 3)


def test_source_checkout_paths() -> None:
    """Unpackaged runs keep assets and models next to the project root."""
    from app.paths import ASSETS_DIR, MODELS_DIR, bundle_dir, user_data_dir

    root = Path(__file__).resolve().parent.parent
    assert bundle_dir() == root
    assert user_data_dir() == root
    assert ASSETS_DIR == root / "assets"
    assert MODELS_DIR == root / "models"
