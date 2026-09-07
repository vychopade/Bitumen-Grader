"""Train three single-output networks, then pick the two strongest and fill the third so Water + Solids + Bitumen add to 100."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from PyQt6.QtCore import QObject
from PyQt6.QtCore import pyqtSignal as Signal
from torch.utils.data import DataLoader

from app.constants import OUTPUT_NAMES
from app.ml.cnn_model import find_feature_standardizer
from app.ml.composition import (
    choose_residual_output,
    close_composition,
    predicted_outputs,
)
from app.ml.recipe import (
    CLS_BINS,
    HEAD_LR_MULTIPLIER,
    HEAD_WEIGHT_DECAY,
    MIN_LR_FRACTION,
    NUM_EPOCHS,
    SMOOTH_L1_BETA,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    learning_rate_for_adaptation,
)

_OUTPUT_ORDER = ("Water", "Solids", "Bitumen")


@dataclass
class RegressionTrainingResult:
    best_val_loss: float
    best_val_mae: dict  # water, solids, bitumen mean absolute error
    final_epoch: int
    stopped_early: bool
    training_history: list  # one dict per epoch for the charts
    output_stats: dict  # means and stds from the train split
    normalise_targets: bool
    test_mae: Optional[dict] = None
    test_loss: Optional[float] = None
    test_sum_deviation: Optional[float] = None
    best_val_r2: Optional[dict] = None
    test_r2: Optional[dict] = None
    best_val_cls_acc: Optional[dict] = None
    test_cls_acc: Optional[dict] = None


@dataclass
class TripleRegressionResult:
    water: RegressionTrainingResult
    solids: RegressionTrainingResult
    bitumen: RegressionTrainingResult
    # Mean absolute sum deviation after closing Water+Solids+Bitumen to 100%
    test_normalised_sum_deviation: Optional[float] = None
    # Per-output MAE after the two-plus-residual close
    test_normalised_mae: Optional[dict] = None
    residual_output: Optional[str] = None
    predicted_outputs: Optional[list] = None
    test_r2: Optional[dict] = None
    val_composition_mae: Optional[dict] = None
    val_composition_r2: Optional[dict] = None
    # Scores after averaging every photo that shares one lab measurement.
    test_measurement_r2: Optional[dict] = None
    test_measurement_mae: Optional[dict] = None
    val_measurement_r2: Optional[dict] = None
    # MAE you would get by ignoring the photo and always answering with the
    # train-split average. Anything above this is worse than not looking.
    test_baseline_mae: Optional[dict] = None
    test_measurement_count: Optional[int] = None


def _part_for(result: TripleRegressionResult, name: str) -> RegressionTrainingResult:
    if name == "Water":
        return result.water
    if name == "Solids":
        return result.solids
    return result.bitumen


def merge_triple_histories(result: TripleRegressionResult) -> list:
    """Line up the three per-head logs by epoch so the charts still show Water, Solids, and Bitumen together."""
    parts = {name: _part_for(result, name) for name in OUTPUT_NAMES}
    lengths = [len(parts[name].training_history) for name in OUTPUT_NAMES]
    count = min(lengths) if lengths else 0
    merged = []
    for index in range(count):
        water = parts["Water"].training_history[index]
        solids = parts["Solids"].training_history[index]
        bitumen = parts["Bitumen"].training_history[index]
        merged.append(
            {
                "epoch": index + 1,
                "train_loss": (
                    water["train_loss"]
                    + solids["train_loss"]
                    + bitumen["train_loss"]
                )
                / 3.0,
                "val_loss": (
                    water["val_loss"] + solids["val_loss"] + bitumen["val_loss"]
                )
                / 3.0,
                "water_mae": water.get("water_mae", 0.0),
                "solids_mae": solids.get("solids_mae", 0.0),
                "bitumen_mae": bitumen.get("bitumen_mae", 0.0),
                "water_r2": water.get("water_r2", 0.0),
                "solids_r2": solids.get("solids_r2", 0.0),
                "bitumen_r2": bitumen.get("bitumen_r2", 0.0),
                "water_cls_acc": water.get("water_cls_acc", 0.0),
                "solids_cls_acc": solids.get("solids_cls_acc", 0.0),
                "bitumen_cls_acc": bitumen.get("bitumen_cls_acc", 0.0),
                "sum_deviation": 0.0,
                "lr": water.get("lr", solids.get("lr", bitumen.get("lr"))),
            }
        )
    return merged


def merge_triple_result(result: TripleRegressionResult) -> RegressionTrainingResult:
    """Flatten the three head results plus composition-closed scores into one RegressionTrainingResult for saving."""
    parts = [result.water, result.solids, result.bitumen]
    best_val_mae = result.val_composition_mae or {
        name: (_part_for(result, name).best_val_mae or {}).get(name, 0.0)
        for name in OUTPUT_NAMES
    }
    best_val_r2 = result.val_composition_r2 or {
        name: (_part_for(result, name).best_val_r2 or {}).get(name, 0.0)
        for name in OUTPUT_NAMES
    }
    losses = [part.best_val_loss for part in parts if part.best_val_loss != float("inf")]
    return RegressionTrainingResult(
        best_val_loss=(sum(losses) / len(losses)) if losses else float("inf"),
        best_val_mae=best_val_mae,
        final_epoch=max(part.final_epoch for part in parts),
        stopped_early=any(part.stopped_early for part in parts),
        training_history=merge_triple_histories(result),
        output_stats=result.water.output_stats,
        normalise_targets=result.water.normalise_targets,
        test_mae=result.test_normalised_mae,
        test_loss=None,
        test_sum_deviation=result.test_normalised_sum_deviation,
        best_val_r2=best_val_r2,
        test_r2=result.test_r2,
        best_val_cls_acc=None,
        test_cls_acc=None,
    )


class RegressionTrainer(QObject):
    """Train three single-output networks (water, solids, bitumen).

    Uses AdamW and Smooth L1. The checkpoint with the best mean validation R²
    is kept. scratch and ft train the whole net; fe freezes the backbone.
    patience 0 means run every epoch unless the user hits Stop.
    """

    progress = Signal(int, float, float, dict, float, dict)
    finished = Signal(object)
    error = Signal(str)
    early_stopped = Signal(int)

    def __init__(
        self,
        models: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device,
        learning_rate=None,
        num_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        head_weight_decay=HEAD_WEIGHT_DECAY,
        output_stats=None,
        normalise_targets=False,
        patience=0,
        test_loader: Optional[DataLoader] = None,
        adaptation: str = "ft",
        bin_edges: Optional[dict] = None,
        parent=None,
        init_output_bias: bool = False,
        grad_clip_max_norm: float = 1.0,
        use_scheduler: bool = True,
    ):
        super().__init__(parent)

        missing = [name for name in _OUTPUT_ORDER if name not in models]
        if missing:
            raise ValueError(
                "models must include Water, Solids, and Bitumen; "
                f"missing {missing}"
            )

        self.models = models
        self.model = None
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = int(num_epochs)
        self.weight_decay = float(weight_decay)
        self.head_weight_decay = float(head_weight_decay)
        self.output_stats = output_stats or {}
        self.normalise_targets = bool(normalise_targets)
        self.patience = max(0, int(patience))
        self.bin_edges = bin_edges or {}
        self.adaptation = (
            adaptation if adaptation in {"scratch", "ft", "fe"} else "ft"
        )
        self.learning_rate = float(
            learning_rate
            if learning_rate is not None
            else learning_rate_for_adaptation(self.adaptation)
        )

        self.init_output_bias = (
            bool(init_output_bias) and not self.normalise_targets
        )
        self.grad_clip_max_norm = float(grad_clip_max_norm)
        self.use_scheduler = bool(use_scheduler)
        self.stop_requested = False
        self.val_groups = self._loader_groups(val_loader)
        self.test_groups = self._loader_groups(test_loader)

    @staticmethod
    def _loader_groups(loader: Optional[DataLoader]) -> Optional[list]:
        """Which lab measurement each photo in an eval loader belongs to.

        Only valid because the val and test loaders are built with shuffle=False,
        so dataset order is prediction order.
        """
        dataset = getattr(loader, "dataset", None)
        getter = getattr(dataset, "sample_groups", None)
        return getter() if callable(getter) else None

    def request_stop(self) -> None:
        """Finish the current epoch, then quit. The UI Stop button calls this."""
        self.stop_requested = True

    def _build_scheduler(self, optimizer):
        """Warm up for a few epochs, then cosine-decay toward a small floor.

        A plateau scheduler was too jumpy on a small validation split: one
        unlucky epoch would crash the learning rate. Cosine is boring but
        predictable.
        """
        total = max(1, self.num_epochs)
        warmup = min(WARMUP_EPOCHS, max(0, total - 1))

        def factor(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / (warmup + 1)
            progress = (epoch - warmup) / max(1, total - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return MIN_LR_FRACTION + (1.0 - MIN_LR_FRACTION) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    def _build_optimizer(self):
        """Head and backbone get their own learning rate and weight decay.

        A pretrained backbone has to move slowly so its filters stay useful.
        A brand-new head needs a faster rate, and more weight decay, so it
        does not overfit the training pans.
        """
        head_ids = {
            id(parameter) for parameter in self.model.head_parameters()
        }
        head, backbone = [], []
        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                continue
            (head if id(parameter) in head_ids else backbone).append(parameter)

        # Strong head decay only makes sense when the backbone is frozen or
        # barely moving. From scratch, the head is just another layer.
        pretrained = bool(getattr(self.model, "pretrained", False))
        head_decay = (
            self.head_weight_decay if pretrained else self.weight_decay
        )
        groups = []
        if backbone:
            groups.append(
                {
                    "params": backbone,
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                }
            )
        if head:
            # Speed up the head only when a pretrained backbone is also
            # training. Frozen-head-only and from-scratch both use one rate.
            multiplier = HEAD_LR_MULTIPLIER if (backbone and pretrained) else 1.0
            groups.append(
                {
                    "params": head,
                    "lr": self.learning_rate * multiplier,
                    "weight_decay": head_decay,
                }
            )
        if not groups:
            groups = [{"params": [], "lr": self.learning_rate}]
        return torch.optim.AdamW(groups, lr=self.learning_rate)

    def _denormalise_batch(
        self, batch: torch.Tensor, names: Optional[tuple] = None
    ) -> torch.Tensor:
        """Turn a z-scored batch back into percents using the train-set means and stds."""
        names = tuple(names) if names is not None else OUTPUT_NAMES
        denormalised = torch.zeros_like(batch)
        for index, name in enumerate(names):
            mean = self.output_stats[name]["mean"]
            std = self.output_stats[name]["std"]
            denormalised[:, index] = batch[:, index] * std + mean
        return denormalised

    def _to_percentages(
        self, batch: torch.Tensor, names: Optional[tuple] = None
    ) -> torch.Tensor:
        if self.normalise_targets:
            return self._denormalise_batch(batch, names)
        return batch

    def _evaluate_loader(
        self,
        loader: DataLoader,
        loss_fn: nn.Module,
        output_index: Optional[int] = None,
    ):
        """Run the model on a loader without updating weights. Returns mean loss, MAE, how far the three grades miss 100, R², and 3-bin accuracy."""
        self.model.eval()
        running_loss = 0.0
        batches = 0
        all_preds = []
        all_targets = []
        names = (
            (OUTPUT_NAMES[output_index],)
            if output_index is not None
            else OUTPUT_NAMES
        )
        n_cols = 1 if output_index is not None else 3

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                if output_index is not None:
                    targets = targets[:, output_index : output_index + 1]
                outputs = self.model(images)
                loss = loss_fn(outputs, targets)
                running_loss += loss.item()
                batches += 1
                all_preds.append(outputs.detach().cpu())
                all_targets.append(targets.detach().cpu())

        mean_loss = running_loss / batches if batches else 0.0
        if all_preds:
            preds = torch.cat(all_preds, dim=0)
            truths = torch.cat(all_targets, dim=0)
            preds = self._to_percentages(preds, names)
            truths = self._to_percentages(truths, names)
        else:
            preds = torch.zeros((0, n_cols))
            truths = torch.zeros((0, n_cols))

        mae_dict = {
            name: (preds[:, index] - truths[:, index]).abs().mean().item()
            if len(preds)
            else 0.0
            for index, name in enumerate(names)
        }
        r2_dict = self._r2_dict(preds, truths, names)
        cls_acc = self._cls_acc_dict(preds, truths, names)
        if output_index is not None:
            # One head cannot close the composition; the UI shows this as blank.
            sum_deviation = 0.0
        else:
            pred_sum = preds.sum(dim=1) if len(preds) else torch.zeros(0)
            sum_deviation = (
                (pred_sum - 100.0).abs().mean().item() if len(preds) else 0.0
            )
        return mean_loss, mae_dict, sum_deviation, r2_dict, cls_acc

    @staticmethod
    def _r2_dict(
        preds: torch.Tensor, truths: torch.Tensor, names: Optional[tuple] = None
    ) -> dict:
        """R² for water, solids, and bitumen on percent-scale predictions."""
        names = tuple(names) if names is not None else OUTPUT_NAMES
        if len(preds) < 2:
            return {name: 0.0 for name in names}
        ss_res = ((truths - preds) ** 2).sum(dim=0)
        ss_tot = ((truths - truths.mean(dim=0)) ** 2).sum(dim=0)
        r2 = 1.0 - ss_res / ss_tot.clamp(min=1e-8)
        return {
            name: r2[index].item() for index, name in enumerate(names)
        }

    def _cls_acc_dict(
        self,
        preds: torch.Tensor,
        truths: torch.Tensor,
        names: Optional[tuple] = None,
    ) -> dict:
        """How often predicted and true values land in the same low/mid/high bin, using the train-set edges."""
        names = tuple(names) if names is not None else OUTPUT_NAMES
        acc = {}
        if len(preds) == 0:
            return {name: 0.0 for name in names}
        for index, name in enumerate(names):
            edges = self.bin_edges.get(name) or []
            if len(edges) != CLS_BINS - 1:
                acc[name] = 0.0
                continue
            edge_tensor = torch.tensor(edges, dtype=preds.dtype)
            pred_bins = torch.bucketize(
                preds[:, index].contiguous(), edge_tensor
            )
            true_bins = torch.bucketize(
                truths[:, index].contiguous(), edge_tensor
            )
            acc[name] = (pred_bins == true_bins).float().mean().item()
        return acc

    @staticmethod
    def _mean_r2(r2_dict: dict) -> float:
        values = [
            float(r2_dict[name]) for name in OUTPUT_NAMES if name in r2_dict
        ]
        return sum(values) / len(values) if values else float("-inf")

    def _apply_adaptation(self) -> None:
        if self.adaptation == "fe":
            self.model.freeze_backbone()
        else:
            self.model.unfreeze_backbone()

    def _init_single_output_bias(self, name: str) -> None:
        """Start the head predicting the training mean instead of zero.

        If labels are z-scored, that mean is 0. Using the raw percent mean
        here would put the bias way off for the first few epochs.
        """
        layer_fn = getattr(self.model, "_output_linear", None)
        layer = layer_fn() if callable(layer_fn) else None
        if layer is None or layer.bias is None:
            return
        if int(layer.bias.shape[0]) != 1:
            self.model.init_output_bias(self.output_stats)
            return
        mean = (
            0.0
            if self.normalise_targets
            else float((self.output_stats.get(name) or {}).get("mean", 0.0))
        )
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.fill_(mean)

    def _fit_feature_standardizer(self) -> None:
        """Fit the head's per-channel mean/std on the training photos.

        One pass over the train loader with the backbone frozen in eval mode.
        The values are stored in the .pt file and reused at grade time.
        """
        standardizer = find_feature_standardizer(self.model)
        backbone = getattr(self.model, "backbone", None)
        if standardizer is None or backbone is None:
            return
        was_training = self.model.training
        self.model.eval()
        blocks = []
        with torch.no_grad():
            for images, _ in self.train_loader:
                blocks.append(
                    backbone(images.to(self.device)).detach().cpu()
                )
        self.model.train(was_training)
        if blocks:
            standardizer.fit(torch.cat(blocks, dim=0))

    def _collect_percent_column(self, loader: DataLoader, output_index: int):
        self.model.eval()
        all_preds = []
        all_targets = []
        name = OUTPUT_NAMES[output_index]
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                all_preds.append(outputs.detach().cpu())
                all_targets.append(targets.detach().cpu())
        if not all_preds:
            return torch.zeros((0, 1)), torch.zeros((0, 3))
        preds = torch.cat(all_preds, dim=0)
        truths = torch.cat(all_targets, dim=0)
        preds = self._to_percentages(preds, (name,))
        return preds, truths

    def _percent_metrics(self, preds: torch.Tensor, truths: torch.Tensor):
        mae = {
            name: (preds[:, index] - truths[:, index]).abs().mean().item()
            for index, name in enumerate(OUTPUT_NAMES)
        }
        r2 = self._r2_dict(preds, truths, OUTPUT_NAMES)
        cls_acc = self._cls_acc_dict(preds, truths, OUTPUT_NAMES)
        sum_deviation = (
            (preds.sum(dim=1) - 100.0).abs().mean().item() if len(preds) else 0.0
        )
        return mae, r2, sum_deviation, cls_acc

    def _collect_all_percent_predictions(self, loader: DataLoader):
        column_preds = []
        truths = None
        for index, name in enumerate(_OUTPUT_ORDER):
            self.model = self.models[name]
            self.model.to(self.device)
            preds, truths = self._collect_percent_column(loader, index)
            column_preds.append(preds)
        if truths is None or len(truths) == 0:
            return torch.zeros((0, 3)), torch.zeros((0, 3))
        raw = torch.cat(column_preds, dim=1)
        truths = self._to_percentages(truths, OUTPUT_NAMES)
        return raw, truths

    @staticmethod
    def _average_by_group(
        preds: torch.Tensor, truths: torch.Tensor, groups: Optional[list]
    ):
        """Collapse every photo of one lab measurement into a single row.

        Four to twenty photos can share one lab result, so scoring per photo
        counts the same measurement many times and reports the noise of a single
        snapshot. Averaging first is also how the grades get used in practice.
        """
        if not groups or len(groups) != len(preds):
            return None, None
        order: dict = {}
        for index, key in enumerate(groups):
            order.setdefault(key, []).append(index)
        keys = list(order)
        pred_rows = torch.stack(
            [preds[order[key]].mean(dim=0) for key in keys]
        )
        truth_rows = torch.stack(
            [truths[order[key]].mean(dim=0) for key in keys]
        )
        return pred_rows, truth_rows

    def _measurement_metrics(
        self, preds: torch.Tensor, truths: torch.Tensor, groups: Optional[list]
    ):
        grouped_preds, grouped_truths = self._average_by_group(
            preds, truths, groups
        )
        if grouped_preds is None or len(grouped_preds) < 2:
            return None, None, None
        mae = {
            name: (grouped_preds[:, index] - grouped_truths[:, index])
            .abs()
            .mean()
            .item()
            for index, name in enumerate(OUTPUT_NAMES)
        }
        r2 = self._r2_dict(grouped_preds, grouped_truths, OUTPUT_NAMES)
        return mae, r2, len(grouped_preds)

    def _baseline_mae(self, truths: torch.Tensor) -> dict:
        """MAE of always answering with the train-split average, as a floor to beat."""
        return {
            name: (
                truths[:, index]
                - float((self.output_stats.get(name) or {}).get("mean", 0.0))
            )
            .abs()
            .mean()
            .item()
            for index, name in enumerate(OUTPUT_NAMES)
        }

    def _composition_metrics(
        self,
        loader: DataLoader,
        residual_output: str,
        groups: Optional[list] = None,
    ):
        raw, truths = self._collect_all_percent_predictions(loader)
        if len(raw) == 0:
            return None
        closed = close_composition(raw, residual_output)
        mae, r2, sum_deviation, cls_acc = self._percent_metrics(closed, truths)
        measurement_mae, measurement_r2, measurement_count = (
            self._measurement_metrics(closed, truths, groups)
        )
        return {
            "mae": mae,
            "r2": r2,
            "sum_deviation": sum_deviation,
            "cls_acc": cls_acc,
            "measurement_mae": measurement_mae,
            "measurement_r2": measurement_r2,
            "measurement_count": measurement_count,
            "baseline_mae": self._baseline_mae(truths),
        }

    def _head_scores(self, results: dict):
        r2 = {}
        mae = {}
        for name in _OUTPUT_ORDER:
            part = results[name]
            r2[name] = float((part.best_val_r2 or {}).get(name, 0.0))
            mae[name] = float((part.best_val_mae or {}).get(name, 0.0))
        return r2, mae

    def _run_one_output(self, name: str, model_index: int) -> RegressionTrainingResult:
        output_index = model_index
        self.model = self.models[name]
        self.model.to(self.device)
        self._apply_adaptation()
        self._fit_feature_standardizer()
        if self.init_output_bias:
            self._init_single_output_bias(name)
        optimizer = self._build_optimizer()
        scheduler = None
        if self.use_scheduler:
            scheduler = self._build_scheduler(optimizer)
        loss_fn = nn.SmoothL1Loss(beta=SMOOTH_L1_BETA)

        best_val_loss = float("inf")
        best_mean_r2 = float("-inf")
        best_val_mae: dict = {}
        best_val_r2: dict = {}
        best_val_cls_acc: dict = {}
        patience_counter = 0
        best_checkpoint_path = None
        training_history: list = []
        final_epoch = 0
        stopped_early = False
        epoch_offset = model_index * self.num_epochs

        # Save the weights from the epoch with the best mean val R squared, not just the last one.
        for epoch in range(1, self.num_epochs + 1):
            if self.stop_requested:
                break

            self.model.train()
            running_train_loss = 0.0
            train_batches = 0

            for images, targets in self.train_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                targets = targets[:, output_index : output_index + 1]

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = loss_fn(outputs, targets)
                loss.backward()
                if self.grad_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_max_norm
                    )
                optimizer.step()

                running_train_loss += loss.item()
                train_batches += 1

            train_loss = (
                running_train_loss / train_batches if train_batches else 0.0
            )

            (
                val_loss,
                val_mae_dict,
                val_sum_deviation,
                val_r2_dict,
                val_cls_acc,
            ) = self._evaluate_loader(
                self.val_loader, loss_fn, output_index=output_index
            )
            mean_r2 = self._mean_r2(val_r2_dict)
            # Read the rate before stepping, so the log shows the rate this
            # epoch actually trained at rather than the next one's.
            epoch_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()

            training_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "water_mae": val_mae_dict.get("Water", 0.0),
                    "solids_mae": val_mae_dict.get("Solids", 0.0),
                    "bitumen_mae": val_mae_dict.get("Bitumen", 0.0),
                    "water_r2": val_r2_dict.get("Water", 0.0),
                    "solids_r2": val_r2_dict.get("Solids", 0.0),
                    "bitumen_r2": val_r2_dict.get("Bitumen", 0.0),
                    "water_cls_acc": val_cls_acc.get("Water", 0.0),
                    "solids_cls_acc": val_cls_acc.get("Solids", 0.0),
                    "bitumen_cls_acc": val_cls_acc.get("Bitumen", 0.0),
                    "sum_deviation": val_sum_deviation,
                    "lr": epoch_lr,
                }
            )

            self.progress.emit(
                epoch + epoch_offset,
                train_loss,
                val_loss,
                val_mae_dict,
                val_sum_deviation,
                val_r2_dict,
            )
            final_epoch = epoch

            improved = mean_r2 > best_mean_r2 or (
                mean_r2 == best_mean_r2 and val_loss < best_val_loss
            )
            if improved:
                best_mean_r2 = mean_r2
                best_val_loss = val_loss
                best_val_mae = val_mae_dict.copy()
                best_val_r2 = val_r2_dict.copy()
                best_val_cls_acc = val_cls_acc.copy()
                if best_checkpoint_path is None:
                    fd, best_checkpoint_path = tempfile.mkstemp(suffix=".pt")
                    os.close(fd)
                torch.save(self.model.state_dict(), best_checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if self.patience > 0 and patience_counter >= self.patience:
                    self.early_stopped.emit(epoch + epoch_offset)
                    stopped_early = True
                    break

        if best_checkpoint_path is not None:
            self.model.load_state_dict(
                torch.load(best_checkpoint_path, map_location=self.device)
            )
            os.remove(best_checkpoint_path)

        test_mae = None
        test_loss = None
        test_sum_deviation = None
        test_r2 = None
        test_cls_acc = None
        if self.test_loader is not None and len(self.test_loader.dataset) > 0:
            (
                test_loss,
                test_mae,
                test_sum_deviation,
                test_r2,
                test_cls_acc,
            ) = self._evaluate_loader(
                self.test_loader, loss_fn, output_index=output_index
            )

        return RegressionTrainingResult(
            best_val_loss=best_val_loss,
            best_val_mae=best_val_mae,
            final_epoch=final_epoch,
            stopped_early=stopped_early,
            training_history=training_history,
            output_stats=self.output_stats,
            normalise_targets=self.normalise_targets,
            test_mae=test_mae,
            test_loss=test_loss,
            test_sum_deviation=test_sum_deviation,
            best_val_r2=best_val_r2 or None,
            test_r2=test_r2,
            best_val_cls_acc=best_val_cls_acc or None,
            test_cls_acc=test_cls_acc,
        )

    def run(self) -> None:
        try:
            results = {}
            for model_index, name in enumerate(_OUTPUT_ORDER):
                if self.stop_requested:
                    break
                results[name] = self._run_one_output(name, model_index)

            if len(results) != len(_OUTPUT_ORDER):
                self.error.emit(
                    "Training stopped before Water, Solids, and Bitumen "
                    "each finished a run."
                )
                return

            r2_by_name, mae_by_name = self._head_scores(results)
            residual = choose_residual_output(r2_by_name, mae_by_name)
            predicted = predicted_outputs(residual)

            val_scores = (
                self._composition_metrics(
                    self.val_loader, residual, self.val_groups
                )
                or {}
            )
            test_scores = {}
            if (
                self.test_loader is not None
                and len(self.test_loader.dataset) > 0
            ):
                test_scores = (
                    self._composition_metrics(
                        self.test_loader, residual, self.test_groups
                    )
                    or {}
                )

            result = TripleRegressionResult(
                water=results["Water"],
                solids=results["Solids"],
                bitumen=results["Bitumen"],
                test_normalised_sum_deviation=test_scores.get("sum_deviation"),
                test_normalised_mae=test_scores.get("mae"),
                residual_output=residual,
                predicted_outputs=predicted,
                test_r2=test_scores.get("r2"),
                val_composition_mae=val_scores.get("mae"),
                val_composition_r2=val_scores.get("r2"),
                test_measurement_r2=test_scores.get("measurement_r2"),
                test_measurement_mae=test_scores.get("measurement_mae"),
                val_measurement_r2=val_scores.get("measurement_r2"),
                test_baseline_mae=test_scores.get("baseline_mae"),
                test_measurement_count=test_scores.get("measurement_count"),
            )
            self.finished.emit(result)

        except Exception as exc:  # noqa: BLE001
            # Push the error string to the UI so the user sees why training died.
            self.error.emit(str(exc))
            return
