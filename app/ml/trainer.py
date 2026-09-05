"""Training loop that follows the paper: Adam, constant learning rate, MSE on the three percents, 100 epochs, batch 32. Fine-tune and baseline use 1e-4. Frozen feature extraction uses 1e-3. Each epoch we log R squared and 3-bin accuracy so you can compare to the study."""

from __future__ import annotations

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
from app.ml.recipe import CLS_BINS, WEIGHT_DECAY, learning_rate_for_adaptation


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


class RegressionTrainer(QObject):
    """Runs train and val on a BitumenRegressor and emits progress each epoch. Adam with one constant learning rate, MSE on the raw percents, and we keep the checkpoint with the best mean val R squared. scratch and ft train the whole net. fe freezes the backbone. patience is 0 by default so we run all epochs unless the user hits Stop."""

    # Fired each epoch with losses, MAE, sum deviation, and R squared.
    progress = Signal(int, float, float, dict, float, dict)
    # Hands back a RegressionTrainingResult when the loop finishes.
    finished = Signal(object)
    error = Signal(str)
    # Epoch number if we stopped early.
    early_stopped = Signal(int)

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device,
        learning_rate=None,
        num_epochs=100,
        weight_decay=WEIGHT_DECAY,
        output_stats=None,
        normalise_targets=False,
        patience=0,
        test_loader: Optional[DataLoader] = None,
        adaptation: str = "ft",
        bin_edges: Optional[dict] = None,
        parent=None,
        init_output_bias: bool = False,
    ):
        super().__init__(parent)

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = int(num_epochs)
        self.weight_decay = float(weight_decay)
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
        self.stop_requested = False

    def request_stop(self) -> None:
        """Sets a flag so the loop finishes the current epoch and then quits. You do not pass anything."""

    def _build_optimizer(self):
        params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        return torch.optim.Adam(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def _denormalise_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Turns a z-scored batch back into percents using the train-set means and stds. Pass a tensor of shape N by 3. You get the same shape in percent."""
        denormalised = torch.zeros_like(batch)
        for index, name in enumerate(OUTPUT_NAMES):
            mean = self.output_stats[name]["mean"]
            std = self.output_stats[name]["std"]
            denormalised[:, index] = batch[:, index] * std + mean
        return denormalised

    def _to_percentages(self, batch: torch.Tensor) -> torch.Tensor:
        if self.normalise_targets:
            return self._denormalise_batch(batch)
        return batch

    def _evaluate_loader(self, loader: DataLoader, loss_fn: nn.Module):
        """Runs the model on a loader without updating weights. Pass a DataLoader and a loss function. You get mean loss, MAE, how far the three grades miss 100, R squared, and 3-bin accuracy."""
        self.model.eval()
        running_loss = 0.0
        batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
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
            preds = self._to_percentages(preds)
            truths = self._to_percentages(truths)
        else:
            preds = torch.zeros((0, 3))
            truths = torch.zeros((0, 3))

        mae_dict = {
            name: (preds[:, index] - truths[:, index]).abs().mean().item()
            if len(preds)
            else 0.0
            for index, name in enumerate(OUTPUT_NAMES)
        }
        r2_dict = self._r2_dict(preds, truths)
        cls_acc = self._cls_acc_dict(preds, truths)
        pred_sum = preds.sum(dim=1) if len(preds) else torch.zeros(0)
        sum_deviation = (
            (pred_sum - 100.0).abs().mean().item() if len(preds) else 0.0
        )
        return mean_loss, mae_dict, sum_deviation, r2_dict, cls_acc

    @staticmethod
    def _r2_dict(preds: torch.Tensor, truths: torch.Tensor) -> dict:
        """R squared for water, solids, and bitumen on percent-scale predictions. Pass predicted and true tensors. You get a dict of three scores."""
        if len(preds) < 2:
            return {name: 0.0 for name in OUTPUT_NAMES}
        ss_res = ((truths - preds) ** 2).sum(dim=0)
        ss_tot = ((truths - truths.mean(dim=0)) ** 2).sum(dim=0)
        r2 = 1.0 - ss_res / ss_tot.clamp(min=1e-8)
        return {
            name: r2[index].item() for index, name in enumerate(OUTPUT_NAMES)
        }

    def _cls_acc_dict(self, preds: torch.Tensor, truths: torch.Tensor) -> dict:
        """How often predicted and true values land in the same low/mid/high bin, using the train-set edges. Pass predicted and true tensors. You get a dict of three accuracies."""
        acc = {}
        if len(preds) == 0:
            return {name: 0.0 for name in OUTPUT_NAMES}
        for index, name in enumerate(OUTPUT_NAMES):
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
        values = [float(r2_dict.get(name, 0.0)) for name in OUTPUT_NAMES]
        return sum(values) / len(values) if values else float("-inf")

    def _apply_adaptation(self) -> None:
        if self.adaptation == "fe":
            self.model.freeze_backbone()
        else:
            self.model.unfreeze_backbone()

    def run(self) -> None:
        try:
            self.model.to(self.device)
            self._apply_adaptation()
            if self.init_output_bias:
                self.model.init_output_bias(self.output_stats)
            optimizer = self._build_optimizer()
            loss_fn = nn.MSELoss()

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

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = loss_fn(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    running_train_loss += loss.item()
                    train_batches += 1

                train_loss = (
                    running_train_loss / train_batches
                    if train_batches
                    else 0.0
                )

                (
                    val_loss,
                    val_mae_dict,
                    val_sum_deviation,
                    val_r2_dict,
                    val_cls_acc,
                ) = self._evaluate_loader(self.val_loader, loss_fn)
                mean_r2 = self._mean_r2(val_r2_dict)

                training_history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "water_mae": val_mae_dict["Water"],
                        "solids_mae": val_mae_dict["Solids"],
                        "bitumen_mae": val_mae_dict["Bitumen"],
                        "water_r2": val_r2_dict["Water"],
                        "solids_r2": val_r2_dict["Solids"],
                        "bitumen_r2": val_r2_dict["Bitumen"],
                        "water_cls_acc": val_cls_acc["Water"],
                        "solids_cls_acc": val_cls_acc["Solids"],
                        "bitumen_cls_acc": val_cls_acc["Bitumen"],
                        "sum_deviation": val_sum_deviation,
                    }
                )

                self.progress.emit(
                    epoch,
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
                        fd, best_checkpoint_path = tempfile.mkstemp(
                            suffix=".pt"
                        )
                        os.close(fd)
                    torch.save(self.model.state_dict(), best_checkpoint_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if self.patience > 0 and patience_counter >= self.patience:
                        self.early_stopped.emit(epoch)
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
            if (
                self.test_loader is not None
                and len(self.test_loader.dataset) > 0
            ):
                (
                    test_loss,
                    test_mae,
                    test_sum_deviation,
                    test_r2,
                    test_cls_acc,
                ) = self._evaluate_loader(self.test_loader, loss_fn)

            result = RegressionTrainingResult(
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
            self.finished.emit(result)

        except Exception as exc:  # noqa: BLE001
            # Push the error string to the UI so the user sees why training died.
            self.error.emit(str(exc))
            return
