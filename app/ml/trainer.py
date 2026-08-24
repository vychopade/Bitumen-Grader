"""Paper-faithful training loop for froth-image regression.

Prince & Prasad (Table 2): Adam, constant LR, MSE on process percentages,
100 epochs, batch 32. Fine-tuning / baseline use 1e-4; frozen feature
extraction uses 1e-3. The loop reports R² (the study's regression metric)
and 3-bin equal-frequency accuracy (the study's classification endpoint).
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PyQt6.QtCore import QObject, pyqtSignal as Signal

from app.constants import OUTPUT_NAMES
from app.ml.recipe import CLS_BINS, WEIGHT_DECAY, learning_rate_for_adaptation


@dataclass
class RegressionTrainingResult:
    best_val_loss: float
    best_val_mae: dict  # {"Water": x, "Solids": x, "Bitumen": x}
    final_epoch: int
    stopped_early: bool
    training_history: list  # list of per-epoch dicts
    output_stats: dict  # from dataset.get_output_stats()
    normalise_targets: bool
    test_mae: Optional[dict] = None
    test_loss: Optional[float] = None
    test_sum_deviation: Optional[float] = None
    best_val_r2: Optional[dict] = None
    test_r2: Optional[dict] = None
    best_val_cls_acc: Optional[dict] = None
    test_cls_acc: Optional[dict] = None


class RegressionTrainer(QObject):
    """Train/validate a BitumenRegressor. Emits progress each epoch.

    Follows the study protocol:
      * Adam with a single constant learning rate (no cosine, no per-layer LRs)
      * MSE on the process percentages (no composition penalty)
      * ``adaptation`` ``scratch`` / ``ft`` trains the whole net; ``fe`` freezes
        the backbone for the entire run
      * best checkpoint is the highest mean validation R²
      * full ``num_epochs`` unless the user stops (``patience`` is 0 by default)

    Optional ``patience`` > 0 restores the older early-stop behaviour for tests.
    """

    # epoch, train_loss, val_loss, val_mae_dict, val_sum_deviation, val_r2_dict
    progress = Signal(int, float, float, dict, float, dict)
    # RegressionTrainingResult
    finished = Signal(object)
    error = Signal(str)
    # epoch when early stopping fired
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
        # Accepted so older call sites / tests do not explode; ignored.
        use_differential_lrs: bool = False,
        backbone_lr_factor: float = 0.1,
        use_cosine_schedule: bool = False,
        freeze_backbone_epochs: int = 0,
        sum_penalty_weight: float = 0.0,
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
        self.adaptation = adaptation if adaptation in {"scratch", "ft", "fe"} else "ft"
        self.learning_rate = float(
            learning_rate if learning_rate is not None else learning_rate_for_adaptation(self.adaptation)
        )

        # Unused leftovers from the previous recipe; kept on the instance so
        # any debug prints / UI that still read them do not fail.
        self.use_differential_lrs = False
        self.backbone_lr_factor = backbone_lr_factor
        self.use_cosine_schedule = False
        self.freeze_backbone_epochs = 0
        self.sum_penalty_weight = 0.0

        self.stop_requested = False

    def request_stop(self) -> None:
        """Ask the loop to stop after the current epoch."""
        self.stop_requested = True

    def _build_optimizer(self):
        params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        return torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

    def _denormalise_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Undo z-scoring for a (N, 3) [Water, Solids, Bitumen] batch."""
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
        """Eval mode: (mean_loss, mae_dict, sum_deviation, r2_dict, cls_acc)."""
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
            name: (preds[:, index] - truths[:, index]).abs().mean().item() if len(preds) else 0.0
            for index, name in enumerate(OUTPUT_NAMES)
        }
        r2_dict = self._r2_dict(preds, truths)
        cls_acc = self._cls_acc_dict(preds, truths)
        pred_sum = preds.sum(dim=1) if len(preds) else torch.zeros(0)
        sum_deviation = (pred_sum - 100.0).abs().mean().item() if len(preds) else 0.0
        return mean_loss, mae_dict, sum_deviation, r2_dict, cls_acc

    @staticmethod
    def _r2_dict(preds: torch.Tensor, truths: torch.Tensor) -> dict:
        """Per-output R² on percentages (paper regression metric)."""
        if len(preds) < 2:
            return {name: 0.0 for name in OUTPUT_NAMES}
        ss_res = ((truths - preds) ** 2).sum(dim=0)
        ss_tot = ((truths - truths.mean(dim=0)) ** 2).sum(dim=0)
        r2 = 1.0 - ss_res / ss_tot.clamp(min=1e-8)
        return {name: r2[index].item() for index, name in enumerate(OUTPUT_NAMES)}

    def _cls_acc_dict(self, preds: torch.Tensor, truths: torch.Tensor) -> dict:
        """3-bin equal-frequency accuracy (paper classification endpoint)."""
        acc = {}
        if len(preds) == 0:
            return {name: 0.0 for name in OUTPUT_NAMES}
        for index, name in enumerate(OUTPUT_NAMES):
            edges = self.bin_edges.get(name) or []
            if len(edges) != CLS_BINS - 1:
                acc[name] = 0.0
                continue
            edge_tensor = torch.tensor(edges, dtype=preds.dtype)
            pred_bins = torch.bucketize(preds[:, index].contiguous(), edge_tensor)
            true_bins = torch.bucketize(truths[:, index].contiguous(), edge_tensor)
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

                train_loss = running_train_loss / train_batches if train_batches else 0.0

                val_loss, val_mae_dict, val_sum_deviation, val_r2_dict, val_cls_acc = self._evaluate_loader(
                    self.val_loader, loss_fn
                )
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

                self.progress.emit(epoch, train_loss, val_loss, val_mae_dict, val_sum_deviation, val_r2_dict)
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
                        self.early_stopped.emit(epoch)
                        stopped_early = True
                        break

            if best_checkpoint_path is not None:
                self.model.load_state_dict(torch.load(best_checkpoint_path, map_location=self.device))
                os.remove(best_checkpoint_path)

            test_mae = None
            test_loss = None
            test_sum_deviation = None
            test_r2 = None
            test_cls_acc = None
            if self.test_loader is not None and len(self.test_loader.dataset) > 0:
                test_loss, test_mae, test_sum_deviation, test_r2, test_cls_acc = self._evaluate_loader(
                    self.test_loader, loss_fn
                )

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

        except Exception as exc:  # noqa: BLE001 - show any training failure in the UI
            self.error.emit(str(exc))
            return
