import os
import tempfile
from typing import Optional

import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import DataLoader
from PyQt6.QtCore import QObject, pyqtSignal as Signal


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


class RegressionTrainer(QObject):
    """Train/validate a BitumenRegressor. Emits progress each epoch.

    Stops early if stop_requested is set or val loss stalls for ``patience``
    epochs. ``adaptation`` is ``scratch`` / ``ft`` (optional freeze warmup then
    train the backbone) or ``fe`` (backbone stays frozen). Optional knobs:
    different LRs for backbone vs head, cosine schedule, sum-to-100 penalty,
    and a held-out test eval after restoring the best checkpoint.
    """

    # epoch, train_loss, val_loss, val_mae_dict, val_sum_deviation
    progress = Signal(int, float, float, dict, float)
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
        learning_rate,
        num_epochs,
        optimizer_name,
        weight_decay,
        output_stats,
        normalise_targets,
        patience,
        test_loader: Optional[DataLoader] = None,
        use_differential_lrs: bool = True,
        backbone_lr_factor: float = 0.1,
        use_cosine_schedule: bool = True,
        freeze_backbone_epochs: int = 3,
        sum_penalty_weight: float = 0.1,
        adaptation: str = "ft",
        parent=None,
    ):
        super().__init__(parent)

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.output_stats = output_stats
        self.normalise_targets = normalise_targets
        self.patience = patience
        self.use_differential_lrs = use_differential_lrs
        self.backbone_lr_factor = backbone_lr_factor
        self.use_cosine_schedule = use_cosine_schedule
        self.freeze_backbone_epochs = max(0, int(freeze_backbone_epochs))
        self.sum_penalty_weight = float(sum_penalty_weight)
        # scratch / ft: optional freeze warmup then train the backbone.
        # fe: freeze the backbone for the whole run (paper feature-extraction).
        self.adaptation = adaptation if adaptation in {"scratch", "ft", "fe"} else "ft"

        # Checked between epochs so we can stop cleanly.
        self.stop_requested = False

    def request_stop(self) -> None:
        """Ask the loop to stop after the current epoch."""
        self.stop_requested = True

    def _build_optimizer(self, *, backbone_trainable: bool):
        head_params = [parameter for parameter in self.model.head_parameters() if parameter.requires_grad]
        if backbone_trainable and self.use_differential_lrs:
            backbone_params = [
                parameter for parameter in self.model.backbone_parameters() if parameter.requires_grad
            ]
            param_groups = [
                {"params": backbone_params, "lr": self.learning_rate * self.backbone_lr_factor},
                {"params": head_params, "lr": self.learning_rate},
            ]
            # Skip empty groups (e.g. backbone still frozen).
            param_groups = [group for group in param_groups if group["params"]]
        else:
            params = head_params
            if backbone_trainable:
                params = params + [
                    parameter for parameter in self.model.backbone_parameters() if parameter.requires_grad
                ]
            param_groups = [{"params": params, "lr": self.learning_rate}]

        if self.optimizer_name == "SGD":
            return torch.optim.SGD(
                param_groups,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        return torch.optim.Adam(
            param_groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _build_scheduler(self, optimizer, remaining_epochs: int):
        if not self.use_cosine_schedule or remaining_epochs <= 0:
            return None
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(remaining_epochs, 1))

    def _denormalise_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Undo z-scoring for a (N, 3) [Water, Solids, Bitumen] batch."""
        output_names = ("Water", "Solids", "Bitumen")
        denormalised = torch.zeros_like(batch)
        for index, name in enumerate(output_names):
            mean = self.output_stats[name]["mean"]
            std = self.output_stats[name]["std"]
            denormalised[:, index] = batch[:, index] * std + mean
        return denormalised

    def _to_percentages(self, batch: torch.Tensor) -> torch.Tensor:
        if self.normalise_targets:
            return self._denormalise_batch(batch)
        return batch

    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
        mse = loss_fn(outputs, targets)
        if self.sum_penalty_weight <= 0:
            return mse
        preds_pct = self._to_percentages(outputs)
        sum_penalty = ((preds_pct.sum(dim=1) - 100.0) ** 2).mean()
        return mse + self.sum_penalty_weight * sum_penalty

    def _evaluate_loader(self, loader: DataLoader, loss_fn: nn.Module):
        """Eval mode: (mean_loss, mae_dict, sum_deviation)."""
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
                loss = self._compute_loss(outputs, targets, loss_fn)
                running_loss += loss.item()
                batches += 1
                all_preds.append(outputs.detach().cpu())
                all_targets.append(targets.detach().cpu())

        mean_loss = running_loss / batches if batches else 0.0
        if all_preds:
            preds = torch.cat(all_preds, dim=0)
            truths = torch.cat(all_targets, dim=0)
            if self.normalise_targets:
                preds = self._denormalise_batch(preds)
                truths = self._denormalise_batch(truths)
        else:
            preds = torch.zeros((0, 3))
            truths = torch.zeros((0, 3))

        mae_dict = {
            "Water": (preds[:, 0] - truths[:, 0]).abs().mean().item() if len(preds) else 0.0,
            "Solids": (preds[:, 1] - truths[:, 1]).abs().mean().item() if len(preds) else 0.0,
            "Bitumen": (preds[:, 2] - truths[:, 2]).abs().mean().item() if len(preds) else 0.0,
        }
        r2_dict = self._r2_dict(preds, truths)
        pred_sum = preds[:, 0] + preds[:, 1] + preds[:, 2]
        sum_deviation = (pred_sum - 100.0).abs().mean().item() if len(preds) else 0.0
        return mean_loss, mae_dict, sum_deviation, r2_dict

    @staticmethod
    def _r2_dict(preds: torch.Tensor, truths: torch.Tensor) -> dict:
        """Per-output R² on denormalised percentages (paper regression metric)."""
        names = ("Water", "Solids", "Bitumen")
        if len(preds) < 2:
            return {name: 0.0 for name in names}
        ss_res = ((truths - preds) ** 2).sum(dim=0)
        ss_tot = ((truths - truths.mean(dim=0)) ** 2).sum(dim=0)
        r2 = 1.0 - ss_res / ss_tot.clamp(min=1e-8)
        return {name: r2[index].item() for index, name in enumerate(names)}

    def run(self) -> None:
        try:
            self.model.to(self.device)
            loss_fn = nn.MSELoss()

            freeze_forever = self.adaptation == "fe"
            freeze_epochs = 0 if freeze_forever else self.freeze_backbone_epochs
            if freeze_forever or freeze_epochs > 0:
                self.model.freeze_backbone()
                optimizer = self._build_optimizer(backbone_trainable=False)
                scheduler_span = self.num_epochs if freeze_forever else freeze_epochs
                scheduler = self._build_scheduler(optimizer, scheduler_span)
            else:
                self.model.unfreeze_backbone()
                optimizer = self._build_optimizer(backbone_trainable=True)
                scheduler = self._build_scheduler(optimizer, self.num_epochs)

            best_val_loss = float("inf")
            best_val_mae: dict = {}
            best_val_r2: dict = {}
            patience_counter = 0
            best_checkpoint_path = None
            training_history: list = []
            final_epoch = 0
            stopped_early = False

            for epoch in range(1, self.num_epochs + 1):
                if self.stop_requested:
                    break

                if not freeze_forever and freeze_epochs > 0 and epoch == freeze_epochs + 1:
                    self.model.unfreeze_backbone()
                    optimizer = self._build_optimizer(backbone_trainable=True)
                    remaining = self.num_epochs - freeze_epochs
                    scheduler = self._build_scheduler(optimizer, remaining)

                # Train
                self.model.train()
                running_train_loss = 0.0
                train_batches = 0

                for images, targets in self.train_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self._compute_loss(outputs, targets, loss_fn)
                    loss.backward()
                    optimizer.step()

                    running_train_loss += loss.item()
                    train_batches += 1

                train_loss = running_train_loss / train_batches if train_batches else 0.0

                # Validate
                val_loss, val_mae_dict, val_sum_deviation, val_r2_dict = self._evaluate_loader(
                    self.val_loader, loss_fn
                )

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
                        "sum_deviation": val_sum_deviation,
                    }
                )

                self.progress.emit(epoch, train_loss, val_loss, val_mae_dict, val_sum_deviation)
                final_epoch = epoch

                if scheduler is not None:
                    scheduler.step()

                # Checkpoint / early stop
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_mae = val_mae_dict.copy()
                    best_val_r2 = val_r2_dict.copy()
                    if best_checkpoint_path is None:
                        fd, best_checkpoint_path = tempfile.mkstemp(suffix=".pt")
                        os.close(fd)
                    torch.save(self.model.state_dict(), best_checkpoint_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
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
            if self.test_loader is not None and len(self.test_loader.dataset) > 0:
                test_loss, test_mae, test_sum_deviation, test_r2 = self._evaluate_loader(
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
            )
            self.finished.emit(result)

        except Exception as exc:  # noqa: BLE001 - show any training failure in the UI
            self.error.emit(str(exc))
            return
