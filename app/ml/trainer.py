import os
import tempfile

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


class RegressionTrainer(QObject):
    """Runs the training/validation loop for a BitumenRegressor model.

    Reports progress after every completed epoch via the ``progress`` signal
    and stops cleanly (checkpointing the best-so-far weights) either on
    cooperative cancellation (``stop_requested``) or early stopping once
    validation loss fails to improve for ``patience`` epochs in a row.
    """

    #: epoch, train_loss, val_loss, val_mae_dict, val_sum_deviation
    progress = Signal(int, float, float, dict, float)
    #: RegressionTrainingResult
    finished = Signal(object)
    error = Signal(str)
    #: epoch number at which early stopping triggered
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
        parent=None,
    ):
        super().__init__(parent)

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.output_stats = output_stats
        self.normalise_targets = normalise_targets
        self.patience = patience

        #: Cooperative cancellation flag, checked between epochs.
        self.stop_requested = False

    def request_stop(self) -> None:
        """Request that the training loop stop cleanly at the next safe point."""
        self.stop_requested = True

    def _build_optimizer(self):
        if self.optimizer_name == "SGD":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _denormalise_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Denormalise a (N, 3) [Water, Solids, Bitumen] batch back to original units."""
        output_names = ("Water", "Solids", "Bitumen")
        denormalised = torch.zeros_like(batch)
        for index, name in enumerate(output_names):
            mean = self.output_stats[name]["mean"]
            std = self.output_stats[name]["std"]
            denormalised[:, index] = batch[:, index] * std + mean
        return denormalised

    def run(self) -> None:
        try:
            self.model.to(self.device)
            optimizer = self._build_optimizer()
            loss_fn = nn.MSELoss()

            best_val_loss = float("inf")
            best_val_mae: dict = {}
            patience_counter = 0
            best_checkpoint_path = None
            training_history: list = []
            final_epoch = 0
            stopped_early = False

            for epoch in range(1, self.num_epochs + 1):
                if self.stop_requested:
                    break

                # -- Train phase --------------------------------------------
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

                # -- Validation phase -----------------------------------------
                self.model.eval()
                running_val_loss = 0.0
                val_batches = 0
                all_preds = []
                all_targets = []

                with torch.no_grad():
                    for images, targets in self.val_loader:
                        images = images.to(self.device)
                        targets = targets.to(self.device)

                        outputs = self.model(images)
                        loss = loss_fn(outputs, targets)
                        running_val_loss += loss.item()
                        val_batches += 1

                        all_preds.append(outputs.detach().cpu())
                        all_targets.append(targets.detach().cpu())

                val_loss = running_val_loss / val_batches if val_batches else 0.0

                if all_preds:
                    preds = torch.cat(all_preds, dim=0)
                    truths = torch.cat(all_targets, dim=0)
                    # Targets are only z-scored when normalise_targets is on;
                    # denormalise solely in that case so MAE stays in % units.
                    if self.normalise_targets:
                        preds = self._denormalise_batch(preds)
                        truths = self._denormalise_batch(truths)
                else:
                    preds = torch.zeros((0, 3))
                    truths = torch.zeros((0, 3))

                water_mae = (preds[:, 0] - truths[:, 0]).abs().mean().item() if len(preds) else 0.0
                solids_mae = (preds[:, 1] - truths[:, 1]).abs().mean().item() if len(preds) else 0.0
                bitumen_mae = (preds[:, 2] - truths[:, 2]).abs().mean().item() if len(preds) else 0.0

                pred_sum = preds[:, 0] + preds[:, 1] + preds[:, 2]
                val_sum_deviation = (pred_sum - 100.0).abs().mean().item() if len(preds) else 0.0

                val_mae_dict = {
                    "Water": water_mae,
                    "Solids": solids_mae,
                    "Bitumen": bitumen_mae,
                }

                training_history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "water_mae": water_mae,
                        "solids_mae": solids_mae,
                        "bitumen_mae": bitumen_mae,
                        "sum_deviation": val_sum_deviation,
                    }
                )

                self.progress.emit(epoch, train_loss, val_loss, val_mae_dict, val_sum_deviation)
                final_epoch = epoch

                # -- Checkpointing / early stopping ---------------------------
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_mae = val_mae_dict.copy()
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

            result = RegressionTrainingResult(
                best_val_loss=best_val_loss,
                best_val_mae=best_val_mae,
                final_epoch=final_epoch,
                stopped_early=stopped_early,
                training_history=training_history,
                output_stats=self.output_stats,
                normalise_targets=self.normalise_targets,
            )
            self.finished.emit(result)

        except Exception as exc:  # noqa: BLE001 - surface any training failure to the UI
            self.error.emit(str(exc))
            return
