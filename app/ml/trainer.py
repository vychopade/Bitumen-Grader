"""
Training loop with callbacks.

Implements the training/validation loop for the CNN model defined in
cnn_model.py, exposing callback hooks (e.g. on_epoch_end, on_batch_end) so
the UI (progress panel) can be updated with live training progress without
the ML code depending directly on PyQt6.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PyQt6.QtCore import QObject, pyqtSignal
from torch.utils.data import DataLoader


@dataclass
class TrainingResult:
    """Summary of a completed (or early-stopped) training run.

    Attributes:
        best_val_accuracy: Highest validation accuracy observed across all
            completed epochs.
        final_epoch: The last epoch number that was completed (1-indexed).
        training_history: Per-epoch log entries, each a dict with keys
            ``epoch``, ``train_loss``, ``val_loss``, and ``val_accuracy``.
    """

    best_val_accuracy: float
    final_epoch: int
    training_history: List[Dict[str, Any]] = field(default_factory=list)


class ModelTrainer(QObject):
    """Runs the training/validation loop for a BitumenCNN model.

    Emits the ``progress_updated`` Qt signal after every completed epoch so
    the UI (e.g. the ProgressPanel component) can display live training
    progress, and supports cooperative cancellation via the
    ``stop_requested`` flag / ``request_stop()`` method.
    """

    #: Emitted after each epoch with (epoch, train_loss, val_loss, val_accuracy).
    progress_updated = pyqtSignal(int, float, float, float)

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        num_epochs: int = 10,
        batch_size: int = 32,
        optimizer_name: str = "Adam",
        weight_decay: float = 0.0,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay

        #: Cooperative cancellation flag, checked between batches/epochs.
        self.stop_requested = False

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()

        self._best_state_dict: Optional[Dict[str, Any]] = None
        self.best_val_accuracy: float = 0.0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Construct the optimizer named by ``optimizer_name`` ("Adam" or "SGD")."""
        if self.optimizer_name.lower() == "sgd":
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

    def request_stop(self) -> None:
        """Request that the training loop stop cleanly at the next safe point."""
        self.stop_requested = True

    def train(self) -> TrainingResult:
        """Run the full training loop.

        Trains for up to ``num_epochs`` epochs, emitting ``progress_updated``
        after each completed epoch and tracking the best checkpoint (highest
        validation accuracy) seen so far. Stops early and cleanly if
        ``stop_requested`` becomes true. Before returning, the model's
        weights are restored to the best checkpoint found.

        Returns:
            A ``TrainingResult`` summarizing the run.
        """
        self.model.to(self.device)

        history: List[Dict[str, Any]] = []
        final_epoch = 0

        for epoch in range(1, self.num_epochs + 1):
            if self.stop_requested:
                break

            train_loss = self._run_train_epoch()
            if self.stop_requested:
                final_epoch = epoch
                break

            val_loss, val_accuracy = self._run_validation_epoch()

            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self._best_state_dict = copy.deepcopy(self.model.state_dict())

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
            )
            final_epoch = epoch

            self.progress_updated.emit(epoch, train_loss, val_loss, val_accuracy)

        if self._best_state_dict is not None:
            self.model.load_state_dict(self._best_state_dict)

        return TrainingResult(
            best_val_accuracy=self.best_val_accuracy,
            final_epoch=final_epoch,
            training_history=history,
        )

    def _run_train_epoch(self) -> float:
        """Run one epoch of training and return the average training loss."""
        self.model.train()
        running_loss = 0.0
        num_samples = 0

        for images, labels in self.train_loader:
            if self.stop_requested:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

        return running_loss / num_samples if num_samples else 0.0

    def _run_validation_epoch(self) -> Tuple[float, float]:
        """Run one epoch of validation, returning (val_loss, val_accuracy)."""
        self.model.eval()
        running_loss = 0.0
        num_samples = 0
        correct = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                num_samples += images.size(0)

                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()

        val_loss = running_loss / num_samples if num_samples else 0.0
        val_accuracy = correct / num_samples if num_samples else 0.0
        return val_loss, val_accuracy
