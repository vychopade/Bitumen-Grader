"""
Training progress panel widget.

Reusable widget combining a progress bar with a scrolling log/console area,
used to surface live feedback (epoch/batch progress, loss, metrics) during
model training. Also embeds a small matplotlib loss-curve chart and shows a
green completion banner once a run finishes successfully.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PyQt6.QtCore import Qt, pyqtSignal  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# --------------------------------------------------------------------------
# Design tokens (kept local so this component has no dependency on MainWindow)
# --------------------------------------------------------------------------

SURFACE_COLOR = "#22252C"
BORDER_COLOR = "#33373F"
ACCENT_COLOR = "#E8A838"
TEXT_PRIMARY = "#E8E9EC"
TEXT_SECONDARY = "#8B909A"
VAL_LINE_COLOR = "#5B9BD5"
SUCCESS_COLOR = "#3CB878"
SUCCESS_HOVER_COLOR = "#58D492"
SUCCESS_BG = "#1E3327"

PLACEHOLDER_VALUE = "\u2014"


class ProgressPanel(QWidget):
    """Live training progress display: epoch bar, metrics, loss chart, and log.

    Usage: call ``reset(total_epochs)`` right before starting a run, then
    ``update_progress(epoch, train_loss, val_loss, val_accuracy)`` once per
    completed epoch (e.g. connected to ``ModelTrainer.progress_updated``),
    and finally either ``show_completion(model_name)`` or ``show_stopped()``
    when the run ends.
    """

    #: Emitted when the user clicks "View in Model Library" on the completion banner.
    view_in_library_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._total_epochs = 0
        self._train_losses: List[float] = []
        self._val_losses: List[float] = []

        self._progress_bar: Optional[QProgressBar] = None
        self._epoch_label: Optional[QLabel] = None
        self._train_loss_value: Optional[QLabel] = None
        self._val_loss_value: Optional[QLabel] = None
        self._val_acc_value: Optional[QLabel] = None
        self._log_view: Optional[QTextEdit] = None
        self._figure: Optional[Figure] = None
        self._canvas: Optional[FigureCanvasQTAgg] = None
        self._axes = None
        self._completion_banner: Optional[QFrame] = None
        self._completion_label: Optional[QLabel] = None

        self._build_ui()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        layout.addLayout(self._build_epoch_row())
        layout.addLayout(self._build_metrics_row())
        layout.addWidget(self._build_chart())
        layout.addWidget(self._build_log())
        layout.addWidget(self._build_completion_banner())

    def _build_epoch_row(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setSpacing(6)

        self._epoch_label = QLabel("Epoch 0 / 0")
        self._epoch_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;")
        column.addWidget(self._epoch_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setStyleSheet(
            f"QProgressBar {{ background-color: {SURFACE_COLOR}; border-radius: 4px; border: none; }}"
            f"QProgressBar::chunk {{ background-color: {ACCENT_COLOR}; border-radius: 4px; }}"
        )
        column.addWidget(self._progress_bar)

        return column

    def _build_metrics_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)

        self._train_loss_value = self._add_metric_card(row, "Train Loss")
        self._val_loss_value = self._add_metric_card(row, "Val Loss")
        self._val_acc_value = self._add_metric_card(row, "Val Accuracy")

        return row

    def _add_metric_card(self, row: QHBoxLayout, title: str) -> QLabel:
        card = QFrame()
        card.setStyleSheet(f"QFrame {{ background-color: {SURFACE_COLOR}; border-radius: 8px; }}")

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 10, 14, 10)
        card_layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        card_layout.addWidget(title_label)

        value_label = QLabel(PLACEHOLDER_VALUE)
        value_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 18px; font-weight: 600; background: transparent;"
        )
        card_layout.addWidget(value_label)

        row.addWidget(card, 1)
        return value_label

    def _build_chart(self) -> FigureCanvasQTAgg:
        self._figure = Figure(figsize=(4, 2), dpi=100)
        self._figure.patch.set_facecolor(SURFACE_COLOR)
        self._axes = self._figure.add_subplot(111)
        self._style_axes()

        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setFixedHeight(180)
        self._canvas.setStyleSheet("background-color: transparent;")

        return self._canvas

    def _style_axes(self) -> None:
        axes = self._axes
        axes.clear()
        axes.set_facecolor(SURFACE_COLOR)
        axes.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        for spine in axes.spines.values():
            spine.set_color(BORDER_COLOR)
        axes.set_xlabel("Epoch", color=TEXT_SECONDARY, fontsize=8)
        axes.set_ylabel("Loss", color=TEXT_SECONDARY, fontsize=8)
        axes.grid(True, color=BORDER_COLOR, linewidth=0.5, alpha=0.6)

    def _build_log(self) -> QTextEdit:
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFixedHeight(140)
        self._log_view.setStyleSheet(
            f"QTextEdit {{ background-color: {SURFACE_COLOR}; color: {TEXT_PRIMARY};"
            f"border: 1px solid {BORDER_COLOR}; border-radius: 8px;"
            f"font-family: Menlo, Consolas, monospace; font-size: 11px; padding: 8px; }}"
        )
        return self._log_view

    def _build_completion_banner(self) -> QFrame:
        self._completion_banner = QFrame()
        self._completion_banner.setStyleSheet(
            f"QFrame {{ background-color: {SUCCESS_BG}; border: 1px solid {SUCCESS_COLOR}; border-radius: 8px; }}"
        )

        row = QHBoxLayout(self._completion_banner)
        row.setContentsMargins(14, 10, 14, 10)
        row.setSpacing(12)

        self._completion_label = QLabel("")
        self._completion_label.setWordWrap(True)
        self._completion_label.setStyleSheet(
            f"color: {SUCCESS_COLOR}; font-size: 12px; font-weight: 600; background: transparent;"
        )
        row.addWidget(self._completion_label, 1)

        view_button = QPushButton("View in Model Library")
        view_button.setObjectName("viewLibraryLink")
        view_button.setCursor(Qt.CursorShape.PointingHandCursor)
        view_button.setStyleSheet(
            f"QPushButton#viewLibraryLink {{ background: transparent; color: {SUCCESS_COLOR}; border: none;"
            f"text-decoration: underline; font-size: 12px; font-weight: 600; padding: 0px; }}"
            f"QPushButton#viewLibraryLink:hover {{ color: {SUCCESS_HOVER_COLOR}; }}"
        )
        view_button.clicked.connect(self.view_in_library_requested.emit)
        row.addWidget(view_button)

        self._completion_banner.setVisible(False)
        return self._completion_banner

    # -- Public API ----------------------------------------------------------

    def reset(self, total_epochs: int) -> None:
        """Clear all displayed progress and prepare the panel for a fresh run."""
        self._total_epochs = total_epochs
        self._train_losses = []
        self._val_losses = []

        self._progress_bar.setRange(0, max(total_epochs, 1))
        self._progress_bar.setValue(0)
        self._epoch_label.setText(f"Epoch 0 / {total_epochs}")

        for label in (self._train_loss_value, self._val_loss_value, self._val_acc_value):
            label.setText(PLACEHOLDER_VALUE)

        self._log_view.clear()
        self._style_axes()
        self._canvas.draw_idle()

        self._completion_banner.setVisible(False)
        self.append_log(f"Starting training for {total_epochs} epoch(s)\u2026")

    def update_progress(self, epoch: int, train_loss: float, val_loss: float, val_accuracy: float) -> None:
        """Update all displays with the results of a just-completed epoch."""
        self._epoch_label.setText(f"Epoch {epoch} / {self._total_epochs}")
        self._progress_bar.setValue(epoch)

        self._train_loss_value.setText(f"{train_loss:.4f}")
        self._val_loss_value.setText(f"{val_loss:.4f}")
        self._val_acc_value.setText(f"{val_accuracy * 100:.1f}%")

        self._train_losses.append(train_loss)
        self._val_losses.append(val_loss)
        self._redraw_chart()

        self.append_log(
            f"Epoch {epoch}/{self._total_epochs} \u2014 train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy * 100:.1f}%"
        )

    def append_log(self, message: str) -> None:
        """Append an arbitrary timestamped line to the log (e.g. status changes/errors)."""
        if self._log_view is None:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log_view.append(f"[{timestamp}] {message}")

    def show_completion(self, model_name: str, saved_at: Optional[datetime] = None) -> None:
        """Show the green "training complete" banner for a successfully saved model."""
        saved_at = saved_at or datetime.now()
        date_str = saved_at.strftime("%Y-%m-%d %H:%M")
        self._completion_label.setText(f'Training complete \u2014 model saved as "{model_name}" ({date_str})')
        self._completion_banner.setVisible(True)
        self.append_log(f'Training complete. Model saved as "{model_name}".')

    def show_stopped(self) -> None:
        """Log that training was stopped early by the user (no model was saved)."""
        self.append_log("Training stopped by user.")

    # -- Internal helpers --------------------------------------------------

    def _redraw_chart(self) -> None:
        self._style_axes()
        epochs = list(range(1, len(self._train_losses) + 1))
        self._axes.plot(epochs, self._train_losses, color=ACCENT_COLOR, linewidth=1.6, label="Train Loss")
        self._axes.plot(epochs, self._val_losses, color=VAL_LINE_COLOR, linewidth=1.6, label="Val Loss")
        if epochs:
            legend = self._axes.legend(
                loc="upper right", fontsize=7, facecolor=SURFACE_COLOR, edgecolor=BORDER_COLOR
            )
            for text in legend.get_texts():
                text.set_color(TEXT_SECONDARY)
        self._figure.tight_layout()
        self._canvas.draw_idle()
