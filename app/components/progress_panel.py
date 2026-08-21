"""
Training progress panel.

Live feedback for a training run: epoch bar, patience note, metric cards,
sum-deviation check, loss/MAE charts, timestamped log, and a done / early-stop
banner. Used by TrainPage; no dependency on MainWindow.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

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

from app.components.charts import style_axes
from app.theme import (
    ACCENT_COLOR,
    BITUMEN_LINE_COLOR,
    BORDER_COLOR,
    SOLIDS_LINE_COLOR,
    SUCCESS_BG,
    SUCCESS_COLOR,
    SUCCESS_HOVER_COLOR,
    SURFACE_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    VAL_LINE_COLOR,
    WARNING_BG,
    WATER_LINE_COLOR,
    sum_deviation_color,
)

PLACEHOLDER_VALUE = "\u2014"


class ProgressPanel(QWidget):
    """Live training progress display.

    Call ``reset`` before a run, ``update_progress`` each epoch, optionally
    ``note_early_stopped``, then ``show_completion`` or ``show_early_stopped_banner``.
    """

    #: Fired when the user clicks "View in Model Library" on the done banner.
    view_in_library_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._total_epochs = 0
        self._patience = 0
        self._best_val_loss = float("inf")
        self._patience_counter = 0

        self._train_losses: List[float] = []
        self._val_losses: List[float] = []
        self._water_maes: List[float] = []
        self._solids_maes: List[float] = []
        self._bitumen_maes: List[float] = []

        self._epoch_label: Optional[QLabel] = None
        self._progress_bar: Optional[QProgressBar] = None
        self._patience_label: Optional[QLabel] = None
        self._metric_values: Dict[str, QLabel] = {}
        self._sum_dev_value: Optional[QLabel] = None

        self._log_view: Optional[QTextEdit] = None

        self._loss_figure: Optional[Figure] = None
        self._loss_canvas: Optional[FigureCanvasQTAgg] = None
        self._loss_axes = None
        self._mae_figure: Optional[Figure] = None
        self._mae_canvas: Optional[FigureCanvasQTAgg] = None
        self._mae_axes = None

        self._completion_banner: Optional[QFrame] = None
        self._completion_label: Optional[QLabel] = None
        self._early_stop_banner: Optional[QFrame] = None
        self._early_stop_label: Optional[QLabel] = None

        self._build_ui()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        layout.addLayout(self._build_epoch_row())

        self._patience_label = QLabel("")
        self._patience_label.setWordWrap(True)
        self._patience_label.setStyleSheet(
            f"color: {ACCENT_COLOR}; font-size: 11px; font-weight: 600; background: transparent;"
        )
        self._patience_label.setVisible(False)
        layout.addWidget(self._patience_label)

        layout.addLayout(self._build_metrics_row())
        layout.addWidget(self._build_sum_deviation_row())
        layout.addLayout(self._build_charts_row())
        layout.addWidget(self._build_log())
        layout.addWidget(self._build_completion_banner())
        layout.addWidget(self._build_early_stop_banner())

    def _build_epoch_row(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setSpacing(6)

        self._epoch_label = QLabel("Epoch 0 / 0")
        self._epoch_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600; background: transparent;"
        )
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
        row.setSpacing(8)
        for key in ("Train Loss", "Val Loss", "Water MAE", "Solids MAE", "Bitumen MAE"):
            self._metric_values[key] = self._add_metric_card(row, key)
        return row

    def _add_metric_card(self, row: QHBoxLayout, title: str) -> QLabel:
        card = QFrame()
        card.setStyleSheet(f"QFrame {{ background-color: {SURFACE_COLOR}; border-radius: 8px; }}")

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)
        card_layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; background: transparent;")
        card_layout.addWidget(title_label)

        value_label = QLabel(PLACEHOLDER_VALUE)
        value_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 15px; font-weight: 600; background: transparent;"
        )
        card_layout.addWidget(value_label)

        row.addWidget(card, 1)
        return value_label

    def _build_sum_deviation_row(self) -> QFrame:
        row_frame = QFrame()
        row_frame.setStyleSheet(f"QFrame {{ background-color: {SURFACE_COLOR}; border-radius: 8px; }}")
        row_frame.setToolTip("Predictions should sum to ~100%. High values mean the model is struggling.")

        row = QHBoxLayout(row_frame)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)

        label = QLabel("Sum deviation:")
        label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; background: transparent;")
        row.addWidget(label)

        self._sum_dev_value = QLabel(PLACEHOLDER_VALUE)
        self._sum_dev_value.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 700; background: transparent;"
        )
        row.addWidget(self._sum_dev_value)
        row.addStretch(1)

        return row_frame

    def _build_charts_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)

        self._loss_figure = Figure(figsize=(4, 2.2), dpi=100)
        self._loss_figure.patch.set_facecolor(SURFACE_COLOR)
        self._loss_axes = self._loss_figure.add_subplot(111)
        style_axes(self._loss_axes, "Loss", facecolor=SURFACE_COLOR, grid_alpha=0.6, clear=True)
        self._loss_canvas = FigureCanvasQTAgg(self._loss_figure)
        self._loss_canvas.setFixedHeight(190)
        self._loss_canvas.setStyleSheet("background-color: transparent;")
        row.addWidget(self._loss_canvas, 1)

        self._mae_figure = Figure(figsize=(4, 2.2), dpi=100)
        self._mae_figure.patch.set_facecolor(SURFACE_COLOR)
        self._mae_axes = self._mae_figure.add_subplot(111)
        style_axes(self._mae_axes, "MAE (%)", facecolor=SURFACE_COLOR, grid_alpha=0.6, clear=True)
        self._mae_canvas = FigureCanvasQTAgg(self._mae_figure)
        self._mae_canvas.setFixedHeight(190)
        self._mae_canvas.setStyleSheet("background-color: transparent;")
        row.addWidget(self._mae_canvas, 1)

        return row

    def _build_log(self) -> QTextEdit:
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFixedHeight(130)
        self._log_view.setStyleSheet(
            f"QTextEdit {{ background-color: {SURFACE_COLOR}; color: {TEXT_PRIMARY};"
            f"border: 1px solid {BORDER_COLOR}; border-radius: 8px;"
            f"font-family: Menlo, Consolas, monospace; font-size: 11px; padding: 8px; }}"
        )
        return self._log_view

    def _build_completion_banner(self) -> QFrame:
        self._completion_banner = QFrame()
        self._completion_banner.setObjectName("completionBanner")
        self._completion_banner.setStyleSheet(
            # Scoped to #completionBanner -- QLabel is a QFrame subclass in
            # Qt, so a bare "QFrame" selector would also draw this border
            # around the nested message label, not just the banner.
            f"QFrame#completionBanner {{ background-color: {SUCCESS_BG}; border: 1px solid {SUCCESS_COLOR};"
            f"border-radius: 8px; }}"
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

        view_button = QPushButton("View in Model Library \u2192")
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

    def _build_early_stop_banner(self) -> QFrame:
        self._early_stop_banner = QFrame()
        self._early_stop_banner.setObjectName("earlyStopBanner")
        self._early_stop_banner.setStyleSheet(
            # Scoped to #earlyStopBanner -- QLabel is a QFrame subclass in
            # Qt, so a bare "QFrame" selector would also draw this border
            # around the nested message label, not just the banner.
            f"QFrame#earlyStopBanner {{ background-color: {WARNING_BG}; border: 1px solid {ACCENT_COLOR};"
            f"border-radius: 8px; }}"
        )

        row = QHBoxLayout(self._early_stop_banner)
        row.setContentsMargins(14, 10, 14, 10)
        row.setSpacing(12)

        self._early_stop_label = QLabel("")
        self._early_stop_label.setWordWrap(True)
        self._early_stop_label.setStyleSheet(
            f"color: {ACCENT_COLOR}; font-size: 12px; font-weight: 600; background: transparent;"
        )
        row.addWidget(self._early_stop_label, 1)

        self._early_stop_banner.setVisible(False)
        return self._early_stop_banner

    # -- Public API ----------------------------------------------------------

    def reset(self, total_epochs: int, patience: int) -> None:
        """Clear the panel and ready it for a new run."""
        self._total_epochs = total_epochs
        self._patience = patience
        self._best_val_loss = float("inf")
        self._patience_counter = 0

        self._train_losses = []
        self._val_losses = []
        self._water_maes = []
        self._solids_maes = []
        self._bitumen_maes = []

        self._progress_bar.setRange(0, max(total_epochs, 1))
        self._progress_bar.setValue(0)
        self._epoch_label.setText(f"Epoch 0 / {total_epochs}")
        self._patience_label.setVisible(False)

        for value_label in self._metric_values.values():
            value_label.setText(PLACEHOLDER_VALUE)
        self._sum_dev_value.setText(PLACEHOLDER_VALUE)
        self._sum_dev_value.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 700; background: transparent;"
        )

        self._log_view.clear()
        style_axes(self._loss_axes, "Loss", facecolor=SURFACE_COLOR, grid_alpha=0.6, clear=True)
        self._loss_canvas.draw_idle()
        style_axes(self._mae_axes, "MAE (%)", facecolor=SURFACE_COLOR, grid_alpha=0.6, clear=True)
        self._mae_canvas.draw_idle()

        self._completion_banner.setVisible(False)
        self._early_stop_banner.setVisible(False)

        self.append_log(f"Starting training \u2014 {total_epochs} epoch(s)\u2026")

    def update_progress(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_mae_dict: Dict[str, float],
        val_sum_deviation: float,
    ) -> None:
        """Refresh displays after one finished epoch."""
        self._epoch_label.setText(f"Epoch {epoch} / {self._total_epochs}")
        self._progress_bar.setValue(epoch)

        # Same best-val / patience logic as RegressionTrainer, so the label matches.
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._patience_counter = 0
        else:
            self._patience_counter += 1

        if self._patience_counter > 0:
            plural = "epoch" if self._patience_counter == 1 else "epochs"
            self._patience_label.setText(
                f"No improvement for {self._patience_counter} {plural} (patience: {self._patience})"
            )
            self._patience_label.setVisible(True)
        else:
            self._patience_label.setVisible(False)

        water = val_mae_dict.get("Water", 0.0)
        solids = val_mae_dict.get("Solids", 0.0)
        bitumen = val_mae_dict.get("Bitumen", 0.0)

        self._metric_values["Train Loss"].setText(f"{train_loss:.4f}")
        self._metric_values["Val Loss"].setText(f"{val_loss:.4f}")
        self._metric_values["Water MAE"].setText(f"\u00b1{water:.2f}%")
        self._metric_values["Solids MAE"].setText(f"\u00b1{solids:.2f}%")
        self._metric_values["Bitumen MAE"].setText(f"\u00b1{bitumen:.2f}%")

        self._sum_dev_value.setText(f"\u00b1{val_sum_deviation:.2f}%")
        self._sum_dev_value.setStyleSheet(
            f"color: {sum_deviation_color(val_sum_deviation)}; font-size: 12px; font-weight: 700; background: transparent;"
        )

        self._train_losses.append(train_loss)
        self._val_losses.append(val_loss)
        self._water_maes.append(water)
        self._solids_maes.append(solids)
        self._bitumen_maes.append(bitumen)
        self._redraw_loss_chart()
        self._redraw_mae_chart()

        self.append_log(
            f"Epoch {epoch}/{self._total_epochs} \u2014 Loss: {train_loss:.4f} | Val: {val_loss:.4f} "
            f"| Water: \u00b1{water:.2f}%  Solids: \u00b1{solids:.2f}%  Bitumen: \u00b1{bitumen:.2f}% "
            f"| Sum dev: \u00b1{val_sum_deviation:.2f}%"
        )

    def note_early_stopped(self, epoch: int) -> None:
        """Log that patience ran out (before ``finished``)."""
        self.append_log(f"Early stop at epoch {epoch} (patience used up).")

    def append_log(self, message: str) -> None:
        """Append a timestamped log line."""
        if self._log_view is None:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log_view.append(f"[{timestamp}] {message}")

    def show_completion(
        self,
        model_name: str,
        best_val_mae: Dict[str, float],
        test_mae: Optional[Dict[str, float]] = None,
    ) -> None:
        """Green "training complete" banner after a successful save."""
        self._early_stop_banner.setVisible(False)
        text = (
            f'Training complete \u2014 "{model_name}" saved.\n'
            f"Val: Water \u00b1{best_val_mae.get('Water', 0.0):.2f}%  "
            f"Solids \u00b1{best_val_mae.get('Solids', 0.0):.2f}%  "
            f"Bitumen \u00b1{best_val_mae.get('Bitumen', 0.0):.2f}%"
        )
        if test_mae:
            text += (
                f"\nTest: Water \u00b1{test_mae.get('Water', 0.0):.2f}%  "
                f"Solids \u00b1{test_mae.get('Solids', 0.0):.2f}%  "
                f"Bitumen \u00b1{test_mae.get('Bitumen', 0.0):.2f}%"
            )
        self._completion_label.setText(text)
        self._completion_banner.setVisible(True)
        self.append_log(f'Training complete. Saved as "{model_name}".')
        if test_mae:
            self.append_log(
                f"Test MAE \u2014 Water \u00b1{test_mae.get('Water', 0.0):.2f}%  "
                f"Solids \u00b1{test_mae.get('Solids', 0.0):.2f}%  "
                f"Bitumen \u00b1{test_mae.get('Bitumen', 0.0):.2f}%"
            )

    def show_early_stopped_banner(
        self,
        epoch: int,
        best_val_mae: Dict[str, float],
        test_mae: Optional[Dict[str, float]] = None,
    ) -> None:
        """Amber early-stop banner instead of the green one."""
        self._completion_banner.setVisible(False)
        text = (
            f"Early stop at epoch {epoch}.\n"
            f"Val: Water \u00b1{best_val_mae.get('Water', 0.0):.2f}%  "
            f"Solids \u00b1{best_val_mae.get('Solids', 0.0):.2f}%  "
            f"Bitumen \u00b1{best_val_mae.get('Bitumen', 0.0):.2f}%"
        )
        if test_mae:
            text += (
                f"\nTest: Water \u00b1{test_mae.get('Water', 0.0):.2f}%  "
                f"Solids \u00b1{test_mae.get('Solids', 0.0):.2f}%  "
                f"Bitumen \u00b1{test_mae.get('Bitumen', 0.0):.2f}%"
            )
        self._early_stop_label.setText(text)
        self._early_stop_banner.setVisible(True)
        if test_mae:
            self.append_log(
                f"Test MAE \u2014 Water \u00b1{test_mae.get('Water', 0.0):.2f}%  "
                f"Solids \u00b1{test_mae.get('Solids', 0.0):.2f}%  "
                f"Bitumen \u00b1{test_mae.get('Bitumen', 0.0):.2f}%"
            )

    # -- Internal helpers --------------------------------------------------

    def _redraw_loss_chart(self) -> None:
        style_axes(self._loss_axes, "Loss", facecolor=SURFACE_COLOR, grid_alpha=0.6, clear=True)
        epochs = list(range(1, len(self._train_losses) + 1))
        self._loss_axes.plot(epochs, self._train_losses, color=ACCENT_COLOR, linewidth=1.6, label="Train")
        self._loss_axes.plot(epochs, self._val_losses, color=VAL_LINE_COLOR, linewidth=1.6, label="Val")
        if epochs:
            legend = self._loss_axes.legend(
                loc="upper right", fontsize=7, facecolor=SURFACE_COLOR, edgecolor=BORDER_COLOR
            )
            for text in legend.get_texts():
                text.set_color(TEXT_SECONDARY)
        self._loss_figure.tight_layout()
        self._loss_canvas.draw_idle()

    def _redraw_mae_chart(self) -> None:
        style_axes(self._mae_axes, "MAE (%)", facecolor=SURFACE_COLOR, grid_alpha=0.6, clear=True)
        epochs = list(range(1, len(self._water_maes) + 1))
        self._mae_axes.plot(epochs, self._water_maes, color=WATER_LINE_COLOR, linewidth=1.6, label="Water")
        self._mae_axes.plot(epochs, self._solids_maes, color=SOLIDS_LINE_COLOR, linewidth=1.6, label="Solids")
        self._mae_axes.plot(epochs, self._bitumen_maes, color=BITUMEN_LINE_COLOR, linewidth=1.6, label="Bitumen")
        if epochs:
            legend = self._mae_axes.legend(
                loc="upper right", fontsize=7, facecolor=SURFACE_COLOR, edgecolor=BORDER_COLOR
            )
            for text in legend.get_texts():
                text.set_color(TEXT_SECONDARY)
        self._mae_figure.tight_layout()
        self._mae_canvas.draw_idle()
