"""Live training progress: epoch bar, metric cards, loss and R squared charts, a timestamped log, and a done or early-stop banner. TrainPage owns this. It does not talk to the main window."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.components.charts import style_axes
from app.theme import (
    ACCENT_COLOR,
    BITUMEN_LINE_COLOR,
    BORDER_COLOR,
    LABEL_RESET_QSS,
    SOLIDS_LINE_COLOR,
    SUCCESS_BG,
    SUCCESS_COLOR,
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
    """The live training panel. Call reset before a run, update_progress each epoch, then show_completion or show_early_stopped_banner when it finishes."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._total_epochs = 0
        self._patience = 0
        self._best_val_loss = float("inf")
        self._patience_counter = 0

        self._train_losses: List[float] = []
        self._val_losses: List[float] = []
        self._water_r2s: List[float] = []
        self._solids_r2s: List[float] = []
        self._bitumen_r2s: List[float] = []

        self._epoch_label: Optional[QLabel] = None
        self._progress_bar: Optional[QProgressBar] = None
        self._patience_label: Optional[QLabel] = None
        self._metric_values: Dict[str, QLabel] = {}
        self._sum_dev_value: Optional[QLabel] = None

        self._log_view: Optional[QTextEdit] = None

        self._loss_figure: Optional[Figure] = None
        self._loss_canvas: Optional[FigureCanvasQTAgg] = None
        self._loss_axes = None
        self._r2_figure: Optional[Figure] = None
        self._r2_canvas: Optional[FigureCanvasQTAgg] = None
        self._r2_axes = None

        self._completion_banner: Optional[QFrame] = None
        self._completion_label: Optional[QLabel] = None
        self._early_stop_banner: Optional[QFrame] = None
        self._early_stop_label: Optional[QLabel] = None

        self._build_ui()

    # Build the widgets and lay them out.

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        layout.addLayout(self._build_epoch_row())

        self._patience_label = QLabel("")
        self._patience_label.setWordWrap(True)
        self._patience_label.setStyleSheet(
            f"color: {ACCENT_COLOR}; font-size: 11px; font-weight: 600;"
            f" background: transparent;"
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
            f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;"
            f" background: transparent;"
        )
        column.addWidget(self._epoch_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setStyleSheet(
            f"QProgressBar {{ background-color: {SURFACE_COLOR};"
            f" border-radius: 4px; border: none; }}"
            f"QProgressBar::chunk {{ background-color: {ACCENT_COLOR};"
            f" border-radius: 4px; }}"
        )
        column.addWidget(self._progress_bar)

        return column

    def _build_metrics_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        for key in (
            "Train Loss",
            "Val Loss",
            "Bitumen R²",
            "Solids R²",
            "Water R²",
        ):
            self._metric_values[key] = self._add_metric_card(row, key)
        return row

    def _add_metric_card(self, row: QHBoxLayout, title: str) -> QLabel:
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background-color: {SURFACE_COLOR};"
            f" border: 1px solid {BORDER_COLOR}; border-radius: 3px; }}"
            f"{LABEL_RESET_QSS}"
        )

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)
        card_layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setWordWrap(True)
        title_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        title_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 10px;"
            f" background: transparent;"
        )
        card_layout.addWidget(title_label)

        value_label = QLabel(PLACEHOLDER_VALUE)
        value_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        value_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 15px; font-weight: 600;"
            f" background: transparent;"
        )
        card_layout.addWidget(value_label)

        row.addWidget(card, 1)
        return value_label

    def _build_sum_deviation_row(self) -> QFrame:
        row_frame = QFrame()
        row_frame.setStyleSheet(
            f"QFrame {{ background-color: {SURFACE_COLOR};"
            f" border: 1px solid {BORDER_COLOR}; border-radius: 3px; }}"
            f"{LABEL_RESET_QSS}"
        )
        row_frame.setToolTip(
            "Predictions should sum to ~100%. High values mean the model is "
            "struggling."
        )

        row = QHBoxLayout(row_frame)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)

        label = QLabel("Sum deviation:")
        label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 12px;"
            f" background: transparent;"
        )
        row.addWidget(label)

        self._sum_dev_value = QLabel(PLACEHOLDER_VALUE)
        self._sum_dev_value.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 700;"
            f" background: transparent;"
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
        style_axes(
            self._loss_axes,
            "Loss",
            facecolor=SURFACE_COLOR,
            grid_alpha=0.6,
            clear=True,
        )
        self._loss_canvas = FigureCanvasQTAgg(self._loss_figure)
        self._loss_canvas.setFixedHeight(190)
        self._loss_canvas.setStyleSheet("background-color: transparent;")
        row.addWidget(self._loss_canvas, 1)

        self._r2_figure = Figure(figsize=(4, 2.2), dpi=100)
        self._r2_figure.patch.set_facecolor(SURFACE_COLOR)
        self._r2_axes = self._r2_figure.add_subplot(111)
        style_axes(
            self._r2_axes,
            "R²",
            facecolor=SURFACE_COLOR,
            grid_alpha=0.6,
            clear=True,
        )
        self._r2_canvas = FigureCanvasQTAgg(self._r2_figure)
        self._r2_canvas.setFixedHeight(190)
        self._r2_canvas.setStyleSheet("background-color: transparent;")
        row.addWidget(self._r2_canvas, 1)

        return row

    def _build_log(self) -> QTextEdit:
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFixedHeight(130)
        self._log_view.setStyleSheet(
            f"QTextEdit {{ background-color: {SURFACE_COLOR};"
            f" color: {TEXT_PRIMARY};"
            f"border: 1px solid {BORDER_COLOR}; border-radius: 3px;"
            f"font-family: Menlo, Consolas, monospace; font-size: 11px;"
            f" padding: 8px; }}"
        )
        return self._log_view

    def _build_completion_banner(self) -> QFrame:
        self._completion_banner = QFrame()
        self._completion_banner.setObjectName("completionBanner")
        self._completion_banner.setStyleSheet(
            # Target the banner by id. QLabel is a QFrame in Qt, so a bare QFrame rule would also box the message inside.
            f"QFrame#completionBanner {{ background-color: {SUCCESS_BG};"
            f" border: 1px solid {SUCCESS_COLOR};"
            f"border-radius: 3px; }}"
        )

        row = QHBoxLayout(self._completion_banner)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)

        self._completion_label = QLabel("")
        self._completion_label.setWordWrap(True)
        self._completion_label.setStyleSheet(
            f"color: {SUCCESS_COLOR}; font-size: 12px;"
            f" background: transparent;"
        )
        row.addWidget(self._completion_label, 1)

        self._completion_banner.setVisible(False)
        return self._completion_banner

    def _build_early_stop_banner(self) -> QFrame:
        self._early_stop_banner = QFrame()
        self._early_stop_banner.setObjectName("earlyStopBanner")
        self._early_stop_banner.setStyleSheet(
            # Target the banner by id. QLabel is a QFrame in Qt, so a bare QFrame rule would also box the message inside.
            f"QFrame#earlyStopBanner {{ background-color: {WARNING_BG};"
            f" border: 1px solid {ACCENT_COLOR};"
            f"border-radius: 3px; }}"
        )

        row = QHBoxLayout(self._early_stop_banner)
        row.setContentsMargins(14, 10, 14, 10)
        row.setSpacing(12)

        self._early_stop_label = QLabel("")
        self._early_stop_label.setWordWrap(True)
        self._early_stop_label.setStyleSheet(
            f"color: {ACCENT_COLOR}; font-size: 12px; font-weight: 600;"
            f" background: transparent;"
        )
        row.addWidget(self._early_stop_label, 1)

        self._early_stop_banner.setVisible(False)
        return self._early_stop_banner

    # Methods TrainPage calls during a run.

    def reset(self, total_epochs: int, patience: int) -> None:
        """Clears the charts and log and sets the epoch bar for a new run. Pass how many epochs and the patience value."""
        self._total_epochs = total_epochs
        self._patience = patience
        self._best_val_loss = float("inf")
        self._patience_counter = 0

        self._train_losses = []
        self._val_losses = []
        self._water_r2s = []
        self._solids_r2s = []
        self._bitumen_r2s = []

        self._progress_bar.setRange(0, max(total_epochs, 1))
        self._progress_bar.setValue(0)
        self._epoch_label.setText(f"Epoch 0 / {total_epochs}")
        self._patience_label.setVisible(False)

        for value_label in self._metric_values.values():
            value_label.setText(PLACEHOLDER_VALUE)
        self._sum_dev_value.setText(PLACEHOLDER_VALUE)
        self._sum_dev_value.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 700;"
            f" background: transparent;"
        )

        self._log_view.clear()
        style_axes(
            self._loss_axes,
            "Loss",
            facecolor=SURFACE_COLOR,
            grid_alpha=0.6,
            clear=True,
        )
        self._loss_canvas.draw_idle()
        style_axes(
            self._r2_axes,
            "R²",
            facecolor=SURFACE_COLOR,
            grid_alpha=0.6,
            clear=True,
        )
        self._r2_canvas.draw_idle()

        self._completion_banner.setVisible(False)
        self._early_stop_banner.setVisible(False)

        self.append_log(
            f"Starting training \u2014 {total_epochs} epoch(s)\u2026"
        )

    def update_progress(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_mae_dict: Dict[str, float],
        val_sum_deviation: float,
        val_r2_dict: Optional[Dict[str, float]] = None,
    ) -> None:
        """Redraws the metrics, charts, and log after one epoch. Pass the epoch number, losses, MAE, sum deviation, and R squared."""
        self._epoch_label.setText(f"Epoch {epoch} / {self._total_epochs}")
        self._progress_bar.setValue(epoch)

        # Same best-val and patience counting as the trainer, so the label matches what the loop is doing.
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._patience_counter = 0
        else:
            self._patience_counter += 1

        if self._patience > 0 and self._patience_counter > 0:
            plural = "epoch" if self._patience_counter == 1 else "epochs"
            self._patience_label.setText(
                f"No improvement for {self._patience_counter} {plural} "
                f"(patience: {self._patience})"
            )
            self._patience_label.setVisible(True)
        else:
            self._patience_label.setVisible(False)

        water = val_mae_dict.get("Water", 0.0)
        solids = val_mae_dict.get("Solids", 0.0)
        bitumen = val_mae_dict.get("Bitumen", 0.0)
        r2 = val_r2_dict or {}
        water_r2 = r2.get("Water", 0.0)
        solids_r2 = r2.get("Solids", 0.0)
        bitumen_r2 = r2.get("Bitumen", 0.0)

        self._metric_values["Train Loss"].setText(f"{train_loss:.4f}")
        self._metric_values["Val Loss"].setText(f"{val_loss:.4f}")
        self._metric_values["Bitumen R²"].setText(f"{bitumen_r2:.3f}")
        self._metric_values["Solids R²"].setText(f"{solids_r2:.3f}")
        self._metric_values["Water R²"].setText(f"{water_r2:.3f}")

        self._sum_dev_value.setText(f"\u00b1{val_sum_deviation:.2f}%")
        self._sum_dev_value.setStyleSheet(
            f"color: {sum_deviation_color(val_sum_deviation)};"
            f" font-size: 12px; font-weight: 700; background: transparent;"
        )

        self._train_losses.append(train_loss)
        self._val_losses.append(val_loss)
        self._water_r2s.append(water_r2)
        self._solids_r2s.append(solids_r2)
        self._bitumen_r2s.append(bitumen_r2)
        self._redraw_loss_chart()
        self._redraw_r2_chart()

        self.append_log(
            f"Epoch {epoch}/{self._total_epochs} \u2014 Loss: "
            f"{train_loss:.4f} | Val: {val_loss:.4f} "
            f"| R² Bitumen: {bitumen_r2:.3f}  Solids: {solids_r2:.3f}  Water: "
            f"{water_r2:.3f} "
            f"| MAE W \u00b1{water:.2f}% S \u00b1{solids:.2f}% B "
            f"\u00b1{bitumen:.2f}% "
            f"| Sum dev: \u00b1{val_sum_deviation:.2f}%"
        )

    def note_early_stopped(self, epoch: int) -> None:
        """Writes an early-stop line to the log. Call this when patience ran out, before the finished signal lands."""
        self.append_log(f"Early stop at epoch {epoch} (patience used up).")

    def append_log(self, message: str) -> None:
        """Adds a timestamped line to the log. Pass the message text."""
        if self._log_view is None:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log_view.append(f"[{timestamp}] {message}")

    def show_completion(
        self,
        model_name: str,
        best_val_mae: Dict[str, float],
        test_mae: Optional[Dict[str, float]] = None,
        best_val_r2: Optional[Dict[str, float]] = None,
        test_r2: Optional[Dict[str, float]] = None,
    ) -> None:
        """Shows the green training-complete banner after a successful save. Pass the model name and the val and test scores."""
        self._early_stop_banner.setVisible(False)
        text = (
            f'Training complete \u2014 "{model_name}" saved.\n'
            f"{self._format_r2_mae_line('Val', best_val_r2, best_val_mae)}"
        )
        if test_mae or test_r2:
            text += f"\n{self._format_r2_mae_line('Test', test_r2, test_mae)}"
        self._completion_label.setText(text)
        self._completion_banner.setVisible(True)
        self.append_log(f'Training complete. Saved as "{model_name}".')
        if test_mae or test_r2:
            self.append_log(
                self._format_r2_mae_line("Test", test_r2, test_mae)
            )

    def show_early_stopped_banner(
        self,
        epoch: int,
        best_val_mae: Dict[str, float],
        test_mae: Optional[Dict[str, float]] = None,
        best_val_r2: Optional[Dict[str, float]] = None,
        test_r2: Optional[Dict[str, float]] = None,
    ) -> None:
        """Shows the amber early-stop banner instead of the green one. Pass the epoch we stopped on and the val and test scores."""
        self._completion_banner.setVisible(False)
        text = (
            f"Early stop at epoch {epoch}.\n"
            f"{self._format_r2_mae_line('Val', best_val_r2, best_val_mae)}"
        )
        if test_mae or test_r2:
            text += f"\n{self._format_r2_mae_line('Test', test_r2, test_mae)}"
        self._early_stop_label.setText(text)
        self._early_stop_banner.setVisible(True)
        if test_mae or test_r2:
            self.append_log(
                self._format_r2_mae_line("Test", test_r2, test_mae)
            )

    @staticmethod
    def _format_r2_mae_line(
        split: str,
        r2: Optional[Dict[str, float]],
        mae: Optional[Dict[str, float]],
    ) -> str:
        r2 = r2 or {}
        mae = mae or {}
        return (
            f"{split}: R² Bitumen {r2.get('Bitumen', 0.0):.3f}  Solids "
            f"{r2.get('Solids', 0.0):.3f}  "
            f"Water {r2.get('Water', 0.0):.3f}  |  MAE B "
            f"\u00b1{mae.get('Bitumen', 0.0):.2f}%  "
            f"S \u00b1{mae.get('Solids', 0.0):.2f}%  W "
            f"\u00b1{mae.get('Water', 0.0):.2f}%"
        )

    # Chart drawing helpers.

    def _redraw_loss_chart(self) -> None:
        style_axes(
            self._loss_axes,
            "Loss",
            facecolor=SURFACE_COLOR,
            grid_alpha=0.6,
            clear=True,
        )
        epochs = list(range(1, len(self._train_losses) + 1))
        self._loss_axes.plot(
            epochs,
            self._train_losses,
            color=ACCENT_COLOR,
            linewidth=1.6,
            label="Train",
        )
        self._loss_axes.plot(
            epochs,
            self._val_losses,
            color=VAL_LINE_COLOR,
            linewidth=1.6,
            label="Val",
        )
        if epochs:
            legend = self._loss_axes.legend(
                loc="upper right",
                fontsize=7,
                facecolor=SURFACE_COLOR,
                edgecolor=BORDER_COLOR,
            )
            for text in legend.get_texts():
                text.set_color(TEXT_SECONDARY)
        self._loss_figure.tight_layout()
        self._loss_canvas.draw_idle()

    def _redraw_r2_chart(self) -> None:
        style_axes(
            self._r2_axes,
            "R²",
            facecolor=SURFACE_COLOR,
            grid_alpha=0.6,
            clear=True,
        )
        epochs = list(range(1, len(self._bitumen_r2s) + 1))
        self._r2_axes.plot(
            epochs,
            self._water_r2s,
            color=WATER_LINE_COLOR,
            linewidth=1.6,
            label="Water",
        )
        self._r2_axes.plot(
            epochs,
            self._solids_r2s,
            color=SOLIDS_LINE_COLOR,
            linewidth=1.6,
            label="Solids",
        )
        self._r2_axes.plot(
            epochs,
            self._bitumen_r2s,
            color=BITUMEN_LINE_COLOR,
            linewidth=1.6,
            label="Bitumen",
        )
        if epochs:
            legend = self._r2_axes.legend(
                loc="lower right",
                fontsize=7,
                facecolor=SURFACE_COLOR,
                edgecolor=BORDER_COLOR,
            )
            for text in legend.get_texts():
                text.set_color(TEXT_SECONDARY)
        self._r2_figure.tight_layout()
        self._r2_canvas.draw_idle()
