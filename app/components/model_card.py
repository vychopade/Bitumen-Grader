"""
Model card.

One saved model: name, date, architecture, R², plus load / retrain /
details / delete. Details expands R²/loss curves and output stats.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PyQt6.QtCore import Qt, pyqtSignal  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.components.charts import style_axes
from app.constants import OUTPUT_NAMES
from app.theme import (
    ACCENT_COLOR,
    BACKGROUND_COLOR,
    BITUMEN_LINE_COLOR,
    BORDER_COLOR,
    DANGER_COLOR,
    SOLIDS_LINE_COLOR,
    SURFACE_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    VAL_LINE_COLOR,
    WATER_LINE_COLOR,
    accent_button_qss,
    ghost_button_qss,
    link_button_qss,
)
from app.utils.model_io import format_created_at, mean_r2, model_r2_splits, resolve_model_r2


class ModelCard(QFrame):
    """Summary card for one saved model, with load / details / delete.

    ``metadata`` matches ``list_saved_models`` (JSON fields plus
    ``model_path`` / ``metadata_path``). The card never touches disk; it
    only emits ``load_requested`` / ``delete_requested``.
    """

    load_requested = pyqtSignal(dict)
    delete_requested = pyqtSignal(dict)
    retrain_requested = pyqtSignal(dict)

    def __init__(self, metadata: Dict[str, Any], is_active: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("modelCard")
        self.metadata = metadata
        self._is_active = is_active
        self._details_expanded = False

        self._active_badge: Optional[QLabel] = None
        self._load_button: Optional[QPushButton] = None
        self._retrain_button: Optional[QPushButton] = None
        self._details_button: Optional[QPushButton] = None
        self._delete_button: Optional[QPushButton] = None
        self._details_section: Optional[QWidget] = None

        self._build_ui()
        self.set_active(is_active)

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        layout.addLayout(self._build_top_row())
        layout.addLayout(self._build_pills_row())
        layout.addWidget(self._build_accuracy_block())
        layout.addLayout(self._build_action_row())

        self._details_section = self._build_details_section()
        self._details_section.setVisible(False)
        layout.addWidget(self._details_section)

    def _build_top_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)

        name_col = QVBoxLayout()
        name_col.setSpacing(2)

        name_label = QLabel(self.metadata.get("name") or "Untitled Model")
        name_label.setWordWrap(True)
        name_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; background: transparent;")
        name_col.addWidget(name_label)

        date_label = QLabel(format_created_at(self.metadata.get("created_at"), fmt="Created %b %d %Y at %H:%M"))
        date_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        name_col.addWidget(date_label)

        row.addLayout(name_col, 1)

        self._active_badge = QLabel("in use")
        self._active_badge.setStyleSheet(
            f"color: {ACCENT_COLOR}; font-size: 11px; background: transparent; padding: 0;"
        )
        row.addWidget(self._active_badge, 0, Qt.AlignmentFlag.AlignTop)

        return row

    def _build_pills_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)

        architecture = self.metadata.get("architecture")
        labels = {
            "baseline": "Baseline CNN",
            "resnet50": "ResNet50",
            "vgg16": "VGG16",
            "resnet18": "ResNet18",
        }
        bits = []
        if architecture:
            bits.append(labels.get(architecture, str(architecture)))
        if self.metadata.get("continued_training"):
            bits.append("retrained")
        epoch_count = self.metadata.get("final_epoch") or len(self.metadata.get("training_history") or [])
        if epoch_count:
            bits.append(f"{epoch_count} epoch{'s' if epoch_count != 1 else ''}")

        meta = QLabel("  ·  ".join(bits) if bits else "")
        meta.setWordWrap(True)
        meta.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        row.addWidget(meta)
        row.addStretch(1)
        return row

    def _build_accuracy_block(self) -> QWidget:
        """Headline R² (test, else validation) plus the three outputs."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)

        resolved = resolve_model_r2(self.metadata)
        if resolved:
            scores = resolved["scores"]
            mean = mean_r2(scores)
            split_label = "Test" if resolved["split"] == "test" else "Validation"
            headline_color = DANGER_COLOR if mean < 0 else TEXT_PRIMARY

            headline = QLabel(f"R²  {mean:.2f}")
            headline.setStyleSheet(
                f"color: {headline_color}; font-size: 20px; background: transparent;"
            )
            layout.addWidget(headline)

            breakdown = QLabel(
                f"{split_label}  ·  Bitumen {scores['Bitumen']:.2f}   "
                f"Solids {scores['Solids']:.2f}   Water {scores['Water']:.2f}"
            )
            breakdown.setWordWrap(True)
            breakdown.setStyleSheet(
                f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;"
            )
            layout.addWidget(breakdown)
            return container

        mae = self.metadata.get("best_val_mae") or {}
        if mae:
            mae_label = QLabel(
                f"Val MAE  Bitumen \u00b1{mae.get('Bitumen', 0.0):.2f}%   "
                f"Solids \u00b1{mae.get('Solids', 0.0):.2f}%   "
                f"Water \u00b1{mae.get('Water', 0.0):.2f}%"
            )
            mae_label.setWordWrap(True)
            mae_label.setStyleSheet(
                f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;"
            )
            layout.addWidget(mae_label)
        return container

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)

        self._load_button = QPushButton("Load")
        self._load_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._load_button.setToolTip("Use this model for grading")
        self._load_button.clicked.connect(lambda: self.load_requested.emit(self.metadata))
        row.addWidget(self._load_button)

        self._retrain_button = QPushButton("Retrain")
        self._retrain_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._retrain_button.setToolTip("Continue this model on a new labelled dataset")
        self._retrain_button.setStyleSheet(ghost_button_qss())
        self._retrain_button.clicked.connect(lambda: self.retrain_requested.emit(self.metadata))
        row.addWidget(self._retrain_button)

        self._details_button = QPushButton("Details")
        self._details_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._details_button.setToolTip("Show R²/loss curves and output stats")
        self._details_button.setStyleSheet(link_button_qss())
        self._details_button.clicked.connect(self._toggle_details)
        row.addWidget(self._details_button)

        row.addStretch(1)

        self._delete_button = QPushButton("Delete")
        self._delete_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._delete_button.setToolTip("Delete model")
        self._delete_button.setStyleSheet(link_button_qss(color=DANGER_COLOR))
        self._delete_button.clicked.connect(self._confirm_delete)
        row.addWidget(self._delete_button)

        return row

    def _build_details_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(10)

        separator = QFrame()
        separator.setFixedHeight(1)
        separator.setStyleSheet(f"background-color: {BORDER_COLOR}; border: none;")
        layout.addWidget(separator)

        history = self.metadata.get("training_history") or []
        layout.addWidget(self._build_mae_chart(history))
        layout.addWidget(self._build_loss_chart(history))
        r2_table = self._build_r2_table()
        if r2_table is not None:
            layout.addWidget(r2_table)
        layout.addWidget(self._build_output_stats_table())

        return section

    def _build_mae_chart(self, history: List[Dict[str, Any]]) -> QWidget:
        if not history:
            return self._build_no_history_placeholder()

        figure = Figure(figsize=(4, 2), dpi=100)
        figure.patch.set_facecolor(SURFACE_COLOR)
        axes = figure.add_subplot(111)
        axes.set_facecolor(SURFACE_COLOR)

        epochs = [entry.get("epoch", index + 1) for index, entry in enumerate(history)]
        water_maes = [entry.get("water_mae", 0.0) for entry in history]
        solids_maes = [entry.get("solids_mae", 0.0) for entry in history]
        bitumen_maes = [entry.get("bitumen_mae", 0.0) for entry in history]

        axes.plot(epochs, water_maes, color=WATER_LINE_COLOR, linewidth=1.6, label="Water")
        axes.plot(epochs, solids_maes, color=SOLIDS_LINE_COLOR, linewidth=1.6, label="Solids")
        axes.plot(epochs, bitumen_maes, color=BITUMEN_LINE_COLOR, linewidth=1.6, label="Bitumen")
        style_axes(axes, "MAE (%)")
        legend = axes.legend(loc="upper right", fontsize=7, facecolor=SURFACE_COLOR, edgecolor=BORDER_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_SECONDARY)
        figure.tight_layout()

        canvas = FigureCanvasQTAgg(figure)
        canvas.setFixedHeight(200)
        canvas.setStyleSheet("background-color: transparent;")
        return canvas

    def _build_loss_chart(self, history: List[Dict[str, Any]]) -> QWidget:
        if not history:
            return self._build_no_history_placeholder()

        figure = Figure(figsize=(4, 2), dpi=100)
        figure.patch.set_facecolor(SURFACE_COLOR)
        axes = figure.add_subplot(111)
        axes.set_facecolor(SURFACE_COLOR)

        epochs = [entry.get("epoch", index + 1) for index, entry in enumerate(history)]
        train_losses = [entry.get("train_loss", 0.0) for entry in history]
        val_losses = [entry.get("val_loss", 0.0) for entry in history]

        axes.plot(epochs, train_losses, color=ACCENT_COLOR, linewidth=1.6, label="Train Loss")
        axes.plot(epochs, val_losses, color=VAL_LINE_COLOR, linewidth=1.6, label="Val Loss")
        style_axes(axes, "Loss")
        legend = axes.legend(loc="upper right", fontsize=7, facecolor=SURFACE_COLOR, edgecolor=BORDER_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_SECONDARY)
        figure.tight_layout()

        canvas = FigureCanvasQTAgg(figure)
        canvas.setFixedHeight(200)
        canvas.setStyleSheet("background-color: transparent;")
        return canvas

    @staticmethod
    def _build_no_history_placeholder() -> QWidget:
        label = QLabel("No training history saved.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFixedHeight(200)
        label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 12px; background-color: {BACKGROUND_COLOR}; border-radius: 6px;"
        )
        return label

    def _build_r2_table(self) -> Optional[QTableWidget]:
        splits = model_r2_splits(self.metadata)
        val_r2 = splits.get("val")
        test_r2 = splits.get("test")
        if not val_r2 and not test_r2:
            return None

        headers = ["Output"]
        columns: List[Dict[str, float]] = []
        if val_r2:
            headers.append("Val R²")
            columns.append(val_r2)
        if test_r2:
            headers.append("Test R²")
            columns.append(test_r2)

        table = QTableWidget(len(OUTPUT_NAMES), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setFixedHeight(130)
        table.horizontalHeader().setStretchLastSection(True)
        table.setStyleSheet(
            f"""
            QTableWidget {{
                background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR}; border-radius: 6px; gridline-color: {BORDER_COLOR};
            }}
            QTableWidget::item {{ padding: 3px; }}
            QHeaderView::section {{
                background-color: {SURFACE_COLOR}; color: {TEXT_SECONDARY}; border: none;
                padding: 5px; font-size: 10px; font-weight: 600;
            }}
            """
        )
        for row, label in enumerate(OUTPUT_NAMES):
            values = [label] + [f"{column.get(label, 0.0):.2f}" for column in columns]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if col > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, col, item)
        return table

    def _build_output_stats_table(self) -> QTableWidget:
        output_stats = self.metadata.get("output_stats") or {}

        table = QTableWidget(len(OUTPUT_NAMES), 3)
        table.setHorizontalHeaderLabels(["Output", "Mean", "Std"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setFixedHeight(130)
        table.horizontalHeader().setStretchLastSection(True)
        table.setStyleSheet(
            f"""
            QTableWidget {{
                background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR}; border-radius: 6px; gridline-color: {BORDER_COLOR};
            }}
            QTableWidget::item {{ padding: 3px; }}
            QHeaderView::section {{
                background-color: {SURFACE_COLOR}; color: {TEXT_SECONDARY}; border: none;
                padding: 5px; font-size: 10px; font-weight: 600;
            }}
            """
        )

        for row, label in enumerate(OUTPUT_NAMES):
            stats = output_stats.get(label, {"mean": 0.0, "std": 0.0})
            for col, text in enumerate((label, f"{stats.get('mean', 0.0):.2f}", f"{stats.get('std', 0.0):.2f}")):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if col > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, col, item)

        return table

    # -- Public API ----------------------------------------------------------

    def set_active(self, is_active: bool) -> None:
        """Mark this card as the loaded model."""
        self._is_active = is_active
        self._active_badge.setVisible(is_active)

        if is_active:
            self.setStyleSheet(
                f"QFrame#modelCard {{ background-color: {SURFACE_COLOR};"
                f"border: 1px solid {ACCENT_COLOR}; border-radius: 3px; }}"
            )
            self._load_button.setText("Loaded")
            self._load_button.setStyleSheet(accent_button_qss())
        else:
            self.setStyleSheet(
                f"QFrame#modelCard {{ background-color: {SURFACE_COLOR};"
                f"border: 1px solid {BORDER_COLOR}; border-radius: 3px; }}"
            )
            self._load_button.setText("Load")
            self._load_button.setStyleSheet(accent_button_qss())

    def _toggle_details(self) -> None:
        self._details_expanded = not self._details_expanded
        self._details_section.setVisible(self._details_expanded)
        self._details_button.setText("Hide details" if self._details_expanded else "Details")

    def _confirm_delete(self) -> None:
        name = self.metadata.get("name") or "this model"
        reply = QMessageBox.question(
            self,
            "Delete Model",
            f'Delete "{name}"? This removes its .pt and .json files for good.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.delete_requested.emit(self.metadata)

