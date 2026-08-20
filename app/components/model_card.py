"""
Model card.

One saved model: name, date, type pill, epochs, best val MAE, plus load /
details / delete. Details expands MAE/loss curves and output stats.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal  # noqa: E402
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF  # noqa: E402
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

from app.constants import OUTPUT_NAMES
from app.theme import (
    ACCENT_COLOR,
    BACKGROUND_COLOR,
    BITUMEN_LINE_COLOR,
    BORDER_COLOR,
    DANGER_COLOR,
    DANGER_HOVER_BG,
    PILL_BACKGROUND,
    REGRESSION_PILL_COLOR,
    SOLIDS_LINE_COLOR,
    SURFACE_COLOR,
    SURFACE_HOVER_COLOR,
    TEXT_INVERSE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    VAL_LINE_COLOR,
    WATER_LINE_COLOR,
    accent_button_qss,
)

OUTPUT_LABELS = OUTPUT_NAMES


def _build_check_icon(color: str, size: int = 14) -> QIcon:
    """Small checkmark drawn with QPainter."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(1.8)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.drawPolyline(
        QPolygonF(
            [
                QPointF(size * 0.18, size * 0.52),
                QPointF(size * 0.42, size * 0.75),
                QPointF(size * 0.84, size * 0.25),
            ]
        )
    )
    painter.end()
    return QIcon(pixmap)


def _build_chevron_icon(direction: str, color: str, size: int = 12) -> QIcon:
    """Up/down chevron for the details accordion."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)

    margin = size * 0.24
    if direction == "up":
        points = [QPointF(margin, size - margin), QPointF(size / 2, margin), QPointF(size - margin, size - margin)]
    else:
        points = [QPointF(margin, margin), QPointF(size / 2, size - margin), QPointF(size - margin, margin)]
    painter.drawPolyline(QPolygonF(points))
    painter.end()
    return QIcon(pixmap)


def _build_trash_icon(color: str, size: int = 16) -> QIcon:
    """Trash-can icon for delete."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(1.4)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)

    top_y = size * 0.3
    painter.drawLine(QPointF(size * 0.18, top_y), QPointF(size * 0.82, top_y))
    painter.drawLine(QPointF(size * 0.4, top_y), QPointF(size * 0.42, size * 0.16))
    painter.drawLine(QPointF(size * 0.42, size * 0.16), QPointF(size * 0.58, size * 0.16))
    painter.drawLine(QPointF(size * 0.58, size * 0.16), QPointF(size * 0.6, top_y))

    body = QRectF(size * 0.26, top_y, size * 0.48, size * 0.56)
    painter.drawRoundedRect(body, 1.5, 1.5)
    painter.drawLine(QPointF(size * 0.4, top_y + size * 0.1), QPointF(size * 0.4, top_y + size * 0.42))
    painter.drawLine(QPointF(size * 0.6, top_y + size * 0.1), QPointF(size * 0.6, top_y + size * 0.42))

    painter.end()
    return QIcon(pixmap)


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
        layout.addWidget(self._build_mae_summary_row())
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
        name_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 15px; font-weight: 700; background: transparent;")
        name_col.addWidget(name_label)

        date_label = QLabel(self._format_date(self.metadata.get("created_at")))
        date_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        name_col.addWidget(date_label)

        row.addLayout(name_col, 1)

        self._active_badge = QLabel("Active")
        self._active_badge.setStyleSheet(
            f"background-color: {ACCENT_COLOR}; color: #13151A; font-size: 10px; font-weight: 700;"
            f"border-radius: 8px; padding: 3px 10px;"
        )
        row.addWidget(self._active_badge, 0, Qt.AlignmentFlag.AlignTop)

        return row

    def _build_pills_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        row.addWidget(self._build_pill("Regression", background=REGRESSION_PILL_COLOR, color="#FFFFFF"))

        architecture = self.metadata.get("architecture")
        if architecture:
            labels = {
                "baseline": "Baseline CNN",
                "resnet50": "ResNet50",
                "vgg16": "VGG16",
                "resnet18": "ResNet18",
            }
            row.addWidget(self._build_pill(labels.get(architecture, str(architecture))))

        if self.metadata.get("continued_training"):
            row.addWidget(self._build_pill("Retrained"))

        epoch_count = self.metadata.get("final_epoch") or len(self.metadata.get("training_history") or [])
        row.addWidget(self._build_pill(f"{epoch_count} epoch{'s' if epoch_count != 1 else ''}"))

        row.addStretch(1)
        return row

    def _build_pill(self, text: str, background: str = PILL_BACKGROUND, color: str = TEXT_PRIMARY) -> QLabel:
        pill = QLabel(text)
        pill.setStyleSheet(
            f"background-color: {background}; color: {color}; font-size: 11px; font-weight: 600;"
            f"border-radius: 8px; padding: 4px 10px;"
        )
        return pill

    def _build_mae_summary_row(self) -> QLabel:
        best_val_mae = self.metadata.get("best_val_mae") or {}
        parts = [
            f"Val Water \u00b1{best_val_mae.get('Water', 0.0):.2f}%",
            f"Solids \u00b1{best_val_mae.get('Solids', 0.0):.2f}%",
            f"Bitumen \u00b1{best_val_mae.get('Bitumen', 0.0):.2f}%",
        ]
        test_mae = self.metadata.get("test_mae") or None
        if test_mae:
            parts.append(
                f"| Test Water \u00b1{test_mae.get('Water', 0.0):.2f}%  "
                f"Solids \u00b1{test_mae.get('Solids', 0.0):.2f}%  "
                f"Bitumen \u00b1{test_mae.get('Bitumen', 0.0):.2f}%"
            )
        label = QLabel("   ".join(parts))
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; background: transparent;")
        return label

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        self._load_button = QPushButton("Load Model")
        self._load_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._load_button.setIconSize(QSize(12, 12))
        self._load_button.setToolTip("Use this model for grading")
        self._load_button.clicked.connect(lambda: self.load_requested.emit(self.metadata))
        row.addWidget(self._load_button)

        self._retrain_button = QPushButton("Retrain")
        self._retrain_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._retrain_button.setToolTip("Continue this model on a new labelled dataset")
        self._retrain_button.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {TEXT_PRIMARY};"
            f"border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 8px 14px; font-size: 12px; }}"
            f"QPushButton:hover {{ background-color: {SURFACE_HOVER_COLOR}; }}"
        )
        self._retrain_button.clicked.connect(lambda: self.retrain_requested.emit(self.metadata))
        row.addWidget(self._retrain_button)

        self._details_button = QPushButton("  Details")
        self._details_button.setIcon(_build_chevron_icon("down", TEXT_PRIMARY))
        self._details_button.setIconSize(QSize(12, 12))
        self._details_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._details_button.setToolTip("Show MAE/loss curves and output stats")
        self._details_button.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {TEXT_PRIMARY};"
            f"border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 8px 14px; font-size: 12px; }}"
            f"QPushButton:hover {{ background-color: {SURFACE_HOVER_COLOR}; }}"
        )
        self._details_button.clicked.connect(self._toggle_details)
        row.addWidget(self._details_button)

        row.addStretch(1)

        self._delete_button = QPushButton()
        self._delete_button.setIcon(_build_trash_icon(DANGER_COLOR))
        self._delete_button.setIconSize(QSize(16, 16))
        self._delete_button.setFixedSize(34, 34)
        self._delete_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._delete_button.setToolTip("Delete model")
        self._delete_button.setStyleSheet(
            f"QPushButton {{ background-color: transparent; border: 1px solid {BORDER_COLOR};"
            f"border-radius: 6px; padding: 0px; }}"
            f"QPushButton:hover {{ background-color: {DANGER_HOVER_BG}; border: 1px solid {DANGER_COLOR}; }}"
        )
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
        self._style_axes(axes, "MAE (%)")
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
        self._style_axes(axes, "Loss")
        legend = axes.legend(loc="upper right", fontsize=7, facecolor=SURFACE_COLOR, edgecolor=BORDER_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_SECONDARY)
        figure.tight_layout()

        canvas = FigureCanvasQTAgg(figure)
        canvas.setFixedHeight(200)
        canvas.setStyleSheet("background-color: transparent;")
        return canvas

    @staticmethod
    def _style_axes(axes, ylabel: str) -> None:
        axes.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        for spine in axes.spines.values():
            spine.set_color(BORDER_COLOR)
        axes.set_xlabel("Epoch", color=TEXT_SECONDARY, fontsize=8)
        axes.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=8)
        axes.grid(True, color=BORDER_COLOR, linewidth=0.5, alpha=0.5)

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
        val_r2 = self.metadata.get("best_val_r2") or {}
        test_r2 = self.metadata.get("test_r2") or {}
        if not val_r2 and not test_r2:
            return None

        has_test = bool(test_r2)
        table = QTableWidget(len(OUTPUT_LABELS), 3 if has_test else 2)
        headers = ["Output", "Val R²"] + (["Test R²"] if has_test else [])
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
        for row, label in enumerate(OUTPUT_LABELS):
            values = [label, f"{val_r2.get(label, 0.0):.3f}"]
            if has_test:
                values.append(f"{test_r2.get(label, 0.0):.3f}")
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if col > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, col, item)
        return table

    def _build_output_stats_table(self) -> QTableWidget:
        output_stats = self.metadata.get("output_stats") or {}

        table = QTableWidget(len(OUTPUT_LABELS), 3)
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

        for row, label in enumerate(OUTPUT_LABELS):
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
        """Update active styling (left border, badge, Load button)."""
        self._is_active = is_active
        self._active_badge.setVisible(is_active)

        if is_active:
            self.setStyleSheet(
                f"""
                QFrame#modelCard {{
                    background-color: {SURFACE_COLOR}; border-radius: 8px;
                    border-top: 1px solid {BORDER_COLOR}; border-right: 1px solid {BORDER_COLOR};
                    border-bottom: 1px solid {BORDER_COLOR}; border-left: 3px solid {ACCENT_COLOR};
                }}
                """
            )
            self._load_button.setText("  Active Model")
            self._load_button.setIcon(_build_check_icon(TEXT_INVERSE))
            self._load_button.setStyleSheet(
                accent_button_qss(extra="font-weight: 700; padding: 8px 14px; font-size: 12px;")
            )
        else:
            self.setStyleSheet(
                f"QFrame#modelCard {{ background-color: {SURFACE_COLOR}; border: 1px solid {BORDER_COLOR};"
                f"border-radius: 8px; }}"
            )
            self._load_button.setText("Load Model")
            self._load_button.setIcon(QIcon())
            self._load_button.setStyleSheet(
                accent_button_qss(extra="font-weight: 700; padding: 8px 14px; font-size: 12px;")
            )

    def is_active(self) -> bool:
        return self._is_active

    # -- Internal handlers --------------------------------------------------

    def _toggle_details(self) -> None:
        self._details_expanded = not self._details_expanded
        self._details_section.setVisible(self._details_expanded)
        self._details_button.setText("  Hide Details" if self._details_expanded else "  Details")
        self._details_button.setIcon(
            _build_chevron_icon("up" if self._details_expanded else "down", TEXT_PRIMARY)
        )

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

    @staticmethod
    def _format_date(created_at: Optional[str]) -> str:
        if not created_at:
            return ""
        try:
            parsed = datetime.fromisoformat(created_at)
        except ValueError:
            return ""
        return parsed.strftime("Created %b %d %Y at %H:%M")
