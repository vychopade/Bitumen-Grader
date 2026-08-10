"""
Model Training page (regression).

Provides the UI for training a ``BitumenRegressor`` to predict Water,
Solids, and Bitumen content from sample images: loading a labelled CSV,
selecting the matching image folder, reviewing a dataset summary,
configuring hyperparameters, and running training on a background QThread
while streaming live progress (loss + MAE curves, compositional check,
timestamped log) into an embedded progress panel.

Dataset loading, filename matching, and train/val splitting are all handled
by ``RegressionDataset`` (see ``app.ml.dataset``); this page only reads a
few of its methods (``get_output_stats``/``get_match_summary``) plus the raw
``matched`` list to build the Step 2/3 summaries, and never holds decoded
images itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd
import torch
from PyQt6.QtCore import QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from torch.utils.data import DataLoader

from app.components.progress_panel import ProgressPanel
from app.ml.cnn_model import BitumenRegressor
from app.ml.dataset import RegressionDataset
from app.ml.trainer import RegressionTrainer, RegressionTrainingResult
from app.pages.model_manager_page import ModelManagerPage
from app.utils.data_io import SUPPORTED_EXTENSIONS as SUPPORTED_LABEL_EXTENSIONS
from app.utils.data_io import read_labels_file
from app.utils.model_io import load_model_metadata, save_model
from app.utils.shortcuts import bind_page_shortcuts, shortcut_tooltip, unbind_page_shortcuts

if TYPE_CHECKING:
    from app.main_window import MainWindow

# --------------------------------------------------------------------------
# Design tokens (kept local so this page has no dependency on MainWindow)
# --------------------------------------------------------------------------

BACKGROUND_COLOR = "#1A1C20"
SURFACE_COLOR = "#22252C"
BORDER_COLOR = "#33373F"
ACCENT_COLOR = "#E8A838"
ACCENT_HOVER_COLOR = "#C98A20"
TEXT_PRIMARY = "#E8E9EC"
TEXT_SECONDARY = "#8B909A"
DANGER_COLOR = "#E5484D"
DANGER_HOVER_BG = "rgba(229, 72, 77, 40)"
SUCCESS_COLOR = "#3CB878"

LEFT_PANEL_WIDTH = 500
MAX_UNMATCHED_PREVIEW = 200
VAL_SPLIT_REBUILD_DEBOUNCE_MS = 300

EXPECTED_COLUMNS = list(RegressionDataset.EXPECTED_COLUMNS)  # ["Image", "Pan", "Water", "Solids", "Bitumen"]
OUTPUT_LABELS = ("Water", "Solids", "Bitumen")

BATCH_SIZE_OPTIONS = (8, 16, 32, 64)
LEARNING_RATE_OPTIONS = (0.0001, 0.001, 0.01)
OPTIMIZER_OPTIONS = ("Adam", "SGD")

_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

HYPERPARAM_INFO_TEXT = (
    "Epochs: How many times the model sees your full dataset.\n"
    "Batch Size: How many images processed at once. 32 works well for most computers.\n"
    "Learning Rate: How fast the model adjusts. 0.001 is a safe starting point.\n"
    "Optimizer: Adam works well for most cases.\n"
    "Weight Decay: Helps prevent the model from memorising your training images. Leave at default.\n"
    "Validation Split: Images held back to test the model during training. 20% is recommended.\n"
    "Pretrained Backbone: Starts from a model pre-trained on millions of images. Almost always improves results.\n"
    "Normalise Targets: Scales Water, Solids, and Bitumen to a common range so the model learns all three equally.\n"
    "Early Stopping Patience: Training stops if results stop improving for this many epochs in a row."
)


def _default_num_workers() -> int:
    """PyTorch DataLoader worker-process count for streaming images from disk.

    Background worker processes decode images in parallel while the CPU/GPU
    trains on the previous batch. Windows' multiprocessing start method
    (``spawn``, with no ``fork``) makes worker processes considerably more
    expensive to start in bundled desktop apps, so we fall back to loading
    on the main process there (0 workers).
    """
    import platform

    return 0 if platform.system() == "Windows" else 4


def _build_info_icon(color: str, size: int = 14) -> QIcon:
    """Draw a small "info circle" icon (avoids relying on the \u24d8 glyph's font coverage)."""
    from PyQt6.QtCore import QPointF, QRectF

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    outline_pen = QPen(QColor(color))
    outline_pen.setWidthF(1.3)
    painter.setPen(outline_pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    margin = size * 0.08
    painter.drawEllipse(QRectF(margin, margin, size - 2 * margin, size - 2 * margin))

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(color))
    dot_radius = size * 0.07
    painter.drawEllipse(QPointF(size / 2, size * 0.32), dot_radius, dot_radius)

    stem_pen = QPen(QColor(color))
    stem_pen.setWidthF(1.8)
    stem_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(stem_pen)
    painter.drawLine(QPointF(size / 2, size * 0.48), QPointF(size / 2, size * 0.74))

    painter.end()
    return QIcon(pixmap)


class _CsvDropZone(QFrame):
    """Dashed drag-and-drop target for the training CSV, with a "Browse" button."""

    file_selected = pyqtSignal(str)

    _SUPPORTED_EXTENSIONS = SUPPORTED_LABEL_EXTENSIONS

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("csvDropZone")
        self.setAcceptDrops(True)
        self.setFixedHeight(120)
        self._build_ui()
        self._apply_style(active=False)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("Drag & drop your CSV or Excel file here")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600; background: transparent;")
        layout.addWidget(title)

        subtitle = QLabel("Accepts .csv, .txt, .xlsx, and .xls files.")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        layout.addWidget(subtitle)

        self.browse_button = QPushButton("Browse")
        self.browse_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_button.setFixedWidth(120)
        self.browse_button.setStyleSheet(
            f"QPushButton {{ background-color: {ACCENT_COLOR}; color: #13151A; font-weight: 600;"
            f"border: none; border-radius: 6px; padding: 7px 14px; }}"
            f"QPushButton:hover {{ background-color: {ACCENT_HOVER_COLOR}; }}"
        )
        self.browse_button.clicked.connect(self._browse_file)
        layout.addWidget(self.browse_button, 0, Qt.AlignmentFlag.AlignHCenter)

    def _apply_style(self, active: bool) -> None:
        border_color = ACCENT_HOVER_COLOR if active else ACCENT_COLOR
        background = "#2A2E36" if active else SURFACE_COLOR
        self.setStyleSheet(
            f"QFrame#csvDropZone {{ background-color: {background}; border: 2px dashed {border_color};"
            f"border-radius: 8px; }}"
        )

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Label File",
            "",
            "All Supported Files (*.csv *.txt *.xlsx *.xls);;CSV Files (*.csv);;"
            "Excel Files (*.xlsx *.xls);;Text Files (*.txt)",
        )
        if path:
            self.file_selected.emit(path)

    @classmethod
    def _is_supported(cls, path: str) -> bool:
        return path.lower().endswith(cls._SUPPORTED_EXTENSIONS)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        mime_data = event.mimeData()
        if mime_data.hasUrls() and any(
            url.isLocalFile() and self._is_supported(url.toLocalFile()) for url in mime_data.urls()
        ):
            event.acceptProposedAction()
            self._apply_style(active=True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: D401 - Qt override
        self._apply_style(active=False)

    def dropEvent(self, event: QDropEvent) -> None:
        self._apply_style(active=False)
        paths = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if url.isLocalFile() and self._is_supported(url.toLocalFile())
        ]
        if paths:
            self.file_selected.emit(paths[0])
            event.acceptProposedAction()
        else:
            event.ignore()


class _MatchSummaryCard(QFrame):
    """Shows "N of M images matched" plus expandable lists of unmatched filenames and invalid rows."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet(f"QFrame {{ background-color: {BACKGROUND_COLOR}; border-radius: 6px; }}")

        self._unmatched_count = 0
        self._invalid_count = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        self._summary_label = QLabel("")
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        self._invalid_summary_label = QLabel("")
        self._invalid_summary_label.setWordWrap(True)
        self._invalid_summary_label.setStyleSheet(
            f"color: {DANGER_COLOR}; font-size: 12px; font-weight: 600; background: transparent;"
        )
        self._invalid_summary_label.setVisible(False)
        layout.addWidget(self._invalid_summary_label)

        self._tip_label = QLabel("")
        self._tip_label.setWordWrap(True)
        self._tip_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        self._tip_label.setVisible(False)
        layout.addWidget(self._tip_label)

        self.toggle_button = QPushButton("")
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {TEXT_SECONDARY}; border: none;"
            f"text-decoration: underline; font-size: 11px; text-align: left; padding: 0px; }}"
            f"QPushButton:hover {{ color: {TEXT_PRIMARY}; }}"
        )
        self.toggle_button.clicked.connect(self._toggle_unmatched)
        self.toggle_button.setVisible(False)
        layout.addWidget(self.toggle_button, 0, Qt.AlignmentFlag.AlignLeft)

        self._unmatched_label = QLabel("")
        self._unmatched_label.setWordWrap(True)
        self._unmatched_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        self._unmatched_label.setVisible(False)
        layout.addWidget(self._unmatched_label)

        self._invalid_toggle_button = QPushButton("")
        self._invalid_toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._invalid_toggle_button.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {DANGER_COLOR}; border: none;"
            f"text-decoration: underline; font-size: 11px; text-align: left; padding: 0px; }}"
            f"QPushButton:hover {{ color: {TEXT_PRIMARY}; }}"
        )
        self._invalid_toggle_button.clicked.connect(self._toggle_invalid)
        self._invalid_toggle_button.setVisible(False)
        layout.addWidget(self._invalid_toggle_button, 0, Qt.AlignmentFlag.AlignLeft)

        self._invalid_rows_label = QLabel("")
        self._invalid_rows_label.setWordWrap(True)
        self._invalid_rows_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        self._invalid_rows_label.setVisible(False)
        layout.addWidget(self._invalid_rows_label)

    def update_summary(self, match_summary: Dict) -> None:
        matched = match_summary["matched"]
        total = match_summary["total_csv_rows"]
        rate_pct = match_summary["match_rate"] * 100

        self._summary_label.setText(f"{matched} of {total} images matched")
        if rate_pct > 90:
            color = SUCCESS_COLOR
        elif rate_pct >= 50:
            color = ACCENT_COLOR
        else:
            color = DANGER_COLOR
        self._summary_label.setStyleSheet(
            f"color: {color}; font-size: 15px; font-weight: 700; background: transparent;"
        )

        if rate_pct < 50:
            self._tip_label.setText(
                "Check that filenames in your CSV match your image files. The app tries "
                "matching with and without file extensions automatically."
            )
            self._tip_label.setVisible(True)
        else:
            self._tip_label.setVisible(False)

        unmatched_files: List[str] = list(match_summary.get("unmatched_files", []))
        self._unmatched_count = len(unmatched_files)
        if unmatched_files:
            preview = unmatched_files[:MAX_UNMATCHED_PREVIEW]
            text = ", ".join(preview)
            if len(unmatched_files) > len(preview):
                text += f", and {len(unmatched_files) - len(preview)} more"
            self._unmatched_label.setText(text)
            self.toggle_button.setVisible(True)
        else:
            self.toggle_button.setVisible(False)

        self._unmatched_label.setVisible(False)
        self.toggle_button.setText(f"Show unmatched filenames ({self._unmatched_count}) \u25be")

        invalid_rows: List[Dict] = list(match_summary.get("invalid_rows", []))
        self._invalid_count = len(invalid_rows)
        if invalid_rows:
            word = "row" if self._invalid_count == 1 else "rows"
            self._invalid_summary_label.setText(
                f"{self._invalid_count} {word} skipped \u2014 could not read their Water/Solids/Bitumen/Pan values."
            )
            self._invalid_summary_label.setVisible(True)

            preview = invalid_rows[:MAX_UNMATCHED_PREVIEW]
            text = "\n".join(f"\u201c{entry['image']}\u201d \u2014 {entry['reason']}" for entry in preview)
            if len(invalid_rows) > len(preview):
                text += f"\n\u2026and {len(invalid_rows) - len(preview)} more"
            self._invalid_rows_label.setText(text)
            self._invalid_toggle_button.setVisible(True)
        else:
            self._invalid_summary_label.setVisible(False)
            self._invalid_toggle_button.setVisible(False)

        self._invalid_rows_label.setVisible(False)
        self._invalid_toggle_button.setText(f"Show invalid rows ({self._invalid_count}) \u25be")

    def _toggle_invalid(self) -> None:
        showing = not self._invalid_rows_label.isVisible()
        self._invalid_rows_label.setVisible(showing)
        verb = "Hide" if showing else "Show"
        arrow = "\u25b4" if showing else "\u25be"
        self._invalid_toggle_button.setText(f"{verb} invalid rows ({self._invalid_count}) {arrow}")

    def _toggle_unmatched(self) -> None:
        showing = not self._unmatched_label.isVisible()
        self._unmatched_label.setVisible(showing)
        verb = "Hide" if showing else "Show"
        arrow = "\u25b4" if showing else "\u25be"
        self.toggle_button.setText(f"{verb} unmatched filenames ({self._unmatched_count}) {arrow}")


class _DatasetSummaryCard(QFrame):
    """Total/train/val counts, per-output ranges, and a pan-grade distribution with bars."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet(f"QFrame {{ background-color: {BACKGROUND_COLOR}; border-radius: 6px; }}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        self._counts_label = QLabel("")
        self._counts_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; background: transparent;")
        layout.addWidget(self._counts_label)

        ranges_title = QLabel("Output ranges:")
        ranges_title.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; font-weight: 600; background: transparent;"
        )
        layout.addWidget(ranges_title)

        self._ranges_label = QLabel("")
        self._ranges_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; background: transparent;")
        layout.addWidget(self._ranges_label)

        pan_title = QLabel("Pan grade distribution:")
        pan_title.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; font-weight: 600; background: transparent;"
        )
        layout.addWidget(pan_title)

        self._pan_container = QVBoxLayout()
        self._pan_container.setSpacing(6)
        layout.addLayout(self._pan_container)

    def update_summary(
        self,
        total: int,
        train_count: int,
        val_count: int,
        val_fraction: float,
        output_ranges: Dict[str, Dict[str, float]],
        pan_distribution: Dict[int, int],
    ) -> None:
        train_pct = round((1 - val_fraction) * 100)
        val_pct = round(val_fraction * 100)
        self._counts_label.setText(
            f"Total matched samples: {total}\n"
            f"Training samples: {train_count} ({train_pct}%)\n"
            f"Validation samples: {val_count} ({val_pct}%)"
        )

        lines = []
        for label in OUTPUT_LABELS:
            values = output_ranges.get(label, {"min": 0.0, "max": 0.0, "mean": 0.0})
            lines.append(
                f"{label}: {values['min']:.2f} \u2013 {values['max']:.2f} % (mean {values['mean']:.2f})"
            )
        self._ranges_label.setText("\n".join(lines))

        while self._pan_container.count():
            item = self._pan_container.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        max_count = max(pan_distribution.values()) if pan_distribution else 1
        for grade in sorted(pan_distribution.keys()):
            count = pan_distribution[grade]
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            label = QLabel(f"Grade {grade}: {count}")
            label.setFixedWidth(90)
            label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
            row_layout.addWidget(label)

            bar = QProgressBar()
            bar.setRange(0, max_count)
            bar.setValue(count)
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            bar.setStyleSheet(
                f"QProgressBar {{ background-color: {SURFACE_COLOR}; border-radius: 5px; border: none; }}"
                f"QProgressBar::chunk {{ background-color: {ACCENT_COLOR}; border-radius: 5px; }}"
            )
            row_layout.addWidget(bar, 1)

            self._pan_container.addWidget(row_widget)


class TrainPage(QWidget):
    """Page for configuring and running regression training jobs.

    Images are never routed through ``main_window`` -- this page is fully
    self-contained: the user loads a
    CSV and picks an image folder directly here, ``RegressionDataset``
    handles filename matching/splitting, and training runs on a QThread
    with ``RegressionTrainer`` (itself a ``QObject``) moved onto that
    thread; its signals are connected straight to the embedded
    ``ProgressPanel``, which Qt automatically marshals onto the
    main thread since the panel lives there.
    """

    def __init__(self, main_window: Optional["MainWindow"] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = main_window

        self._csv_path: Optional[str] = None
        self._csv_dataframe: Optional[pd.DataFrame] = None
        self._image_dir: Optional[str] = None
        self._train_dataset: Optional[RegressionDataset] = None
        self._val_dataset: Optional[RegressionDataset] = None

        self._csv_drop_zone: Optional[_CsvDropZone] = None
        self._csv_loaded_label: Optional[QLabel] = None
        self._csv_error_banner: Optional[QFrame] = None
        self._csv_error_label: Optional[QLabel] = None
        self._preview_table: Optional[QTableWidget] = None
        self._column_mapping_frame: Optional[QFrame] = None

        self._select_folder_button: Optional[QPushButton] = None
        self._folder_path_label: Optional[QLabel] = None
        self._folder_count_label: Optional[QLabel] = None
        self._match_card: Optional[_MatchSummaryCard] = None

        self._step3_frame: Optional[QFrame] = None
        self._dataset_summary_card: Optional[_DatasetSummaryCard] = None

        self._model_name_edit: Optional[QLineEdit] = None
        self._info_button: Optional[QPushButton] = None
        self._info_label: Optional[QLabel] = None
        self._epochs_spin: Optional[QSpinBox] = None
        self._batch_size_combo: Optional[QComboBox] = None
        self._lr_combo: Optional[QComboBox] = None
        self._optimizer_combo: Optional[QComboBox] = None
        self._weight_decay_spin: Optional[QDoubleSpinBox] = None
        self._val_split_slider: Optional[QSlider] = None
        self._val_split_label: Optional[QLabel] = None
        self._pretrained_checkbox: Optional[QCheckBox] = None
        self._normalise_checkbox: Optional[QCheckBox] = None
        self._patience_spin: Optional[QSpinBox] = None
        self._start_button: Optional[QPushButton] = None
        self._stop_button: Optional[QPushButton] = None
        self._status_label: Optional[QLabel] = None
        self._progress_panel: Optional[ProgressPanel] = None

        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.timeout.connect(self._maybe_rebuild_dataset_summary)

        self._thread: Optional[QThread] = None
        self._trainer: Optional[RegressionTrainer] = None
        self._pending_model_name: Optional[str] = None

        self._shortcut_bindings: List[tuple] = []
        self._tab_order_applied = False

        self._build_ui()
        self._update_start_button_state()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 24)
        root.setSpacing(18)

        root.addLayout(self._build_header())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 4, 4)
        content_layout.setSpacing(20)

        content_layout.addWidget(self._build_left_panel())
        content_layout.addWidget(self._build_right_panel(), 1)

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    def _apply_tab_order(self) -> None:
        """Chain focus order top-to-bottom through the left panel, then the right."""
        chain = [
            self._csv_drop_zone.browse_button if self._csv_drop_zone else None,
            self._select_folder_button,
            self._model_name_edit,
            self._info_button,
            self._epochs_spin,
            self._batch_size_combo,
            self._lr_combo,
            self._optimizer_combo,
            self._weight_decay_spin,
            self._val_split_slider,
            self._patience_spin,
            self._pretrained_checkbox,
            self._normalise_checkbox,
            self._start_button,
            self._stop_button,
        ]
        for earlier, later in zip(chain, chain[1:]):
            if earlier is not None and later is not None:
                QWidget.setTabOrder(earlier, later)

    def _build_header(self) -> QVBoxLayout:
        header = QVBoxLayout()
        header.setSpacing(4)

        title = QLabel("Train a New Model")
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 20px; font-weight: 600;")
        header.addWidget(title)

        subtitle = QLabel(
            "Load your labelled CSV/Excel file and image folder, configure settings, and "
            "train a model to predict Water, Solids, and Bitumen content."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px;")
        header.addWidget(subtitle)

        return header

    def _make_section_frame(self, title: str):
        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ background-color: {SURFACE_COLOR}; border-radius: 8px; }}")

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 600; background: transparent;"
        )
        layout.addWidget(title_label)

        return frame, layout

    def _secondary_button_style(self) -> str:
        return (
            f"QPushButton {{ background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};"
            f"border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 8px 12px; font-size: 12px; }}"
            f"QPushButton:hover {{ background-color: #2A2E36; }}"
            f"QPushButton:disabled {{ color: {TEXT_SECONDARY}; border: 1px solid #2A2D34; }}"
        )

    # -- Left panel: CSV / folder / dataset summary --------------------------

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        container.setFixedWidth(LEFT_PANEL_WIDTH)
        container.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(self._build_csv_section())
        layout.addWidget(self._build_folder_section())

        self._step3_frame = self._build_dataset_summary_section()
        self._step3_frame.setVisible(False)
        layout.addWidget(self._step3_frame)

        layout.addStretch(1)
        return container

    def _build_csv_section(self) -> QFrame:
        section, layout = self._make_section_frame("Step 1 \u2014 Load CSV / Excel File")

        self._csv_drop_zone = _CsvDropZone()
        self._csv_drop_zone.file_selected.connect(self._on_csv_selected)
        layout.addWidget(self._csv_drop_zone)
        self._shortcut_bindings.append((self._csv_drop_zone.browse_button, "B"))

        self._csv_loaded_label = QLabel("")
        self._csv_loaded_label.setWordWrap(True)
        self._csv_loaded_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        self._csv_loaded_label.setVisible(False)
        layout.addWidget(self._csv_loaded_label)

        self._csv_error_banner = QFrame()
        self._csv_error_banner.setObjectName("csvErrorBanner")
        self._csv_error_banner.setStyleSheet(
            # Scoped to #csvErrorBanner -- QLabel is a QFrame subclass in Qt,
            # so a bare "QFrame" selector would also draw this border around
            # the word-wrapped error label nested inside, not just the banner.
            f"QFrame#csvErrorBanner {{ background-color: rgba(229, 72, 77, 25); border: 1px solid {DANGER_COLOR};"
            f"border-radius: 6px; }}"
        )
        error_layout = QVBoxLayout(self._csv_error_banner)
        error_layout.setContentsMargins(12, 10, 12, 10)
        self._csv_error_label = QLabel("")
        self._csv_error_label.setWordWrap(True)
        self._csv_error_label.setStyleSheet(f"color: {DANGER_COLOR}; font-size: 11px; background: transparent;")
        error_layout.addWidget(self._csv_error_label)
        self._csv_error_banner.setVisible(False)
        layout.addWidget(self._csv_error_banner)

        self._preview_table = QTableWidget()
        self._preview_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._preview_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._preview_table.verticalHeader().setVisible(False)
        self._preview_table.setFixedHeight(160)
        self._preview_table.horizontalHeader().setStretchLastSection(True)
        self._preview_table.setStyleSheet(
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
        self._preview_table.setVisible(False)
        layout.addWidget(self._preview_table)

        self._column_mapping_frame = QFrame()
        self._column_mapping_frame.setStyleSheet(
            f"QFrame {{ background-color: {BACKGROUND_COLOR}; border-radius: 6px; }}"
        )
        mapping_form = QFormLayout(self._column_mapping_frame)
        mapping_form.setContentsMargins(12, 10, 12, 10)
        mapping_form.setSpacing(6)
        self._add_mapping_row(mapping_form, "Filename column:", "Image")
        self._add_mapping_row(mapping_form, "Outputs to predict:", ", ".join(OUTPUT_LABELS))
        self._add_mapping_row(mapping_form, "Display only:", "Pan")
        self._column_mapping_frame.setVisible(False)
        layout.addWidget(self._column_mapping_frame)

        return section

    def _add_mapping_row(self, form: QFormLayout, label_text: str, value_text: str) -> None:
        label = QLabel(label_text)
        label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        value = QLabel(value_text)
        value.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 11px; font-weight: 600; background: transparent;")
        form.addRow(label, value)

    def _build_folder_section(self) -> QFrame:
        section, layout = self._make_section_frame("Step 2 \u2014 Select Image Folder")

        self._select_folder_button = QPushButton("Select Folder")
        self._select_folder_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._select_folder_button.setStyleSheet(self._secondary_button_style())
        self._select_folder_button.setToolTip(shortcut_tooltip("Select the folder containing your images", "F"))
        self._select_folder_button.clicked.connect(self._on_select_folder)
        layout.addWidget(self._select_folder_button)
        self._shortcut_bindings.append((self._select_folder_button, "F"))

        self._folder_path_label = QLabel("")
        self._folder_path_label.setWordWrap(True)
        self._folder_path_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 11px; background: transparent;")
        self._folder_path_label.setVisible(False)
        layout.addWidget(self._folder_path_label)

        self._folder_count_label = QLabel("")
        self._folder_count_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        self._folder_count_label.setVisible(False)
        layout.addWidget(self._folder_count_label)

        self._match_card = _MatchSummaryCard()
        self._match_card.setVisible(False)
        self._shortcut_bindings.append((self._match_card.toggle_button, "U"))
        layout.addWidget(self._match_card)

        return section

    def _build_dataset_summary_section(self) -> QFrame:
        section, layout = self._make_section_frame("Step 3 \u2014 Dataset Summary")
        self._dataset_summary_card = _DatasetSummaryCard()
        layout.addWidget(self._dataset_summary_card)
        return section

    # -- Right panel: model settings & hyperparameters ------------------------

    def _build_right_panel(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(self._build_model_settings_section())
        layout.addWidget(self._build_hyperparams_section())

        self._start_button = QPushButton("Start Training")
        self._start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_button.setFixedHeight(48)
        self._start_button.setStyleSheet(
            f"QPushButton {{ background-color: {ACCENT_COLOR}; color: #13151A; font-weight: 700;"
            f"font-size: 14px; border: none; border-radius: 6px; }}"
            f"QPushButton:hover {{ background-color: {ACCENT_HOVER_COLOR}; }}"
            f"QPushButton:disabled {{ background-color: #4A4230; color: #8B8168; }}"
        )
        self._start_button.clicked.connect(self._on_start_training)
        layout.addWidget(self._start_button)
        self._shortcut_bindings.append((self._start_button, "S"))

        self._stop_button = QPushButton("Stop Training")
        self._stop_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_button.setFixedHeight(38)
        self._stop_button.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {DANGER_COLOR}; font-weight: 600;"
            f"border: 1px solid {DANGER_COLOR}; border-radius: 6px; }}"
            f"QPushButton:hover {{ background-color: {DANGER_HOVER_BG}; }}"
            f"QPushButton:disabled {{ color: #6B5050; border: 1px solid #4A3838; }}"
        )
        self._stop_button.setToolTip(shortcut_tooltip("Stop the current training run", "O"))
        self._stop_button.clicked.connect(self._on_stop_training)
        self._stop_button.setVisible(False)
        layout.addWidget(self._stop_button)
        self._shortcut_bindings.append((self._stop_button, "O"))

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #F27C7C; font-size: 12px; background: transparent;")
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        self._progress_panel = ProgressPanel()
        self._progress_panel.setVisible(False)
        self._progress_panel.view_in_library_requested.connect(self._on_view_library_requested)
        layout.addWidget(self._progress_panel)

        layout.addStretch(1)
        return container

    def _build_model_settings_section(self) -> QFrame:
        section, layout = self._make_section_frame("Model Settings")

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self._model_name_edit = QLineEdit()
        self._model_name_edit.setPlaceholderText("e.g. Site-A Run 1")
        self._model_name_edit.textChanged.connect(lambda _text: self._update_start_button_state())
        self._add_form_row(form, "Model Name", self._model_name_edit)

        layout.addLayout(form)

        section.setStyleSheet(
            f"""
            QFrame {{ background-color: {SURFACE_COLOR}; border-radius: 8px; }}
            QLineEdit {{
                background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 6px 8px;
            }}
            """
        )
        return section

    def _build_hyperparams_section(self) -> QFrame:
        section = QFrame()

        outer = QVBoxLayout(section)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(12)

        header_row = QHBoxLayout()
        header_row.setSpacing(6)

        title = QLabel("Hyperparameters")
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 600; background: transparent;")
        header_row.addWidget(title)

        self._info_button = QPushButton()
        self._info_button.setIcon(_build_info_icon(TEXT_SECONDARY))
        self._info_button.setIconSize(QSize(14, 14))
        self._info_button.setFixedSize(22, 22)
        self._info_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._info_button.setToolTip("What do these parameters mean?")
        self._info_button.setStyleSheet(
            "QPushButton { background: transparent; border: none; border-radius: 11px; }"
            "QPushButton:hover { background-color: rgba(139, 144, 154, 40); }"
        )
        self._info_button.clicked.connect(self._toggle_hyperparam_info)
        header_row.addWidget(self._info_button)
        header_row.addStretch(1)
        outer.addLayout(header_row)

        self._info_label = QLabel(HYPERPARAM_INFO_TEXT)
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; background-color: {BACKGROUND_COLOR};"
            f"border-radius: 6px; padding: 10px;"
        )
        self._info_label.setVisible(False)
        outer.addWidget(self._info_label)

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 200)
        self._epochs_spin.setValue(30)
        self._epochs_spin.setToolTip("How many times the model sees your full dataset.")
        self._add_form_row(form, "Epochs (1\u2013200)", self._epochs_spin)

        self._batch_size_combo = QComboBox()
        self._batch_size_combo.addItems([str(v) for v in BATCH_SIZE_OPTIONS])
        self._batch_size_combo.setCurrentText("32")
        self._batch_size_combo.setToolTip("How many images processed at once. 32 works well for most computers.")
        self._add_form_row(form, "Batch Size", self._batch_size_combo)

        self._lr_combo = QComboBox()
        self._lr_combo.addItems([str(v) for v in LEARNING_RATE_OPTIONS])
        self._lr_combo.setCurrentText("0.001")
        self._lr_combo.setToolTip("How fast the model adjusts. 0.001 is a safe starting point.")
        self._add_form_row(form, "Learning Rate", self._lr_combo)

        self._optimizer_combo = QComboBox()
        self._optimizer_combo.addItems(list(OPTIMIZER_OPTIONS))
        self._optimizer_combo.setCurrentText("Adam")
        self._optimizer_combo.setToolTip("Adam works well for most cases.")
        self._add_form_row(form, "Optimizer", self._optimizer_combo)

        self._weight_decay_spin = QDoubleSpinBox()
        self._weight_decay_spin.setRange(0.0, 0.1)
        self._weight_decay_spin.setSingleStep(0.0001)
        self._weight_decay_spin.setDecimals(4)
        self._weight_decay_spin.setValue(0.0001)
        self._weight_decay_spin.setToolTip(
            "Helps prevent the model from memorising your training images. Leave at default."
        )
        self._add_form_row(form, "Weight Decay (0\u20130.1)", self._weight_decay_spin)

        val_split_row = QWidget()
        val_split_layout = QHBoxLayout(val_split_row)
        val_split_layout.setContentsMargins(0, 0, 0, 0)
        val_split_layout.setSpacing(10)
        self._val_split_slider = QSlider(Qt.Orientation.Horizontal)
        self._val_split_slider.setRange(10, 40)
        self._val_split_slider.setValue(20)
        self._val_split_slider.setToolTip("Images held back to test the model during training. 20% is recommended.")
        self._val_split_label = QLabel("20%")
        self._val_split_label.setFixedWidth(36)
        self._val_split_label.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent;")
        self._val_split_slider.valueChanged.connect(self._on_val_split_changed)
        val_split_layout.addWidget(self._val_split_slider, 1)
        val_split_layout.addWidget(self._val_split_label)
        self._add_form_row(form, "Validation Split (10\u201340%)", val_split_row)

        self._patience_spin = QSpinBox()
        self._patience_spin.setRange(1, 20)
        self._patience_spin.setValue(5)
        self._patience_spin.setToolTip("Training stops if results stop improving for this many epochs in a row.")
        self._add_form_row(form, "Early Stopping Patience (1\u201320)", self._patience_spin)

        outer.addLayout(form)

        self._pretrained_checkbox = QCheckBox("Use Pretrained Backbone")
        self._pretrained_checkbox.setChecked(True)
        self._pretrained_checkbox.setToolTip(
            "Starts from a model pre-trained on millions of images. Almost always improves results."
        )
        outer.addWidget(self._pretrained_checkbox)

        self._normalise_checkbox = QCheckBox("Normalise Targets")
        self._normalise_checkbox.setChecked(True)
        self._normalise_checkbox.setToolTip(
            "Scales Water, Solids, and Bitumen to a common range so the model learns all three equally."
        )
        outer.addWidget(self._normalise_checkbox)

        section.setStyleSheet(
            f"""
            QFrame {{ background-color: {SURFACE_COLOR}; border-radius: 8px; }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 6px 8px;
            }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QCheckBox {{ color: {TEXT_PRIMARY}; font-size: 12px; background: transparent; }}
            QSlider::groove:horizontal {{ background: {BACKGROUND_COLOR}; height: 6px; border-radius: 3px; }}
            QSlider::sub-page:horizontal {{ background: {ACCENT_COLOR}; border-radius: 3px; }}
            QSlider::handle:horizontal {{
                background: {ACCENT_COLOR}; width: 14px; margin: -5px 0; border-radius: 7px;
            }}
            """
        )

        return section

    def _add_form_row(self, form: QFormLayout, label_text: str, field: QWidget) -> None:
        label = QLabel(label_text)
        label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; background: transparent;")
        form.addRow(label, field)

    def _toggle_hyperparam_info(self) -> None:
        self._info_label.setVisible(not self._info_label.isVisible())

    # -- CSV / folder / dataset summary --------------------------------------

    def showEvent(self, event) -> None:  # noqa: D401 - Qt override
        super().showEvent(event)
        bind_page_shortcuts(self._shortcut_bindings)
        if not self._tab_order_applied:
            self._apply_tab_order()
            self._tab_order_applied = True

    def hideEvent(self, event) -> None:  # noqa: D401 - Qt override
        super().hideEvent(event)
        unbind_page_shortcuts(self._shortcut_bindings)

    def _on_csv_selected(self, path: str) -> None:
        try:
            df = read_labels_file(path)
        except Exception as exc:  # noqa: BLE001 - surface any parse failure to the user
            self._show_status_error(f"Could not read label file: {exc}")
            return

        self._csv_path = path
        missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]

        if missing:
            self._csv_dataframe = None
            self._csv_loaded_label.setVisible(False)
            self._preview_table.setVisible(False)
            self._column_mapping_frame.setVisible(False)
            self._csv_error_label.setText(
                f"Missing columns: {', '.join(missing)}.\nExpected: {', '.join(EXPECTED_COLUMNS)}"
            )
            self._csv_error_banner.setVisible(True)
        else:
            self._csv_dataframe = df
            self._csv_error_banner.setVisible(False)
            self._csv_loaded_label.setText(f"Loaded \u201c{Path(path).name}\u201d \u2014 {len(df)} row(s).")
            self._csv_loaded_label.setVisible(True)
            self._populate_preview_table(df)
            self._preview_table.setVisible(True)
            self._column_mapping_frame.setVisible(True)

        self._maybe_rebuild_dataset_summary()
        self._update_start_button_state()

    def _populate_preview_table(self, df: pd.DataFrame) -> None:
        preview = df.head(10)
        self._preview_table.clear()
        self._preview_table.setColumnCount(len(preview.columns))
        self._preview_table.setRowCount(len(preview))
        self._preview_table.setHorizontalHeaderLabels([str(column) for column in preview.columns])

        for row_index in range(len(preview)):
            for col_index, column in enumerate(preview.columns):
                value = preview.iloc[row_index, col_index]
                text = "" if pd.isna(value) else str(value)
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._preview_table.setItem(row_index, col_index, item)

        self._preview_table.resizeColumnsToContents()

    def _on_select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return

        try:
            image_count = sum(
                1
                for entry in Path(folder).iterdir()
                if entry.is_file() and entry.suffix.lower() in RegressionDataset.EXTENSION_CANDIDATES
            )
        except OSError as exc:
            self._show_status_error(f"Could not read folder: {exc}")
            return

        self._image_dir = folder
        self._folder_path_label.setText(folder)
        self._folder_path_label.setVisible(True)
        count_word = "image" if image_count == 1 else "images"
        self._folder_count_label.setText(f"{image_count} {count_word} found in this folder.")
        self._folder_count_label.setVisible(True)

        self._maybe_rebuild_dataset_summary()
        self._update_start_button_state()

    def _on_val_split_changed(self, value: int) -> None:
        self._val_split_label.setText(f"{value}%")
        self._rebuild_timer.start(VAL_SPLIT_REBUILD_DEBOUNCE_MS)

    def _maybe_rebuild_dataset_summary(self) -> None:
        """Re-run filename matching/splitting and refresh the Step 2/3 summaries.

        Called after the CSV or folder is (re)selected, and again (debounced)
        whenever the Validation Split slider moves, so the displayed
        train/val counts always reflect the current split percentage.
        """
        if self._csv_dataframe is None or not self._image_dir:
            self._match_card.setVisible(False)
            self._step3_frame.setVisible(False)
            self._train_dataset = None
            self._val_dataset = None
            return

        val_fraction = self._val_split_slider.value() / 100.0
        try:
            train_dataset = RegressionDataset(
                self._csv_path, self._image_dir, split="train", val_fraction=val_fraction, normalise=True, seed=42
            )
            val_dataset = RegressionDataset(
                self._csv_path, self._image_dir, split="val", val_fraction=val_fraction, normalise=True, seed=42
            )
        except Exception as exc:  # noqa: BLE001 - matching can fail in many ways (bad path, bad CSV, etc.)
            self._show_status_error(f"Could not match images: {exc}")
            self._match_card.setVisible(False)
            self._step3_frame.setVisible(False)
            self._train_dataset = None
            self._val_dataset = None
            return

        self._clear_status()
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset

        match_summary = train_dataset.get_match_summary()
        self._match_card.update_summary(match_summary)
        self._match_card.setVisible(True)

        if match_summary["matched"] > 0:
            output_ranges = self._compute_full_output_ranges(train_dataset.matched)
            pan_distribution = self._compute_full_pan_distribution(train_dataset.matched)
            self._dataset_summary_card.update_summary(
                total=match_summary["matched"],
                train_count=len(train_dataset),
                val_count=len(val_dataset),
                val_fraction=val_fraction,
                output_ranges=output_ranges,
                pan_distribution=pan_distribution,
            )
            self._step3_frame.setVisible(True)
        else:
            self._step3_frame.setVisible(False)

    @staticmethod
    def _compute_full_output_ranges(matched: List[Dict]) -> Dict[str, Dict[str, float]]:
        ranges: Dict[str, Dict[str, float]] = {}
        for key, label in (("water", "Water"), ("solids", "Solids"), ("bitumen", "Bitumen")):
            values = [item[key] for item in matched]
            if values:
                ranges[label] = {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}
            else:
                ranges[label] = {"min": 0.0, "max": 0.0, "mean": 0.0}
        return ranges

    @staticmethod
    def _compute_full_pan_distribution(matched: List[Dict]) -> Dict[int, int]:
        distribution: Dict[int, int] = {}
        for item in matched:
            pan = item["pan"]
            distribution[pan] = distribution.get(pan, 0) + 1
        return distribution

    # -- Start-button gating ---------------------------------------------------

    def _update_start_button_state(self) -> None:
        if self._thread is not None:
            return  # a run is in progress; _set_training_ui_active manages the button then.

        matched_count = self._train_dataset.get_match_summary()["matched"] if self._train_dataset else 0
        ready = (
            self._csv_dataframe is not None
            and self._image_dir is not None
            and matched_count > 0
            and bool(self._model_name_edit.text().strip())
        )
        self._start_button.setEnabled(ready)
        if ready:
            self._start_button.setToolTip(shortcut_tooltip("Start training the model", "S"))
        else:
            self._start_button.setToolTip("Complete Steps 1\u20133 first")

    # -- Status helpers ---------------------------------------------------

    def _clear_status(self) -> None:
        self._status_label.setText("")
        self._status_label.setVisible(False)

    def _show_status_error(self, message: str) -> None:
        self._status_label.setText(message)
        self._status_label.setVisible(True)

    # -- Training lifecycle ---------------------------------------------------

    def _on_start_training(self) -> None:
        if self._thread is not None:
            return

        self._clear_status()

        model_name = self._model_name_edit.text().strip()
        if not model_name:
            self._show_status_error("Please enter a model name.")
            return
        if any(character in model_name for character in ("/", "\\")):
            self._show_status_error("Model name cannot contain \u201c/\u201d or \u201c\\\u201d.")
            return

        if self._csv_dataframe is None or self._image_dir is None:
            self._show_status_error("Complete Steps 1\u20133 before starting training.")
            return

        val_fraction = self._val_split_slider.value() / 100.0
        normalise_targets = self._normalise_checkbox.isChecked()

        try:
            train_dataset = RegressionDataset(
                self._csv_path,
                self._image_dir,
                split="train",
                val_fraction=val_fraction,
                normalise=normalise_targets,
                seed=42,
            )
            val_dataset = RegressionDataset(
                self._csv_path,
                self._image_dir,
                split="val",
                val_fraction=val_fraction,
                normalise=normalise_targets,
                seed=42,
            )
        except Exception as exc:  # noqa: BLE001 - surface any dataset-build failure to the user
            self._show_status_error(f"Could not prepare dataset: {exc}")
            return

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            self._show_status_error(
                "Not enough matched images to train. Add more images, or check your CSV filenames."
            )
            return

        batch_size = int(self._batch_size_combo.currentText())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Stream images from disk via background worker processes rather than
        # holding every decoded image in RAM -- see _default_num_workers for
        # why this scales to very large datasets.
        num_workers = _default_num_workers()
        loader_kwargs = {"num_workers": num_workers}
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
        if device.type == "cuda":
            loader_kwargs["pin_memory"] = True

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

        model = BitumenRegressor(pretrained=self._pretrained_checkbox.isChecked())

        trainer = RegressionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=float(self._lr_combo.currentText()),
            num_epochs=self._epochs_spin.value(),
            optimizer_name=self._optimizer_combo.currentText(),
            weight_decay=self._weight_decay_spin.value(),
            output_stats=train_dataset.get_output_stats(),
            normalise_targets=normalise_targets,
            patience=self._patience_spin.value(),
        )

        self._pending_model_name = model_name
        self._launch_training_thread(trainer)

    def _launch_training_thread(self, trainer: RegressionTrainer) -> None:
        self._trainer = trainer
        self._thread = QThread(self)
        trainer.moveToThread(self._thread)

        self._thread.started.connect(trainer.run)
        trainer.progress.connect(self._on_epoch_progress)
        trainer.early_stopped.connect(self._on_early_stopped)
        trainer.finished.connect(self._on_training_finished)
        trainer.error.connect(self._on_training_error)
        trainer.finished.connect(self._thread.quit)
        trainer.error.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)

        self._set_training_ui_active(True)
        self._progress_panel.setVisible(True)
        self._progress_panel.reset(trainer.num_epochs, trainer.patience)

        self._thread.start()

    def _on_epoch_progress(
        self, epoch: int, train_loss: float, val_loss: float, val_mae_dict: dict, val_sum_deviation: float
    ) -> None:
        self._progress_panel.update_progress(epoch, train_loss, val_loss, val_mae_dict, val_sum_deviation)

    def _on_early_stopped(self, epoch: int) -> None:
        self._progress_panel.note_early_stopped(epoch)

    def _on_training_finished(self, result: RegressionTrainingResult) -> None:
        model = self._trainer.model
        try:
            paths = save_model(
                model=model,
                name=self._pending_model_name,
                output_stats=result.output_stats,
                result=result,
                save_dir=_MODELS_DIR,
            )
        except OSError as exc:
            self._progress_panel.append_log(f"Failed to save model: {exc}")
            self._show_status_error(f"Failed to save model: {exc}")
            self._set_training_ui_active(False)
            return

        if result.stopped_early:
            self._progress_panel.show_early_stopped_banner(result.final_epoch, result.best_val_mae)
        else:
            self._progress_panel.show_completion(self._pending_model_name, result.best_val_mae)

        if self.main_window is not None:
            try:
                metadata = load_model_metadata(paths["metadata_path"])
            except (OSError, ValueError):
                metadata = None
            self.main_window.set_active_model(str(paths["model_path"]), metadata)

        self._set_training_ui_active(False)

    def _on_training_error(self, message: str) -> None:
        self._progress_panel.append_log(f"Training failed: {message}")
        self._show_status_error(f"Training failed: {message}")
        self._set_training_ui_active(False)

    def _on_thread_finished(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._trainer = None

    def _on_stop_training(self) -> None:
        if self._trainer is not None:
            self._trainer.request_stop()
            self._progress_panel.append_log("Stop requested\u2026")
        self._stop_button.setEnabled(False)

    def _set_training_ui_active(self, active: bool) -> None:
        self._stop_button.setVisible(active)
        self._stop_button.setEnabled(active)

        if active:
            self._start_button.setEnabled(False)
            self._start_button.setToolTip("Training in progress\u2026")
        else:
            # Re-derive enabled/tooltip from current dataset state rather than
            # unconditionally re-enabling.
            self._update_start_button_state()

        for widget in (
            self._csv_drop_zone,
            self._select_folder_button,
            self._model_name_edit,
            self._epochs_spin,
            self._batch_size_combo,
            self._lr_combo,
            self._optimizer_combo,
            self._weight_decay_spin,
            self._val_split_slider,
            self._patience_spin,
            self._pretrained_checkbox,
            self._normalise_checkbox,
        ):
            if widget is not None:
                widget.setEnabled(not active)

    def _on_view_library_requested(self) -> None:
        if self.main_window is None:
            return
        stack = getattr(self.main_window, "_stack", None)
        sidebar = getattr(self.main_window, "_sidebar", None)
        pages = getattr(self.main_window, "_pages", [])

        target_index = next((i for i, page in enumerate(pages) if isinstance(page, ModelManagerPage)), None)
        if target_index is None or stack is None:
            return

        stack.setCurrentIndex(target_index)
        if sidebar is not None and hasattr(sidebar, "set_active_index"):
            sidebar.set_active_index(target_index)
