"""
Train page.

Load a labelled CSV, pick the image folder, choose architecture and split,
and run training. ``RegressionDataset`` handles matching/splitting; this page
just builds the summaries and wires ``RegressionTrainer`` to the progress panel.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd
import torch
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QDragEnterEvent,
    QDropEvent,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from torch.utils.data import DataLoader

from app.components.progress_panel import ProgressPanel
from app.constants import OUTPUT_NAMES
from app.ml.cnn_model import (
    ARCHITECTURE_LABELS,
    TRAINABLE_ARCHITECTURES,
    BitumenRegressor,
    IMAGE_SIZE,
)
from app.ml.dataset import RegressionDataset
from app.ml.trainer import RegressionTrainer, RegressionTrainingResult
from app.paths import MODELS_DIR
from app.theme import (
    ACCENT_COLOR,
    BACKGROUND_COLOR,
    BORDER_COLOR,
    DANGER_COLOR,
    PAGE_MARGINS,
    PAGE_SPACING,
    SUCCESS_COLOR,
    SURFACE_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    accent_button_qss,
    card_qss,
    danger_outline_button_qss,
    drop_zone_qss,
    secondary_button_qss,
)
from app.utils.data_io import SUPPORTED_EXTENSIONS as SUPPORTED_LABEL_EXTENSIONS
from app.utils.data_io import read_labels_file
from app.utils.image_utils import image_size_from_metadata, is_legacy_resnet18
from app.utils.model_io import list_saved_models, load_model_metadata, save_model
from app.utils.shortcuts import bind_page_shortcuts, shortcut_tooltip, unbind_page_shortcuts

if TYPE_CHECKING:
    from app.main_window import MainWindow

# Edited imports land here; prefer the original photo folder when sending to Train.
_EDIT_CACHE_DIR_NAME = "bitumengrader_edited_images"

LEFT_PANEL_WIDTH = 500
MAX_UNMATCHED_PREVIEW = 200
VAL_SPLIT_REBUILD_DEBOUNCE_MS = 300

EXPECTED_COLUMNS = list(RegressionDataset.EXPECTED_COLUMNS)  # ["Image", "Pan", "Water", "Solids", "Bitumen"]
OUTPUT_LABELS = OUTPUT_NAMES

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
OPTIMIZER_NAME = "Adam"
VAL_FRACTION = 0.20
TEST_FRACTION = 0.15
EARLY_STOPPING_PATIENCE = 10
SUM_PENALTY_WEIGHT = 0.10
TRANSFER_FREEZE_EPOCHS = 3


def _default_num_workers() -> int:
    """How many DataLoader workers to use when reading images from disk.

    Extra workers decode the next batch while training runs. On Windows,
    spawning them is slow in a desktop app, so we use 0 there.
    """
    import platform

    return 0 if platform.system() == "Windows" else 4


class _CsvDropZone(QFrame):
    """Dashed drop area for the training CSV, plus a Browse button."""

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

        title = QLabel("Drop your CSV or Excel file here")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600; background: transparent;")
        layout.addWidget(title)

        subtitle = QLabel(".csv, .txt, .xlsx, or .xls")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        layout.addWidget(subtitle)

        self.browse_button = QPushButton("Browse")
        self.browse_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_button.setFixedWidth(120)
        self.browse_button.setStyleSheet(accent_button_qss(extra="padding: 7px 14px;"))
        self.browse_button.clicked.connect(self._browse_file)
        layout.addWidget(self.browse_button, 0, Qt.AlignmentFlag.AlignHCenter)

    def _apply_style(self, active: bool) -> None:
        self.setStyleSheet(drop_zone_qss("csvDropZone", active=active))

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
    """Match count plus expandable unmatched / invalid-row lists."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet(card_qss(inset=True))

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
                "Check that CSV filenames match your image files. Matching works "
                "with or without the file extension."
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
                f"{self._invalid_count} {word} skipped \u2014 couldn't read Water/Solids/Bitumen/Pan."
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
    """Sample counts, output ranges, and pan-grade bars."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet(card_qss(inset=True))

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

        pan_title = QLabel("Pan grades:")
        pan_title.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; font-weight: 600; background: transparent;"
        )
        layout.addWidget(pan_title)

        self._pan_container = QVBoxLayout()
        self._pan_container.setSpacing(6)
        layout.addLayout(self._pan_container)

        self._campaigns_title = QLabel("Held-out campaigns:")
        self._campaigns_title.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; font-weight: 600; background: transparent;"
        )
        self._campaigns_title.setVisible(False)
        layout.addWidget(self._campaigns_title)

        self._campaigns_label = QLabel("")
        self._campaigns_label.setWordWrap(True)
        self._campaigns_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; background: transparent;")
        self._campaigns_label.setVisible(False)
        layout.addWidget(self._campaigns_label)

    def update_summary(
        self,
        total: int,
        train_count: int,
        val_count: int,
        test_count: int,
        val_fraction: float,
        test_fraction: float,
        output_ranges: Dict[str, Dict[str, float]],
        pan_distribution: Dict[int, int],
        split_mode: str = "random",
        split_campaigns: Optional[Dict[str, List[str]]] = None,
        split_fallback_reason: Optional[str] = None,
    ) -> None:
        train_pct = round((1 - val_fraction - test_fraction) * 100)
        val_pct = round(val_fraction * 100)
        test_pct = round(test_fraction * 100)
        self._counts_label.setText(
            f"Matched: {total}\n"
            f"Train: {train_count} ({train_pct}%)\n"
            f"Validation: {val_count} ({val_pct}%)\n"
            f"Test: {test_count} ({test_pct}%)"
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

        show_campaigns = split_mode == "experiment" or bool(split_fallback_reason)
        if show_campaigns:
            campaigns = split_campaigns or {}
            lines = []
            for key, title in (("train", "Train"), ("val", "Val"), ("test", "Test")):
                names = campaigns.get(key) or []
                lines.append(f"{title}: {', '.join(names) if names else '—'}")
            if split_fallback_reason:
                lines.append(split_fallback_reason)
            self._campaigns_label.setText("\n".join(lines))
            self._campaigns_title.setVisible(True)
            self._campaigns_label.setVisible(True)
        else:
            self._campaigns_title.setVisible(False)
            self._campaigns_label.setVisible(False)


class TrainPage(QWidget):
    """Configure and run a training job.

    Self-contained: load CSV + image folder here, match/split via
    ``RegressionDataset``, train on a QThread, and stream progress into
    the embedded ``ProgressPanel``.
    """

    def __init__(self, main_window: Optional["MainWindow"] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = main_window

        self._csv_path: Optional[str] = None
        self._csv_dataframe: Optional[pd.DataFrame] = None
        self._image_dir: Optional[str] = None
        self._train_dataset: Optional[RegressionDataset] = None
        self._val_dataset: Optional[RegressionDataset] = None
        self._test_dataset: Optional[RegressionDataset] = None

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
        self._continue_checkbox: Optional[QCheckBox] = None
        self._continue_combo: Optional[QComboBox] = None
        self._architecture_combo: Optional[QComboBox] = None
        self._split_mode_combo: Optional[QComboBox] = None
        self._strategy_note: Optional[QLabel] = None
        self._epochs_spin: Optional[QSpinBox] = None
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
        self._pending_train_config: Optional[Dict] = None
        self._parent_model_meta: Optional[Dict] = None

        self._shortcut_bindings: List[tuple] = []
        self._tab_order_applied = False

        self._build_ui()
        self._update_start_button_state()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(*PAGE_MARGINS)
        root.setSpacing(PAGE_SPACING)

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
        """Focus order: left panel top-to-bottom, then right panel."""
        chain = [
            self._csv_drop_zone.browse_button if self._csv_drop_zone else None,
            self._select_folder_button,
            self._model_name_edit,
            self._continue_checkbox,
            self._continue_combo,
            self._architecture_combo,
            self._split_mode_combo,
            self._epochs_spin,
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
            "Load labelled froth photos and train a model for Water, Solids, and Bitumen. "
            "The baseline CNN (from scratch) is the default; ImageNet transfer is optional. "
            "You can also continue a saved model on a new dataset."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px;")
        header.addWidget(subtitle)

        return header

    def _make_section_frame(self, title: str):
        frame = QFrame()
        frame.setStyleSheet(card_qss())

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 600; background: transparent;"
        )
        layout.addWidget(title_label)

        return frame, layout

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
        section, layout = self._make_section_frame("Step 1 \u2014 Load labels")

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
        section, layout = self._make_section_frame("Step 2 \u2014 Image folder")

        self._select_folder_button = QPushButton("Select Folder")
        self._select_folder_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._select_folder_button.setStyleSheet(secondary_button_qss())
        self._select_folder_button.setToolTip(shortcut_tooltip("Pick the folder with your images", "F"))
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
        section, layout = self._make_section_frame("Step 3 \u2014 Dataset summary")
        self._dataset_summary_card = _DatasetSummaryCard()
        layout.addWidget(self._dataset_summary_card)
        return section

    # -- Right panel: model settings & training --------------------------------

    def _build_right_panel(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(self._build_model_settings_section())
        layout.addWidget(self._build_strategy_section())

        self._start_button = QPushButton("Start Training")
        self._start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_button.setFixedHeight(48)
        self._start_button.setStyleSheet(
            accent_button_qss(extra="font-weight: 700; font-size: 14px; padding: 0px;")
        )
        self._start_button.clicked.connect(self._on_start_training)
        layout.addWidget(self._start_button)
        self._shortcut_bindings.append((self._start_button, "S"))

        self._stop_button = QPushButton("Stop Training")
        self._stop_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_button.setFixedHeight(38)
        self._stop_button.setStyleSheet(danger_outline_button_qss())
        self._stop_button.setToolTip(shortcut_tooltip("Stop this training run", "O"))
        self._stop_button.clicked.connect(self._on_stop_training)
        self._stop_button.setVisible(False)
        layout.addWidget(self._stop_button)
        self._shortcut_bindings.append((self._stop_button, "O"))

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(f"color: {DANGER_COLOR}; font-size: 12px; background: transparent;")
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

        self._continue_checkbox = QCheckBox("Continue training a saved model")
        self._continue_checkbox.setToolTip(
            "Load an existing checkpoint and keep training it on the dataset below. "
            "Saves a new model; the original is left unchanged."
        )
        self._continue_checkbox.toggled.connect(self._on_continue_toggled)
        layout.addWidget(self._continue_checkbox)

        self._continue_combo = QComboBox()
        self._continue_combo.setToolTip("Which saved model to continue from.")
        self._continue_combo.currentIndexChanged.connect(self._on_continue_model_changed)
        self._continue_combo.setVisible(False)
        layout.addWidget(self._continue_combo)

        section.setStyleSheet(
            f"""
            QFrame {{ background-color: {SURFACE_COLOR}; border-radius: 8px; }}
            QLineEdit, QComboBox {{
                background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 6px 8px;
            }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QCheckBox {{ color: {TEXT_PRIMARY}; font-size: 12px; background: transparent; }}
            """
        )
        return section

    def _build_strategy_section(self) -> QFrame:
        section, layout = self._make_section_frame("Training")

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self._architecture_combo = QComboBox()
        for key in TRAINABLE_ARCHITECTURES:
            self._architecture_combo.addItem(ARCHITECTURE_LABELS[key], key)
        self._architecture_combo.setCurrentIndex(0)
        self._architecture_combo.setToolTip(
            "Baseline CNN from scratch is the most robust default for froth images."
        )
        self._architecture_combo.currentIndexChanged.connect(self._on_strategy_changed)
        self._add_form_row(form, "Architecture", self._architecture_combo)

        self._split_mode_combo = QComboBox()
        self._split_mode_combo.addItem("Random split", "random")
        self._split_mode_combo.addItem("Experiment hold-out", "experiment")
        self._split_mode_combo.setToolTip(
            "Random split interpolates within familiar runs. Experiment hold-out "
            "keeps entire flotation campaigns out of training."
        )
        self._split_mode_combo.currentIndexChanged.connect(self._on_split_mode_changed)
        self._add_form_row(form, "Split", self._split_mode_combo)

        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 200)
        self._epochs_spin.setValue(100)
        self._epochs_spin.setToolTip(
            "How many full passes over your data. Training stops sooner if validation stalls."
        )
        self._add_form_row(form, "Epochs", self._epochs_spin)

        layout.addLayout(form)

        self._strategy_note = QLabel("")
        self._strategy_note.setWordWrap(True)
        self._strategy_note.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; background-color: {BACKGROUND_COLOR};"
            f"border-radius: 6px; padding: 8px;"
        )
        layout.addWidget(self._strategy_note)

        section.setStyleSheet(
            f"""
            QFrame {{ background-color: {SURFACE_COLOR}; border-radius: 8px; }}
            QComboBox, QSpinBox {{
                background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 6px 8px;
            }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            """
        )
        self._on_strategy_changed()
        return section

    def _add_form_row(self, form: QFormLayout, label_text: str, field: QWidget) -> None:
        label = QLabel(label_text)
        label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; background: transparent;")
        form.addRow(label, field)

    def _current_architecture(self) -> str:
        if self._architecture_combo is None:
            return "baseline"
        data = self._architecture_combo.currentData()
        return str(data) if data else "baseline"

    def _current_adaptation(self) -> str:
        return "scratch" if self._current_architecture() == "baseline" else "ft"

    def _current_head(self) -> str:
        if self._parent_model_meta:
            return str(self._parent_model_meta.get("head") or "native")
        return "native"

    def _is_transfer_architecture(self, architecture: Optional[str] = None) -> bool:
        architecture = architecture or self._current_architecture()
        return architecture in {"resnet50", "vgg16", "resnet18"}


    def _current_split_mode(self) -> str:
        if self._split_mode_combo is None:
            return "random"
        data = self._split_mode_combo.currentData()
        return str(data) if data else "random"

    def _on_split_mode_changed(self, _index: int = 0) -> None:
        self._rebuild_timer.start(VAL_SPLIT_REBUILD_DEBOUNCE_MS)

    def _on_continue_toggled(self, checked: bool) -> None:
        if self._continue_combo is not None:
            self._continue_combo.setVisible(checked)
        if checked:
            self._refresh_continue_combo()
            self._on_continue_model_changed()
        else:
            self._parent_model_meta = None
            if self._architecture_combo is not None:
                self._architecture_combo.setEnabled(True)
                # Drop a legacy-only item if it was added for continue-training.
                self._restore_trainable_architectures()
        self._on_strategy_changed()
        self._rebuild_timer.start(VAL_SPLIT_REBUILD_DEBOUNCE_MS)

    def _restore_trainable_architectures(self) -> None:
        if self._architecture_combo is None:
            return
        current = self._current_architecture()
        self._architecture_combo.blockSignals(True)
        self._architecture_combo.clear()
        for key in TRAINABLE_ARCHITECTURES:
            self._architecture_combo.addItem(ARCHITECTURE_LABELS[key], key)
        index = self._architecture_combo.findData(current if current in TRAINABLE_ARCHITECTURES else "baseline")
        self._architecture_combo.setCurrentIndex(max(0, index))
        self._architecture_combo.blockSignals(False)

    def _refresh_continue_combo(self) -> None:
        if self._continue_combo is None:
            return
        previous_path = self._continue_combo.currentData()
        self._continue_combo.blockSignals(True)
        self._continue_combo.clear()
        try:
            models = list_saved_models(MODELS_DIR)
        except OSError:
            models = []
        if not models:
            self._continue_combo.addItem("No saved models yet", None)
        else:
            for entry in models:
                name = entry.get("name") or "Untitled"
                created = (entry.get("created_at") or "")[:10]
                label = f"{name} ({created})" if created else name
                self._continue_combo.addItem(label, entry)
        if previous_path:
            for index in range(self._continue_combo.count()):
                data = self._continue_combo.itemData(index)
                if isinstance(data, dict) and data.get("model_path") == previous_path:
                    self._continue_combo.setCurrentIndex(index)
                    break
        self._continue_combo.blockSignals(False)

    def _on_continue_model_changed(self, _index: int = 0) -> None:
        data = self._continue_combo.currentData() if self._continue_combo is not None else None
        if not isinstance(data, dict):
            self._parent_model_meta = None
            self._on_strategy_changed()
            return
        self._parent_model_meta = data
        architecture = data.get("architecture", "resnet18")
        if self._architecture_combo is not None:
            self._architecture_combo.blockSignals(True)
            if self._architecture_combo.findData(architecture) < 0:
                self._architecture_combo.addItem(
                    ARCHITECTURE_LABELS.get(architecture, architecture), architecture
                )
            index = self._architecture_combo.findData(architecture)
            if index >= 0:
                self._architecture_combo.setCurrentIndex(index)
            self._architecture_combo.setEnabled(False)
            self._architecture_combo.blockSignals(False)
        if self._model_name_edit is not None and not self._model_name_edit.text().strip():
            base = data.get("name") or "model"
            self._model_name_edit.setText(f"{base} retrained")
        self._on_strategy_changed()
        self._rebuild_timer.start(VAL_SPLIT_REBUILD_DEBOUNCE_MS)

    def prepare_retrain(self, metadata: Dict) -> None:
        """Pre-select a library model so the user can continue it on a new dataset."""
        if self._continue_checkbox is None:
            return
        self._refresh_continue_combo()
        self._continue_checkbox.setChecked(True)
        target_path = metadata.get("model_path")
        if self._continue_combo is not None and target_path:
            for index in range(self._continue_combo.count()):
                data = self._continue_combo.itemData(index)
                if isinstance(data, dict) and data.get("model_path") == target_path:
                    self._continue_combo.setCurrentIndex(index)
                    break
            else:
                self._continue_combo.addItem(metadata.get("name") or "Saved model", metadata)
                self._continue_combo.setCurrentIndex(self._continue_combo.count() - 1)
        base = metadata.get("name") or "model"
        if self._model_name_edit is not None:
            self._model_name_edit.setText(f"{base} retrained")
        self._on_continue_model_changed()

    def _on_strategy_changed(self, _index: int = 0) -> None:
        is_baseline = self._current_architecture() == "baseline"
        continuing = bool(self._continue_checkbox is not None and self._continue_checkbox.isChecked())

        if self._strategy_note is not None:
            if continuing:
                parent = (self._parent_model_meta or {}).get("name") or "the selected model"
                self._strategy_note.setText(
                    f"Continuing \u201c{parent}\u201d on the dataset below. Architecture "
                    "stays locked so weights load correctly. A new model file is saved; the original is kept."
                )
            elif is_baseline:
                self._strategy_note.setText(
                    "Baseline CNN trained from scratch is the most robust default for froth images. "
                    "Prefer an experiment hold-out split before relying on a model in a new campaign."
                )
            else:
                self._strategy_note.setText(
                    "ImageNet transfer is optional and mainly helps Solids. Fine-tuning is applied "
                    "automatically. Compare against the baseline under an experiment hold-out before deploying."
                )

    # -- CSV / folder / dataset summary --------------------------------------

    def showEvent(self, event) -> None:  # noqa: D401 - Qt override
        super().showEvent(event)
        bind_page_shortcuts(self._shortcut_bindings)
        if not self._tab_order_applied:
            self._apply_tab_order()
            self._tab_order_applied = True
        if self._continue_checkbox is not None and self._continue_checkbox.isChecked():
            self._refresh_continue_combo()
        self._sync_imported_images()

    def hideEvent(self, event) -> None:  # noqa: D401 - Qt override
        super().hideEvent(event)
        unbind_page_shortcuts(self._shortcut_bindings)

    def _on_csv_selected(self, path: str) -> None:
        try:
            df = read_labels_file(path)
        except Exception as exc:  # noqa: BLE001 - surface any parse failure to the user
            self._show_status_error(f"Couldn't read label file: {exc}")
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
        self._apply_image_folder(folder)

    def _apply_image_folder(self, folder: str) -> bool:
        """Set Step 2 from a folder path. Returns False if the folder can't be read."""
        try:
            image_count = sum(
                1
                for entry in Path(folder).iterdir()
                if entry.is_file() and entry.suffix.lower() in RegressionDataset.EXTENSION_CANDIDATES
            )
        except OSError as exc:
            self._show_status_error(f"Couldn't read folder: {exc}")
            return False

        self._image_dir = folder
        self._folder_path_label.setText(folder)
        self._folder_path_label.setVisible(True)
        count_word = "image" if image_count == 1 else "images"
        self._folder_count_label.setText(f"{image_count} {count_word} in this folder.")
        self._folder_count_label.setVisible(True)

        self._maybe_rebuild_dataset_summary()
        self._update_start_button_state()
        return True

    @staticmethod
    def _folder_from_imported_paths(paths: List[str]) -> Optional[str]:
        """Pick the image folder to use after Import → Send to Training.

        Uses the most common parent directory, ignoring the edit-cache folder
        when originals are still present.
        """
        parents: List[str] = []
        for path in paths:
            parent = Path(path).expanduser().resolve().parent
            if parent.is_dir():
                parents.append(str(parent))
        if not parents:
            return None
        counts = Counter(parents)
        preferred = {folder: count for folder, count in counts.items() if Path(folder).name != _EDIT_CACHE_DIR_NAME}
        pool = preferred or dict(counts)
        return max(pool, key=pool.get)

    def _sync_imported_images(self) -> None:
        """Apply the folder from Import's Send to Training payload, once."""
        if self.main_window is None:
            return
        incoming = getattr(self.main_window, "training_images", None)
        if not incoming:
            return
        self.main_window.training_images = None

        if self._thread is not None:
            self._show_status_error("Training is already running, so the imported folder wasn't applied.")
            return

        paths = [str(entry.get("path")) for entry in incoming if entry.get("path")]
        folder = self._folder_from_imported_paths(paths)
        if not folder:
            self._show_status_error("Couldn't use the imported images — no readable folder.")
            return
        self._apply_image_folder(folder)

    def _dataset_kwargs(self, *, normalise: bool = True) -> Dict:
        image_size = IMAGE_SIZE
        legacy_crop = False
        if self._continue_checkbox is not None and self._continue_checkbox.isChecked() and self._parent_model_meta:
            image_size = image_size_from_metadata(self._parent_model_meta)
            legacy_crop = is_legacy_resnet18(self._parent_model_meta)
        return {
            "csv_path": self._csv_path,
            "image_dir": self._image_dir,
            "val_fraction": VAL_FRACTION,
            "test_fraction": TEST_FRACTION,
            "normalise": normalise,
            "seed": 42,
            "split_mode": self._current_split_mode(),
            "image_size": image_size,
            "legacy_crop": legacy_crop,
        }

    def _maybe_rebuild_dataset_summary(self) -> None:
        """Re-match filenames and refresh Step 2/3 after CSV/folder/split changes."""
        if self._csv_dataframe is None or not self._image_dir:
            self._match_card.setVisible(False)
            self._step3_frame.setVisible(False)
            self._train_dataset = None
            self._val_dataset = None
            self._test_dataset = None
            return

        kwargs = self._dataset_kwargs(normalise=True)
        try:
            train_dataset = RegressionDataset(split="train", **kwargs)
            val_dataset = RegressionDataset(split="val", **kwargs)
            test_dataset = RegressionDataset(split="test", **kwargs)
        except Exception as exc:  # noqa: BLE001 - matching can fail in many ways (bad path, bad CSV, etc.)
            self._show_status_error(f"Couldn't match images: {exc}")
            self._match_card.setVisible(False)
            self._step3_frame.setVisible(False)
            self._train_dataset = None
            self._val_dataset = None
            self._test_dataset = None
            return

        self._clear_status()
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._test_dataset = test_dataset

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
                test_count=len(test_dataset),
                val_fraction=kwargs["val_fraction"],
                test_fraction=kwargs["test_fraction"],
                output_ranges=output_ranges,
                pan_distribution=pan_distribution,
                split_mode=train_dataset.split_mode,
                split_campaigns=train_dataset.split_campaigns,
                split_fallback_reason=train_dataset.split_fallback_reason,
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
            self._start_button.setToolTip(shortcut_tooltip("Start training", "S"))
        else:
            self._start_button.setToolTip("Finish Steps 1\u20133 first")

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
            self._show_status_error("Enter a model name.")
            return
        if any(character in model_name for character in ("/", "\\")):
            self._show_status_error("Model name can't contain \u201c/\u201d or \u201c\\\u201d.")
            return

        if self._csv_dataframe is None or self._image_dir is None:
            self._show_status_error("Finish Steps 1\u20133 before starting.")
            return

        continuing = bool(self._continue_checkbox is not None and self._continue_checkbox.isChecked())
        parent_meta = self._parent_model_meta if continuing else None
        if continuing and not isinstance(parent_meta, dict):
            self._show_status_error("Choose a saved model to continue training.")
            return
        if continuing:
            parent_path = parent_meta.get("model_path")
            if not parent_path or not Path(parent_path).exists():
                self._show_status_error("Couldn't find the saved model weights to continue from.")
                return

        kwargs = self._dataset_kwargs(normalise=True)

        try:
            train_dataset = RegressionDataset(split="train", **kwargs)
            val_dataset = RegressionDataset(split="val", **kwargs)
            test_dataset = RegressionDataset(split="test", **kwargs)
        except Exception as exc:  # noqa: BLE001 - surface any dataset-build failure to the user
            self._show_status_error(f"Couldn't prepare dataset: {exc}")
            return

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            self._show_status_error(
                "Not enough matched images. Add more, or check CSV filenames."
            )
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load images from disk in worker processes instead of keeping
        # everything in RAM — see _default_num_workers.
        num_workers = _default_num_workers()
        loader_kwargs = {"num_workers": num_workers}
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
        if device.type == "cuda":
            loader_kwargs["pin_memory"] = True

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
        test_loader = None
        if len(test_dataset) > 0:
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

        architecture = self._current_architecture()
        head = self._current_head()
        adaptation = self._current_adaptation()
        pretrained = self._is_transfer_architecture(architecture) and not continuing

        try:
            if continuing:
                architecture = parent_meta.get("architecture", architecture)
                head = parent_meta.get("head", head)
                adaptation = "scratch" if architecture == "baseline" else "ft"
                model = BitumenRegressor.from_checkpoint(parent_meta["model_path"], parent_meta, device)
                model.train()
            else:
                model = BitumenRegressor(architecture=architecture, pretrained=pretrained, head=head)
        except Exception as exc:  # noqa: BLE001 - architecture / checkpoint mismatches
            self._show_status_error(f"Couldn't load model: {exc}")
            return

        freeze_epochs = 0 if architecture == "baseline" else TRANSFER_FREEZE_EPOCHS

        trainer = RegressionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=LEARNING_RATE,
            num_epochs=self._epochs_spin.value(),
            optimizer_name=OPTIMIZER_NAME,
            weight_decay=WEIGHT_DECAY,
            output_stats=train_dataset.get_output_stats(),
            normalise_targets=True,
            patience=EARLY_STOPPING_PATIENCE,
            test_loader=test_loader,
            use_differential_lrs=self._is_transfer_architecture(architecture),
            use_cosine_schedule=True,
            freeze_backbone_epochs=freeze_epochs,
            sum_penalty_weight=SUM_PENALTY_WEIGHT,
            adaptation=adaptation,
        )

        image_size = kwargs["image_size"]
        self._pending_model_name = model_name
        self._pending_train_config = {
            "architecture": architecture,
            "head": head,
            "adaptation": adaptation,
            "pretrained": pretrained,
            "image_size": image_size,
            "split_mode": train_dataset.split_mode,
            "continued_training": continuing,
            "parent_model": (parent_meta or {}).get("name") if continuing else None,
            "parent_model_path": (parent_meta or {}).get("model_path") if continuing else None,
        }
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
                save_dir=MODELS_DIR,
                extra_metadata=self._pending_train_config,
            )
        except OSError as exc:
            self._progress_panel.append_log(f"Couldn't save model: {exc}")
            self._show_status_error(f"Couldn't save model: {exc}")
            self._set_training_ui_active(False)
            return

        if result.stopped_early:
            self._progress_panel.show_early_stopped_banner(
                result.final_epoch, result.best_val_mae, test_mae=result.test_mae
            )
        else:
            self._progress_panel.show_completion(
                self._pending_model_name, result.best_val_mae, test_mae=result.test_mae
            )

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
            self._progress_panel.append_log("Stopping\u2026")
        self._stop_button.setEnabled(False)

    def _set_training_ui_active(self, active: bool) -> None:
        self._stop_button.setVisible(active)
        self._stop_button.setEnabled(active)

        if active:
            self._start_button.setEnabled(False)
            self._start_button.setToolTip("Training\u2026")
        else:
            # Re-check enabled/tooltip from current dataset state.
            self._update_start_button_state()
            if self._continue_checkbox is not None and self._continue_checkbox.isChecked():
                self._on_continue_model_changed()

        for widget in (
            self._csv_drop_zone,
            self._select_folder_button,
            self._model_name_edit,
            self._continue_checkbox,
            self._continue_combo,
            self._architecture_combo,
            self._split_mode_combo,
            self._epochs_spin,
        ):
            if widget is not None:
                widget.setEnabled(not active)

    def _on_view_library_requested(self) -> None:
        if self.main_window is not None:
            self.main_window.navigate_to("library")
