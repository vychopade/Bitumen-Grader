"""
Model Training page.

Provides the UI for configuring and launching CNN training runs: selecting
the training dataset, assigning grade labels, setting hyperparameters, and
running training on a background QThread while streaming live progress into
an embedded ProgressPanel.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from PIL import Image
from PyQt6.QtCore import QObject, QPoint, QPointF, QRectF, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from app.components.progress_panel import ProgressPanel
from app.ml.cnn_model import DEFAULT_GRADE_LABELS, BitumenCNN
from app.ml.trainer import ModelTrainer, TrainingResult
from app.pages.model_manager_page import ModelManagerPage
from app.utils.image_utils import preprocess_for_inference, preprocess_for_training
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

LEFT_PANEL_WIDTH = 400
MIN_GRADES = 2
MAX_GRADES = 10
MIN_TRAINING_IMAGES = 4
PLACEHOLDER_GRADE = "\u2014 Select Grade \u2014"
BULK_ASSIGN_PLACEHOLDER = "Assign grade to selected\u2026"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif")
BULK_STATUS_TIMEOUT_MS = 4000

BATCH_SIZE_OPTIONS = (8, 16, 32, 64)
LEARNING_RATE_OPTIONS = (0.0001, 0.001, 0.01)
OPTIMIZER_OPTIONS = ("Adam", "SGD")

_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

HYPERPARAM_INFO_TEXT = (
    "Epochs: number of full passes through the training set.\n"
    "Batch Size: images processed together per training step.\n"
    "Learning Rate: how much weights are adjusted per step.\n"
    "Optimizer: Adam adapts automatically; SGD is simpler but may need tuning.\n"
    "Weight Decay: L2 regularization strength that reduces overfitting.\n"
    "Validation Split: fraction of images held out to measure performance.\n"
    "Pretrained Backbone: start from ImageNet weights instead of random init."
)


def _build_info_icon(color: str, size: int = 14) -> QIcon:
    """Draw a small "info circle" icon (avoids relying on the \u24d8 glyph's font coverage)."""
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


class _ImageGradeDataset(Dataset):
    """Wraps ``(PIL.Image, label_index)`` pairs, applying train/inference preprocessing."""

    def __init__(self, samples: List[Tuple[Image.Image, int]], train: bool):
        self._samples = samples
        self._train = train

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        image, label = self._samples[index]
        tensor = preprocess_for_training(image) if self._train else preprocess_for_inference(image)
        return tensor, label


class _TrainingWorker(QObject):
    """Runs ``ModelTrainer.train()`` on a background QThread and reports the outcome."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, trainer: ModelTrainer):
        super().__init__()
        self._trainer = trainer

    def run(self) -> None:
        try:
            result = self._trainer.train()
        except Exception as exc:  # noqa: BLE001 - surface any training failure to the UI
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


@dataclass(eq=False)
class _TrainingImageRow:
    """A single training image plus the table-cell widgets representing it."""

    path: str
    image: Image.Image
    combo: QComboBox
    remove_button: QPushButton


class _FolderImportDialog(QDialog):
    """Small modal asking whether to bulk-label every image found in an imported folder."""

    def __init__(self, grade_labels: List[str], folder_name: str, image_count: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Import Folder")
        self.setModal(True)
        self.setFixedWidth(360)
        self.setStyleSheet(f"QDialog {{ background-color: {SURFACE_COLOR}; }}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 18)
        layout.setSpacing(14)

        count_word = "image" if image_count == 1 else "images"
        message = QLabel(
            f'Found {image_count} {count_word} in \u201c{folder_name}\u201d.\n\n'
            f"Assign a grade to all images in this folder?"
        )
        message.setWordWrap(True)
        message.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; background: transparent;")
        layout.addWidget(message)

        self.grade_combo = QComboBox()
        self.grade_combo.addItems(grade_labels)
        self.grade_combo.setStyleSheet(
            f"QComboBox {{ background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};"
            f"border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 6px 8px; }}"
        )
        layout.addWidget(self.grade_combo)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)

        skip_button = QPushButton("Skip")
        skip_button.setCursor(Qt.CursorShape.PointingHandCursor)
        skip_button.setStyleSheet(
            f"QPushButton {{ background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};"
            f"border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 8px 14px; }}"
            f"QPushButton:hover {{ background-color: #2A2E36; }}"
        )
        skip_button.clicked.connect(self.reject)
        button_row.addWidget(skip_button)

        confirm_button = QPushButton("Confirm")
        confirm_button.setCursor(Qt.CursorShape.PointingHandCursor)
        confirm_button.setStyleSheet(
            f"QPushButton {{ background-color: {ACCENT_COLOR}; color: #13151A; font-weight: 700;"
            f"border: none; border-radius: 6px; padding: 8px 14px; }}"
            f"QPushButton:hover {{ background-color: {ACCENT_HOVER_COLOR}; }}"
        )
        confirm_button.clicked.connect(self.accept)
        confirm_button.setDefault(True)
        button_row.addWidget(confirm_button)

        layout.addLayout(button_row)

    def selected_grade(self) -> str:
        return self.grade_combo.currentText()


class TrainPage(QWidget):
    """Page for configuring and running CNN training jobs.

    Training images populate automatically from ``main_window.training_images``
    (written by ``ImageImportPage``'s "Send to Training" action) and can be
    supplemented via "Add More Images". Training itself runs on a QThread:
    the ``ModelTrainer`` (a QObject) is moved to that thread, and its
    ``progress_updated`` signal is connected to the ``ProgressPanel`` update
    slot, which Qt automatically marshals onto the main thread since the
    panel lives there.
    """

    def __init__(self, main_window: Optional["MainWindow"] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = main_window

        self._grade_labels: List[str] = list(DEFAULT_GRADE_LABELS)
        self._image_rows: List[_TrainingImageRow] = []

        self._table: Optional[QTableWidget] = None
        self._image_status_label: Optional[QLabel] = None
        self._add_images_button: Optional[QPushButton] = None
        self._import_folder_button: Optional[QPushButton] = None
        self._select_all_button: Optional[QPushButton] = None
        self._select_none_button: Optional[QPushButton] = None
        self._bulk_grade_combo: Optional[QComboBox] = None
        self._apply_bulk_button: Optional[QPushButton] = None
        self._bulk_status_label: Optional[QLabel] = None
        self._bulk_status_timer = QTimer(self)
        self._bulk_status_timer.setSingleShot(True)
        self._bulk_status_timer.timeout.connect(self._clear_bulk_status)
        self._grade_list: Optional[QListWidget] = None
        self._add_grade_button: Optional[QPushButton] = None
        self._remove_grade_button: Optional[QPushButton] = None

        self._info_label: Optional[QLabel] = None
        self._model_name_edit: Optional[QLineEdit] = None
        self._epochs_spin: Optional[QSpinBox] = None
        self._batch_size_combo: Optional[QComboBox] = None
        self._lr_combo: Optional[QComboBox] = None
        self._optimizer_combo: Optional[QComboBox] = None
        self._weight_decay_spin: Optional[QDoubleSpinBox] = None
        self._val_split_slider: Optional[QSlider] = None
        self._val_split_label: Optional[QLabel] = None
        self._pretrained_checkbox: Optional[QCheckBox] = None
        self._label_completion_label: Optional[QLabel] = None
        self._start_button: Optional[QPushButton] = None
        self._stop_button: Optional[QPushButton] = None
        self._status_label: Optional[QLabel] = None
        self._progress_panel: Optional[ProgressPanel] = None

        self._thread: Optional[QThread] = None
        self._worker: Optional[_TrainingWorker] = None
        self._trainer: Optional[ModelTrainer] = None
        self._pending_model_name: Optional[str] = None
        self._pending_grade_labels: List[str] = []
        self._pending_num_classes = 0
        self._shortcut_bindings: List[tuple] = []
        self._tab_order_applied = False

        self._build_ui()
        self._sync_from_main_window()

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
            self._select_all_button,
            self._select_none_button,
            self._bulk_grade_combo,
            self._apply_bulk_button,
            self._table,
            self._add_images_button,
            self._import_folder_button,
            self._grade_list,
            self._add_grade_button,
            self._remove_grade_button,
            self._info_button,
            self._model_name_edit,
            self._epochs_spin,
            self._batch_size_combo,
            self._lr_combo,
            self._optimizer_combo,
            self._weight_decay_spin,
            self._val_split_slider,
            self._pretrained_checkbox,
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

        subtitle = QLabel("Configure hyperparameters and train a CNN to classify bitumen grades.")
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px;")
        header.addWidget(subtitle)

        return header

    # -- Left panel: dataset & label setup ----------------------------------

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        container.setFixedWidth(LEFT_PANEL_WIDTH)
        container.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(self._build_training_images_section())
        layout.addWidget(self._build_grade_labels_section())
        layout.addStretch(1)

        return container

    def _make_section_frame(self, title: str) -> Tuple[QFrame, QVBoxLayout]:
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

    def _build_training_images_section(self) -> QFrame:
        section, layout = self._make_section_frame("Training Images")

        layout.addLayout(self._build_bulk_assign_toolbar())

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Filename", "Grade", ""])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setShowGrid(False)
        self._table.setFixedHeight(220)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_table_context_menu)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(1, 148)
        self._table.setColumnWidth(2, 68)
        self._table.setStyleSheet(
            f"""
            QTableWidget {{
                background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR}; border-radius: 6px; gridline-color: {BORDER_COLOR};
            }}
            QTableWidget::item {{ padding: 4px; }}
            QTableWidget::item:selected {{ background-color: rgba(232, 168, 56, 45); color: {TEXT_PRIMARY}; }}
            QHeaderView::section {{
                background-color: {SURFACE_COLOR}; color: {TEXT_SECONDARY}; border: none;
                padding: 6px; font-size: 11px; font-weight: 600;
            }}
            """
        )
        layout.addWidget(self._table)

        self._image_status_label = QLabel("No images loaded yet.")
        self._image_status_label.setWordWrap(True)
        self._image_status_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;"
        )
        layout.addWidget(self._image_status_label)

        self._bulk_status_label = QLabel("")
        self._bulk_status_label.setWordWrap(True)
        self._bulk_status_label.setStyleSheet(
            f"color: {SUCCESS_COLOR}; font-size: 11px; font-weight: 600; background: transparent;"
        )
        self._bulk_status_label.setVisible(False)
        layout.addWidget(self._bulk_status_label)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)

        self._add_images_button = QPushButton("Add More Images")
        self._add_images_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._add_images_button.setStyleSheet(self._secondary_button_style())
        self._add_images_button.setToolTip(shortcut_tooltip("Add more training images", "M"))
        self._add_images_button.clicked.connect(self._on_add_more_images)
        button_row.addWidget(self._add_images_button)
        self._shortcut_bindings.append((self._add_images_button, "M"))

        self._import_folder_button = QPushButton("Import Folder")
        self._import_folder_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._import_folder_button.setStyleSheet(self._secondary_button_style())
        self._import_folder_button.setToolTip(shortcut_tooltip("Import every image in a folder", "F"))
        self._import_folder_button.clicked.connect(self._on_import_folder)
        button_row.addWidget(self._import_folder_button)
        self._shortcut_bindings.append((self._import_folder_button, "F"))

        layout.addLayout(button_row)

        return section

    def _build_bulk_assign_toolbar(self) -> QVBoxLayout:
        toolbar = QVBoxLayout()
        toolbar.setSpacing(6)

        selection_row = QHBoxLayout()
        selection_row.setSpacing(6)

        compact_button_style = (
            f"QPushButton {{ background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};"
            f"border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 5px 10px; font-size: 11px; }}"
            f"QPushButton:hover {{ background-color: #2A2E36; }}"
            f"QPushButton:disabled {{ color: {TEXT_SECONDARY}; border: 1px solid #2A2D34; }}"
        )

        self._select_all_button = QPushButton("Select All")
        self._select_all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._select_all_button.setStyleSheet(compact_button_style)
        self._select_all_button.setToolTip(shortcut_tooltip("Select every image in the table", "L"))
        self._select_all_button.clicked.connect(self._on_select_all)
        selection_row.addWidget(self._select_all_button)
        self._shortcut_bindings.append((self._select_all_button, "L"))

        self._select_none_button = QPushButton("Select None")
        self._select_none_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._select_none_button.setStyleSheet(compact_button_style)
        self._select_none_button.setToolTip(shortcut_tooltip("Clear the table selection", "N"))
        self._select_none_button.clicked.connect(self._on_select_none)
        selection_row.addWidget(self._select_none_button)
        self._shortcut_bindings.append((self._select_none_button, "N"))
        selection_row.addStretch(1)

        toolbar.addLayout(selection_row)

        assign_row = QHBoxLayout()
        assign_row.setSpacing(6)

        self._bulk_grade_combo = QComboBox()
        self._bulk_grade_combo.addItem(BULK_ASSIGN_PLACEHOLDER)
        self._bulk_grade_combo.addItems(self._grade_labels)
        self._bulk_grade_combo.setStyleSheet(
            f"QComboBox {{ background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};"
            f"border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 4px 8px; font-size: 11px; }}"
        )
        assign_row.addWidget(self._bulk_grade_combo, 1)

        self._apply_bulk_button = QPushButton("Apply to Selected")
        self._apply_bulk_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_bulk_button.setStyleSheet(
            f"QPushButton {{ background-color: {ACCENT_COLOR}; color: #13151A; font-weight: 700;"
            f"border: none; border-radius: 6px; padding: 5px 10px; font-size: 11px; }}"
            f"QPushButton:hover {{ background-color: {ACCENT_HOVER_COLOR}; }}"
            f"QPushButton:disabled {{ background-color: #4A4230; color: #8B8168; }}"
        )
        self._apply_bulk_button.setToolTip(shortcut_tooltip("Assign the chosen grade to every selected image", "P"))
        self._apply_bulk_button.clicked.connect(self._on_apply_to_selected)
        assign_row.addWidget(self._apply_bulk_button)
        self._shortcut_bindings.append((self._apply_bulk_button, "P"))

        toolbar.addLayout(assign_row)

        return toolbar

    def _build_grade_labels_section(self) -> QFrame:
        section, layout = self._make_section_frame("Grade Labels")

        self._grade_list = QListWidget()
        self._grade_list.setFixedHeight(140)
        self._grade_list.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self._grade_list.setStyleSheet(
            f"""
            QListWidget {{
                background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR}; border-radius: 6px;
            }}
            QListWidget::item {{ padding: 6px 8px; }}
            QListWidget::item:selected {{ background-color: {ACCENT_COLOR}; color: #13151A; }}
            """
        )
        for label in self._grade_labels:
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self._grade_list.addItem(item)
        self._grade_list.itemChanged.connect(self._on_grade_item_changed)
        self._grade_list.itemSelectionChanged.connect(self._update_grade_buttons_enabled)
        layout.addWidget(self._grade_list)

        hint = QLabel("Double-click a grade to rename it (2\u201310 grades allowed).")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        layout.addWidget(hint)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)

        self._add_grade_button = QPushButton("+ Add Grade")
        self._add_grade_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._add_grade_button.setStyleSheet(self._secondary_button_style())
        self._add_grade_button.setToolTip(shortcut_tooltip("Add a new grade label", "A"))
        self._add_grade_button.clicked.connect(self._add_grade)
        button_row.addWidget(self._add_grade_button)
        self._shortcut_bindings.append((self._add_grade_button, "A"))

        self._remove_grade_button = QPushButton("\u2212 Remove Selected")
        self._remove_grade_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._remove_grade_button.setStyleSheet(self._secondary_button_style())
        self._remove_grade_button.setToolTip(shortcut_tooltip("Remove the selected grade label", "R"))
        self._remove_grade_button.clicked.connect(self._remove_selected_grade)
        button_row.addWidget(self._remove_grade_button)
        self._shortcut_bindings.append((self._remove_grade_button, "R"))

        layout.addLayout(button_row)

        self._update_grade_buttons_enabled()
        return section

    def _secondary_button_style(self) -> str:
        return (
            f"QPushButton {{ background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};"
            f"border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 8px 12px; font-size: 12px; }}"
            f"QPushButton:hover {{ background-color: #2A2E36; }}"
            f"QPushButton:disabled {{ color: {TEXT_SECONDARY}; border: 1px solid #2A2D34; }}"
        )

    # -- Right panel: hyperparameters & controls ----------------------------

    def _build_right_panel(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(self._build_hyperparams_section())

        self._label_completion_label = QLabel("0 of 0 images labelled")
        self._label_completion_label.setStyleSheet(
            f"color: {ACCENT_COLOR}; font-size: 12px; font-weight: 600; background: transparent;"
        )
        layout.addWidget(self._label_completion_label)

        self._start_button = QPushButton("Start Training")
        self._start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_button.setFixedHeight(48)
        self._start_button.setStyleSheet(
            f"QPushButton {{ background-color: {ACCENT_COLOR}; color: #13151A; font-weight: 700;"
            f"font-size: 14px; border: none; border-radius: 6px; }}"
            f"QPushButton:hover {{ background-color: {ACCENT_HOVER_COLOR}; }}"
            f"QPushButton:disabled {{ background-color: #4A4230; color: #8B8168; }}"
        )
        self._start_button.setToolTip(shortcut_tooltip("Start training the model", "S"))
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
        self._stop_button.setEnabled(False)
        self._stop_button.setToolTip(shortcut_tooltip("Stop the current training run", "O"))
        self._stop_button.clicked.connect(self._on_stop_training)
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

        info_button = QPushButton()
        info_button.setIcon(_build_info_icon(TEXT_SECONDARY))
        info_button.setIconSize(QSize(14, 14))
        info_button.setFixedSize(22, 22)
        info_button.setCursor(Qt.CursorShape.PointingHandCursor)
        info_button.setToolTip("What do these parameters mean?")
        info_button.setStyleSheet(
            "QPushButton { background: transparent; border: none; border-radius: 11px; }"
            "QPushButton:hover { background-color: rgba(139, 144, 154, 40); }"
        )
        info_button.clicked.connect(self._toggle_hyperparam_info)
        header_row.addWidget(info_button)
        header_row.addStretch(1)
        outer.addLayout(header_row)
        self._info_button = info_button

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

        self._model_name_edit = QLineEdit()
        self._model_name_edit.setPlaceholderText("e.g. Site-A Trial 1")
        self._add_form_row(form, "Model Name", self._model_name_edit)

        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 200)
        self._epochs_spin.setValue(25)
        self._add_form_row(form, "Epochs (1\u2013200)", self._epochs_spin)

        self._batch_size_combo = QComboBox()
        self._batch_size_combo.addItems([str(v) for v in BATCH_SIZE_OPTIONS])
        self._batch_size_combo.setCurrentText("16")
        self._add_form_row(form, "Batch Size", self._batch_size_combo)

        self._lr_combo = QComboBox()
        self._lr_combo.addItems([str(v) for v in LEARNING_RATE_OPTIONS])
        self._lr_combo.setCurrentText("0.001")
        self._add_form_row(form, "Learning Rate", self._lr_combo)

        self._optimizer_combo = QComboBox()
        self._optimizer_combo.addItems(list(OPTIMIZER_OPTIONS))
        self._optimizer_combo.setCurrentText("Adam")
        self._add_form_row(form, "Optimizer", self._optimizer_combo)

        self._weight_decay_spin = QDoubleSpinBox()
        self._weight_decay_spin.setRange(0.0, 0.1)
        self._weight_decay_spin.setSingleStep(0.0001)
        self._weight_decay_spin.setDecimals(4)
        self._weight_decay_spin.setValue(0.0001)
        self._add_form_row(form, "Weight Decay (0\u20130.1)", self._weight_decay_spin)

        val_split_row = QWidget()
        val_split_layout = QHBoxLayout(val_split_row)
        val_split_layout.setContentsMargins(0, 0, 0, 0)
        val_split_layout.setSpacing(10)
        self._val_split_slider = QSlider(Qt.Orientation.Horizontal)
        self._val_split_slider.setRange(10, 40)
        self._val_split_slider.setValue(20)
        self._val_split_label = QLabel("20%")
        self._val_split_label.setFixedWidth(36)
        self._val_split_label.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent;")
        self._val_split_slider.valueChanged.connect(lambda v: self._val_split_label.setText(f"{v}%"))
        val_split_layout.addWidget(self._val_split_slider, 1)
        val_split_layout.addWidget(self._val_split_label)
        self._add_form_row(form, "Validation Split (10\u201340%)", val_split_row)

        outer.addLayout(form)

        self._pretrained_checkbox = QCheckBox("Use Pretrained Backbone")
        self._pretrained_checkbox.setChecked(True)
        outer.addWidget(self._pretrained_checkbox)

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

    # -- Dataset sync / management -------------------------------------------

    def showEvent(self, event) -> None:  # noqa: D401 - Qt override
        super().showEvent(event)
        self._sync_from_main_window()
        bind_page_shortcuts(self._shortcut_bindings)
        if not self._tab_order_applied:
            self._apply_tab_order()
            self._tab_order_applied = True

    def hideEvent(self, event) -> None:  # noqa: D401 - Qt override
        super().hideEvent(event)
        unbind_page_shortcuts(self._shortcut_bindings)

    def _sync_from_main_window(self) -> None:
        if self.main_window is None:
            return
        incoming = getattr(self.main_window, "training_images", None)
        if not incoming:
            return

        existing_paths = {row.path for row in self._image_rows}
        for entry in incoming:
            path = entry.get("path")
            image = entry.get("image")
            if not path or image is None or path in existing_paths:
                continue
            self._add_training_image(path, image)
            existing_paths.add(path)

    def _on_add_more_images(self) -> None:
        file_filter = "Images (*.jpg *.jpeg *.png *.tif)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", file_filter)
        added, failed_names = self._load_and_add_images(paths)

        if failed_names:
            self._report_failed_loads(failed_names)
        elif added:
            self._show_bulk_status(f"Added {added} image(s).")

    def _on_import_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Import Folder")
        if not folder:
            return

        folder_path = Path(folder)
        found_paths = sorted(
            str(entry) for entry in folder_path.iterdir()
            if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not found_paths:
            self._show_status_error(f'No supported images (.jpg, .jpeg, .png, .tif) found in "{folder_path.name}".')
            return

        dialog = _FolderImportDialog(self._grade_labels, folder_path.name, len(found_paths), self)
        confirmed = dialog.exec() == QDialog.DialogCode.Accepted
        chosen_grade = dialog.selected_grade() if confirmed else None

        added, failed_names = self._load_and_add_images(found_paths, assign_grade=chosen_grade)

        if failed_names:
            self._report_failed_loads(failed_names)
        elif added:
            if confirmed:
                self._show_bulk_status(f'Imported {added} image(s) from "{folder_path.name}" and labelled "{chosen_grade}".')
            else:
                self._show_bulk_status(f'Imported {added} image(s) from "{folder_path.name}".')

    def _load_and_add_images(
        self, paths: List[str], assign_grade: Optional[str] = None
    ) -> Tuple[int, List[str]]:
        """Load and append each path as a training image, skipping duplicates/unsupported/corrupt files."""
        added = 0
        failed_names: List[str] = []
        for path in paths:
            if not path.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            if any(row.path == path for row in self._image_rows):
                continue
            try:
                with Image.open(path) as opened:
                    opened.load()
                    image = opened.convert("RGB")
            except (OSError, ValueError):
                failed_names.append(Path(path).name)
                continue
            self._add_training_image(path, image)
            if assign_grade:
                self._image_rows[-1].combo.setCurrentText(assign_grade)
            added += 1
        return added, failed_names

    def _report_failed_loads(self, failed_names: List[str]) -> None:
        names = ", ".join(failed_names[:3])
        if len(failed_names) > 3:
            names += f", and {len(failed_names) - 3} more"
        count_word = "image" if len(failed_names) == 1 else "images"
        self._show_status_error(f"Could not load {len(failed_names)} {count_word}: {names}")

    def _add_training_image(self, path: str, image: Image.Image) -> None:
        if any(row.path == path for row in self._image_rows):
            return

        row_index = self._table.rowCount()
        self._table.insertRow(row_index)

        filename_item = QTableWidgetItem(Path(path).name)
        filename_item.setFlags(filename_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        filename_item.setToolTip(path)
        self._table.setItem(row_index, 0, filename_item)

        combo = QComboBox()
        self._populate_grade_combo(combo)
        combo.setStyleSheet(
            f"QComboBox {{ background-color: {SURFACE_COLOR}; color: {TEXT_PRIMARY};"
            f"border: 1px solid {BORDER_COLOR}; border-radius: 4px; padding: 3px 6px; font-size: 11px; }}"
        )
        self._table.setCellWidget(row_index, 1, combo)

        remove_button = QPushButton("Remove")
        remove_button.setCursor(Qt.CursorShape.PointingHandCursor)
        remove_button.setToolTip(f"Remove {Path(path).name} from the training set")
        remove_button.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {DANGER_COLOR}; border: none;"
            f"font-size: 11px; padding: 2px; }}"
            f"QPushButton:hover {{ text-decoration: underline; }}"
        )
        self._table.setCellWidget(row_index, 2, remove_button)

        row = _TrainingImageRow(path=path, image=image, combo=combo, remove_button=remove_button)
        combo.currentIndexChanged.connect(lambda _index: self._on_row_grade_changed())
        remove_button.clicked.connect(lambda _checked, r=row: self._remove_training_image(r))

        self._image_rows.append(row)
        self._update_image_status_label()
        self._update_label_completion()

    def _remove_training_image(self, row: _TrainingImageRow) -> None:
        if row not in self._image_rows:
            return
        index = self._image_rows.index(row)
        self._table.removeRow(index)
        self._image_rows.pop(index)
        self._update_image_status_label()
        self._update_label_completion()

    def _on_row_grade_changed(self) -> None:
        self._update_image_status_label()
        self._update_label_completion()

    def _label_counts(self) -> Tuple[int, int]:
        total = len(self._image_rows)
        labelled = sum(1 for row in self._image_rows if row.combo.currentText() != PLACEHOLDER_GRADE)
        return total, labelled

    def _update_image_status_label(self) -> None:
        total, labelled = self._label_counts()
        if total == 0:
            self._image_status_label.setText("No images loaded yet.")
            return
        self._image_status_label.setText(f"{total} image(s) loaded \u2014 {labelled} assigned a grade.")

    def _update_label_completion(self) -> None:
        if self._label_completion_label is None:
            return
        total, labelled = self._label_counts()
        self._label_completion_label.setText(f"{labelled} of {total} images labelled")

        all_labelled = total > 0 and labelled == total
        color = SUCCESS_COLOR if all_labelled else ACCENT_COLOR
        self._label_completion_label.setStyleSheet(
            f"color: {color}; font-size: 12px; font-weight: 600; background: transparent;"
        )

        if self._start_button is None:
            return
        if self._thread is not None:
            # A training run is in progress; _set_training_ui_active manages the button then.
            return
        self._start_button.setEnabled(all_labelled)
        if all_labelled:
            self._start_button.setToolTip(shortcut_tooltip("Start training the model", "S"))
        else:
            self._start_button.setToolTip("Label all images before training")

    # -- Grade label management ----------------------------------------------

    def _populate_grade_combo(self, combo: QComboBox, current_text: str = "") -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(PLACEHOLDER_GRADE)
        combo.addItems(self._grade_labels)
        index = combo.findText(current_text) if current_text else -1
        combo.setCurrentIndex(index if index > 0 else 0)
        combo.blockSignals(False)

    def _refresh_grade_dropdowns(self) -> None:
        for row in self._image_rows:
            self._populate_grade_combo(row.combo, current_text=row.combo.currentText())
        if self._bulk_grade_combo is not None:
            current = self._bulk_grade_combo.currentText()
            self._bulk_grade_combo.blockSignals(True)
            self._bulk_grade_combo.clear()
            self._bulk_grade_combo.addItem(BULK_ASSIGN_PLACEHOLDER)
            self._bulk_grade_combo.addItems(self._grade_labels)
            index = self._bulk_grade_combo.findText(current)
            self._bulk_grade_combo.setCurrentIndex(index if index >= 0 else 0)
            self._bulk_grade_combo.blockSignals(False)
        self._update_image_status_label()
        self._update_label_completion()

    def _on_grade_item_changed(self, item: QListWidgetItem) -> None:
        row = self._grade_list.row(item)
        new_text = item.text().strip()
        old_text = self._grade_labels[row]

        if not new_text or (new_text in self._grade_labels and new_text != old_text):
            self._grade_list.blockSignals(True)
            item.setText(old_text)
            self._grade_list.blockSignals(False)
            return

        self._grade_labels[row] = new_text
        self._refresh_grade_dropdowns()

    def _add_grade(self) -> None:
        if len(self._grade_labels) >= MAX_GRADES:
            return

        base_name = "New Grade"
        candidate = base_name
        counter = 1
        existing = set(self._grade_labels)
        while candidate in existing:
            counter += 1
            candidate = f"{base_name} {counter}"

        self._grade_labels.append(candidate)

        item = QListWidgetItem(candidate)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self._grade_list.blockSignals(True)
        self._grade_list.addItem(item)
        self._grade_list.blockSignals(False)

        self._refresh_grade_dropdowns()
        self._update_grade_buttons_enabled()

    def _remove_selected_grade(self) -> None:
        row = self._grade_list.currentRow()
        if row < 0 or len(self._grade_labels) <= MIN_GRADES:
            return

        self._grade_list.blockSignals(True)
        self._grade_list.takeItem(row)
        self._grade_list.blockSignals(False)
        del self._grade_labels[row]

        self._refresh_grade_dropdowns()
        self._update_grade_buttons_enabled()

    def _update_grade_buttons_enabled(self) -> None:
        if self._add_grade_button is not None:
            self._add_grade_button.setEnabled(len(self._grade_labels) < MAX_GRADES)
        if self._remove_grade_button is not None:
            has_selection = self._grade_list.currentRow() >= 0
            self._remove_grade_button.setEnabled(has_selection and len(self._grade_labels) > MIN_GRADES)

    # -- Bulk selection & label assignment ------------------------------------

    def _get_selected_row_indices(self) -> List[int]:
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return []
        return sorted({index.row() for index in selection_model.selectedRows()})

    def _on_select_all(self) -> None:
        self._table.selectAll()

    def _on_select_none(self) -> None:
        self._table.clearSelection()

    def _on_apply_to_selected(self) -> None:
        grade = self._bulk_grade_combo.currentText() if self._bulk_grade_combo is not None else ""
        if not grade or grade == BULK_ASSIGN_PLACEHOLDER:
            self._show_status_error("Choose a grade to assign before clicking \u201cApply to Selected.\u201d")
            return

        selected_indices = self._get_selected_row_indices()
        if not selected_indices:
            self._show_status_error("Select at least one image in the table first.")
            return

        rows = [self._image_rows[index] for index in selected_indices if index < len(self._image_rows)]
        self._apply_grade_to_rows(rows, grade)
        self._show_bulk_status(f"Grade assigned to {len(rows)} image(s).")

    def _apply_grade_to_rows(self, rows: List[_TrainingImageRow], grade: str) -> None:
        for row in rows:
            row.combo.setCurrentText(grade)

    def _show_bulk_status(self, message: str) -> None:
        if self._bulk_status_label is None:
            return
        self._bulk_status_label.setText(message)
        self._bulk_status_label.setVisible(True)
        self._bulk_status_timer.start(BULK_STATUS_TIMEOUT_MS)

    def _clear_bulk_status(self) -> None:
        if self._bulk_status_label is None:
            return
        self._bulk_status_label.setVisible(False)

    # -- Right-click context menu ---------------------------------------------

    def _show_table_context_menu(self, pos: QPoint) -> None:
        index = self._table.indexAt(pos)
        if not index.isValid() or index.row() >= len(self._image_rows):
            return
        clicked_row_index = index.row()

        selected_indices = self._get_selected_row_indices()
        if clicked_row_index not in selected_indices:
            self._table.selectRow(clicked_row_index)
            selected_indices = [clicked_row_index]

        clicked_row = self._image_rows[clicked_row_index]
        selected_rows = [self._image_rows[i] for i in selected_indices if i < len(self._image_rows)]

        menu = QMenu(self)
        menu.setStyleSheet(
            f"""
            QMenu {{ background-color: {SURFACE_COLOR}; color: {TEXT_PRIMARY}; border: 1px solid {BORDER_COLOR}; }}
            QMenu::item {{ padding: 6px 20px; }}
            QMenu::item:selected {{ background-color: {ACCENT_COLOR}; color: #13151A; }}
            QMenu::separator {{ height: 1px; background-color: {BORDER_COLOR}; margin: 4px 0; }}
            """
        )

        assign_this_menu = menu.addMenu("Assign grade to this image")
        for label in self._grade_labels:
            action = assign_this_menu.addAction(label)
            action.triggered.connect(
                lambda _checked=False, r=clicked_row, g=label: self._apply_grade_to_rows_with_status([r], g)
            )

        assign_selected_menu = menu.addMenu(f"Assign grade to all selected images ({len(selected_rows)})")
        for label in self._grade_labels:
            action = assign_selected_menu.addAction(label)
            action.triggered.connect(
                lambda _checked=False, rs=selected_rows, g=label: self._apply_grade_to_rows_with_status(rs, g)
            )

        menu.addSeparator()
        remove_action = menu.addAction("Remove from training set")
        remove_action.triggered.connect(lambda: self._remove_selected_or_clicked(selected_rows, clicked_row))

        menu.exec(self._table.viewport().mapToGlobal(pos))

    def _apply_grade_to_rows_with_status(self, rows: List[_TrainingImageRow], grade: str) -> None:
        self._apply_grade_to_rows(rows, grade)
        self._show_bulk_status(f"Grade assigned to {len(rows)} image(s).")

    def _remove_selected_or_clicked(
        self, selected_rows: List[_TrainingImageRow], clicked_row: _TrainingImageRow
    ) -> None:
        rows_to_remove = selected_rows if len(selected_rows) > 1 else [clicked_row]
        for row in list(rows_to_remove):
            self._remove_training_image(row)

    # -- Training lifecycle ---------------------------------------------------

    def _clear_status(self) -> None:
        self._status_label.setText("")
        self._status_label.setVisible(False)

    def _show_status_error(self, message: str) -> None:
        self._status_label.setText(message)
        self._status_label.setVisible(True)

    def _on_start_training(self) -> None:
        if self._thread is not None:
            return

        self._clear_status()

        model_name = self._model_name_edit.text().strip()
        if not model_name:
            self._show_status_error("Please enter a model name.")
            return
        if (_MODELS_DIR / f"{model_name}.pt").exists():
            self._show_status_error(f'A model named "{model_name}" already exists. Choose a different name.')
            return

        if len(self._grade_labels) < MIN_GRADES:
            self._show_status_error(f"At least {MIN_GRADES} grade labels are required.")
            return

        assigned = [
            (row.image, row.combo.currentText())
            for row in self._image_rows
            if row.combo.currentText() != PLACEHOLDER_GRADE
        ]
        unassigned_count = len(self._image_rows) - len(assigned)
        if unassigned_count > 0:
            self._show_status_error(f"{unassigned_count} image(s) still need a grade assigned.")
            return
        if len(assigned) < MIN_TRAINING_IMAGES:
            self._show_status_error(f"Add at least {MIN_TRAINING_IMAGES} labeled images to train a model.")
            return
        if len({label for _, label in assigned}) < 2:
            self._show_status_error("Assign at least 2 different grades among your images.")
            return

        grade_labels = list(self._grade_labels)
        label_to_index = {label: index for index, label in enumerate(grade_labels)}
        samples = [(image, label_to_index[label]) for image, label in assigned]

        try:
            train_samples, val_samples = self._split_samples(samples, self._val_split_slider.value() / 100.0)
        except ValueError as exc:
            self._show_status_error(f"Could not split dataset: {exc}")
            return

        use_pretrained = self._pretrained_checkbox.isChecked()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BitumenCNN(num_classes=len(grade_labels), pretrained=use_pretrained)

        batch_size = int(self._batch_size_combo.currentText())
        train_loader = DataLoader(_ImageGradeDataset(train_samples, train=True), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(_ImageGradeDataset(val_samples, train=False), batch_size=batch_size, shuffle=False)

        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=float(self._lr_combo.currentText()),
            num_epochs=self._epochs_spin.value(),
            batch_size=batch_size,
            optimizer_name=self._optimizer_combo.currentText(),
            weight_decay=self._weight_decay_spin.value(),
        )

        self._pending_model_name = model_name
        self._pending_grade_labels = grade_labels
        self._pending_num_classes = len(grade_labels)

        self._launch_training_thread(trainer)

    @staticmethod
    def _split_samples(
        samples: List[Tuple[Image.Image, int]], val_fraction: float
    ) -> Tuple[List[Tuple[Image.Image, int]], List[Tuple[Image.Image, int]]]:
        labels = [label for _, label in samples]
        label_counts = Counter(labels)
        can_stratify = all(count >= 2 for count in label_counts.values())

        try:
            if can_stratify:
                train_samples, val_samples = train_test_split(
                    samples, test_size=val_fraction, random_state=42, stratify=labels
                )
            else:
                train_samples, val_samples = train_test_split(samples, test_size=val_fraction, random_state=42)
        except ValueError:
            train_samples, val_samples = train_test_split(samples, test_size=val_fraction, random_state=42)

        if not val_samples:
            raise ValueError("validation split produced 0 images; add more images or lower the split percentage")
        if not train_samples:
            raise ValueError("training split produced 0 images; add more images or raise the split percentage")
        return train_samples, val_samples

    def _launch_training_thread(self, trainer: ModelTrainer) -> None:
        self._trainer = trainer
        self._thread = QThread(self)
        self._worker = _TrainingWorker(trainer)

        trainer.moveToThread(self._thread)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        trainer.progress_updated.connect(self._on_epoch_progress)
        self._worker.finished.connect(self._on_training_finished)
        self._worker.failed.connect(self._on_training_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)

        self._set_training_ui_active(True)
        self._progress_panel.setVisible(True)
        self._progress_panel.reset(trainer.num_epochs)

        self._thread.start()

    def _on_epoch_progress(self, epoch: int, train_loss: float, val_loss: float, val_accuracy: float) -> None:
        self._progress_panel.update_progress(epoch, train_loss, val_loss, val_accuracy)

    def _on_training_finished(self, result: TrainingResult) -> None:
        if result.final_epoch < 1:
            self._progress_panel.show_stopped()
            self._set_training_ui_active(False)
            return

        model = self._trainer.model
        try:
            paths = save_model(
                model=model,
                name=self._pending_model_name,
                num_classes=self._pending_num_classes,
                grade_labels=self._pending_grade_labels,
                training_history=result.training_history,
                save_dir=_MODELS_DIR,
                best_val_accuracy=result.best_val_accuracy,
            )
        except OSError as exc:
            self._progress_panel.append_log(f"Failed to save model: {exc}")
            self._show_status_error(f"Failed to save model: {exc}")
            self._set_training_ui_active(False)
            return

        self._progress_panel.show_completion(self._pending_model_name)

        if self.main_window is not None:
            try:
                metadata = load_model_metadata(paths["metadata_path"])
            except (OSError, ValueError):
                metadata = None
            self.main_window.set_active_model(str(paths["model_path"]), metadata)

        self._set_training_ui_active(False)

    def _on_training_failed(self, message: str) -> None:
        self._progress_panel.append_log(f"Training failed: {message}")
        self._show_status_error(f"Training failed: {message}")
        self._set_training_ui_active(False)

    def _on_thread_finished(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
        if self._worker is not None:
            self._worker.deleteLater()
        self._thread = None
        self._worker = None
        self._trainer = None

    def _on_stop_training(self) -> None:
        if self._trainer is not None:
            self._trainer.request_stop()
            self._progress_panel.append_log("Stop requested\u2026")
        self._stop_button.setEnabled(False)

    def _set_training_ui_active(self, active: bool) -> None:
        self._stop_button.setEnabled(active)

        if active:
            self._start_button.setEnabled(False)
            self._start_button.setToolTip("Training in progress\u2026")
        else:
            # Re-derive enabled/tooltip from label-completion state rather than
            # unconditionally re-enabling, since not every image may be labelled.
            self._update_label_completion()

        for widget in (
            self._model_name_edit,
            self._epochs_spin,
            self._batch_size_combo,
            self._lr_combo,
            self._optimizer_combo,
            self._weight_decay_spin,
            self._val_split_slider,
            self._pretrained_checkbox,
            self._add_grade_button,
            self._remove_grade_button,
            self._grade_list,
            self._add_images_button,
            self._import_folder_button,
            self._select_all_button,
            self._select_none_button,
            self._bulk_grade_combo,
            self._apply_bulk_button,
        ):
            if widget is not None:
                widget.setEnabled(not active)

        if active:
            for row in self._image_rows:
                row.combo.setEnabled(False)
                row.remove_button.setEnabled(False)
        else:
            for row in self._image_rows:
                row.combo.setEnabled(True)
                row.remove_button.setEnabled(True)
            self._update_grade_buttons_enabled()

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
