"""Train page. You drop a labels file and a photo folder, pick an architecture, and start a run. Matching and splitting happen in RegressionDataset. This page just shows the summaries and pipes the trainer into the progress panel."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd
import torch
from PyQt6.QtCore import Qt, QThread, QTimer
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
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
)
from app.ml.dataset import RegressionDataset
from app.ml.recipe import (
    BATCH_SIZE,
    DEFAULT_SPLIT_MODE,
    IMAGE_SIZE,
    NUM_EPOCHS,
    TEST_FRACTION,
    VAL_FRACTION,
    WEIGHT_DECAY,
    learning_rate_for_adaptation,
)
from app.ml.trainer import RegressionTrainer, RegressionTrainingResult
from app.pages.train_widgets import (
    _CsvDropZone,
    _DatasetSummaryCard,
    _FolderDropZone,
    _MatchSummaryCard,
)
from app.paths import MODELS_DIR
from app.theme import (
    BACKGROUND_COLOR,
    BORDER_COLOR,
    DANGER_COLOR,
    LABEL_RESET_QSS,
    PAGE_MARGINS,
    PAGE_SPACING,
    SURFACE_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    accent_button_qss,
    card_qss,
    danger_outline_button_qss,
)
from app.utils.data_io import read_labels_file
from app.utils.image_utils import image_size_from_metadata, is_legacy_resnet18
from app.utils.media import collect_images
from app.utils.model_io import (
    list_saved_models,
    load_model_metadata,
    save_model,
)

if TYPE_CHECKING:
    from app.main_window import MainWindow

LEFT_PANEL_WIDTH = 500
VAL_SPLIT_REBUILD_DEBOUNCE_MS = 300

EXPECTED_COLUMNS = list(RegressionDataset.EXPECTED_COLUMNS)
_COMBO_SIZE_ADJUST = (
    QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
)


def _default_num_workers() -> int:
    """How many extra processes should decode photos while the GPU trains. On Windows spawning those workers is slow in a desktop app, so we use 0 there and decode on the main process."""
    return 0 if platform.system() == "Windows" else 4


class TrainPage(QWidget):
    """The Train tab. You load a labels file and photo folder here, then start a job on a background thread. Progress streams into the panel on the right."""

    def __init__(
        self,
        main_window: Optional["MainWindow"] = None,
        parent: Optional[QWidget] = None,
    ):
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

        self._folder_drop_zone: Optional[_FolderDropZone] = None
        self._folder_path_label: Optional[QLabel] = None
        self._folder_count_label: Optional[QLabel] = None
        self._match_card: Optional[_MatchSummaryCard] = None

        self._step3_frame: Optional[QFrame] = None
        self._dataset_summary_card: Optional[_DatasetSummaryCard] = None

        self._model_name_edit: Optional[QLineEdit] = None
        self._continue_checkbox: Optional[QCheckBox] = None
        self._continue_combo: Optional[QComboBox] = None
        self._architecture_combo: Optional[QComboBox] = None
        self._adaptation_combo: Optional[QComboBox] = None
        self._head_combo: Optional[QComboBox] = None
        self._adaptation_label: Optional[QLabel] = None
        self._head_label: Optional[QLabel] = None
        self._epochs_spin: Optional[QSpinBox] = None
        self._batch_size_spin: Optional[QSpinBox] = None
        self._lr_spin: Optional[QDoubleSpinBox] = None
        self._start_button: Optional[QPushButton] = None
        self._stop_button: Optional[QPushButton] = None
        self._status_label: Optional[QLabel] = None
        self._progress_panel: Optional[ProgressPanel] = None

        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.timeout.connect(
            self._maybe_rebuild_dataset_summary
        )

        self._thread: Optional[QThread] = None
        self._trainer: Optional[RegressionTrainer] = None
        self._pending_model_name: Optional[str] = None
        self._pending_train_config: Optional[Dict] = None
        self._parent_model_meta: Optional[Dict] = None

        self._build_ui()
        self._update_start_button_state()

    # Build the widgets and lay them out.

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(*PAGE_MARGINS)
        root.setSpacing(PAGE_SPACING)

        root.addLayout(self._build_header())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
        )

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 4, 4)
        content_layout.setSpacing(20)

        content_layout.addWidget(self._build_left_panel())
        content_layout.addWidget(self._build_right_panel(), 1)

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    def _build_header(self) -> QVBoxLayout:
        header = QVBoxLayout()
        header.setSpacing(4)

        title = QLabel("Train")
        title.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 16px; background: transparent;"
        )
        header.addWidget(title)

        return header

    def _make_section_frame(self, title: str):
        frame = QFrame()
        frame.setStyleSheet(card_qss())

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 13px; background: transparent;"
        )
        layout.addWidget(title_label)

        return frame, layout

    # Left column: labels file, photo folder, and the match summary.

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
        section, layout = self._make_section_frame("Labels")

        self._csv_drop_zone = _CsvDropZone()
        self._csv_drop_zone.file_selected.connect(self._on_csv_selected)
        layout.addWidget(self._csv_drop_zone)

        self._csv_loaded_label = QLabel("")
        self._csv_loaded_label.setWordWrap(True)
        self._csv_loaded_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px;"
            f" background: transparent;"
        )
        self._csv_loaded_label.setVisible(False)
        layout.addWidget(self._csv_loaded_label)

        self._csv_error_banner = QFrame()
        self._csv_error_banner.setObjectName("csvErrorBanner")
        self._csv_error_banner.setStyleSheet(
            # Target the banner by id. QLabel is a QFrame in Qt, so a bare QFrame rule would also box the error text inside.
            f"QFrame#csvErrorBanner {{ background-color: rgba(229, 72, 77, "
            f"25); border: 1px solid {DANGER_COLOR};"
            f"border-radius: 3px; }}"
        )
        error_layout = QVBoxLayout(self._csv_error_banner)
        error_layout.setContentsMargins(12, 10, 12, 10)
        self._csv_error_label = QLabel("")
        self._csv_error_label.setWordWrap(True)
        self._csv_error_label.setStyleSheet(
            f"color: {DANGER_COLOR}; font-size: 11px; background: transparent;"
        )
        error_layout.addWidget(self._csv_error_label)
        self._csv_error_banner.setVisible(False)
        layout.addWidget(self._csv_error_banner)

        self._preview_table = QTableWidget()
        self._preview_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._preview_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self._preview_table.verticalHeader().setVisible(False)
        self._preview_table.setFixedHeight(160)
        self._preview_table.horizontalHeader().setStretchLastSection(True)
        self._preview_table.setStyleSheet(
            f"""
            QTableWidget {{
                background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR}; border-radius: 3px;
                gridline-color: {BORDER_COLOR};
            }}
            QTableWidget::item {{ padding: 3px; }}
            QHeaderView::section {{
                background-color: {SURFACE_COLOR}; color: {TEXT_SECONDARY};
                border: none;
                padding: 5px; font-size: 10px; font-weight: 600;
            }}
            """
        )
        self._preview_table.setVisible(False)
        layout.addWidget(self._preview_table)

        self._column_mapping_frame = QFrame()
        self._column_mapping_frame.setObjectName("columnMappingFrame")
        self._column_mapping_frame.setStyleSheet(
            f"QFrame#columnMappingFrame {{ background-color: "
            f"{BACKGROUND_COLOR}; border-radius: 3px; }}"
            f"{LABEL_RESET_QSS}"
        )
        mapping_form = QFormLayout(self._column_mapping_frame)
        mapping_form.setContentsMargins(12, 10, 12, 10)
        mapping_form.setSpacing(6)
        self._add_mapping_row(mapping_form, "Filename column:", "Image")
        self._add_mapping_row(
            mapping_form, "Outputs to predict:", ", ".join(OUTPUT_NAMES)
        )
        self._add_mapping_row(mapping_form, "Display only:", "Pan (batch)")
        self._column_mapping_frame.setVisible(False)
        layout.addWidget(self._column_mapping_frame)

        return section

    def _add_mapping_row(
        self, form: QFormLayout, label_text: str, value_text: str
    ) -> None:
        label = QLabel(label_text)
        label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px;"
            f" background: transparent;"
        )
        value = QLabel(value_text)
        value.setWordWrap(True)
        value.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 11px; font-weight: 600;"
            f" background: transparent;"
        )
        form.addRow(label, value)

    def _build_folder_section(self) -> QFrame:
        section, layout = self._make_section_frame("Photo folder")

        self._folder_drop_zone = _FolderDropZone()
        self._folder_drop_zone.folder_selected.connect(
            self._apply_image_folder
        )
        layout.addWidget(self._folder_drop_zone)

        self._folder_path_label = QLabel("")
        self._folder_path_label.setWordWrap(True)
        self._folder_path_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 11px; background: transparent;"
        )
        self._folder_path_label.setVisible(False)
        layout.addWidget(self._folder_path_label)

        self._folder_count_label = QLabel("")
        self._folder_count_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px;"
            f" background: transparent;"
        )
        self._folder_count_label.setVisible(False)
        layout.addWidget(self._folder_count_label)

        self._match_card = _MatchSummaryCard()
        self._match_card.setVisible(False)
        layout.addWidget(self._match_card)

        return section

    def _build_dataset_summary_section(self) -> QFrame:
        section, layout = self._make_section_frame("Dataset")
        self._dataset_summary_card = _DatasetSummaryCard()
        layout.addWidget(self._dataset_summary_card)
        return section

    # Right column: architecture settings and the start button.

    def _build_right_panel(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(self._build_model_settings_section())
        layout.addWidget(self._build_strategy_section())

        self._start_button = QPushButton("Start training")
        self._start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_button.setStyleSheet(accent_button_qss())
        self._start_button.clicked.connect(self._on_start_training)
        layout.addWidget(self._start_button)

        self._stop_button = QPushButton("Stop")
        self._stop_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_button.setStyleSheet(danger_outline_button_qss())
        self._stop_button.clicked.connect(self._on_stop_training)
        self._stop_button.setVisible(False)
        layout.addWidget(self._stop_button)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(
            f"color: {DANGER_COLOR}; font-size: 12px; background: transparent;"
        )
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        self._progress_panel = ProgressPanel()
        self._progress_panel.setVisible(False)
        layout.addWidget(self._progress_panel)

        layout.addStretch(1)
        return container

    def _build_model_settings_section(self) -> QFrame:
        section, layout = self._make_section_frame("Model")

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self._model_name_edit = QLineEdit()
        self._model_name_edit.setPlaceholderText("e.g. Site-A Run 1")
        self._model_name_edit.textChanged.connect(
            lambda _text: self._update_start_button_state()
        )
        self._add_form_row(form, "Model Name", self._model_name_edit)

        layout.addLayout(form)

        self._continue_checkbox = QCheckBox("Continue training a saved model")
        self._continue_checkbox.setToolTip(
            "Load an existing checkpoint and keep training it on the dataset "
            "below. "
            "Saves a new model; the original is left unchanged."
        )
        self._continue_checkbox.toggled.connect(self._on_continue_toggled)
        layout.addWidget(self._continue_checkbox)

        self._continue_combo = QComboBox()
        self._continue_combo.setToolTip("Which saved model to continue from.")
        self._continue_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._continue_combo.setSizeAdjustPolicy(_COMBO_SIZE_ADJUST)
        self._continue_combo.setMinimumContentsLength(12)
        self._continue_combo.currentIndexChanged.connect(
            self._on_continue_model_changed
        )
        self._continue_combo.setVisible(False)
        layout.addWidget(self._continue_combo)

        section.setStyleSheet(
            f"""
            QFrame {{ background-color: {SURFACE_COLOR};
            border: 1px solid {BORDER_COLOR}; border-radius: 3px; }}
            {LABEL_RESET_QSS}
            QLineEdit {{
                background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR}; border-radius: 3px;
                padding: 6px 10px; min-height: 26px;
            }}
            QComboBox {{
                background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR}; border-radius: 3px;
                padding: 6px 28px 6px 10px; min-height: 26px;
            }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QCheckBox {{ color: {TEXT_PRIMARY}; font-size: 12px;
            background: transparent; }}
            """
        )
        return section

    def _build_strategy_section(self) -> QFrame:
        section, layout = self._make_section_frame("Training")

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self._architecture_combo = QComboBox()
        for key in TRAINABLE_ARCHITECTURES:
            self._architecture_combo.addItem(ARCHITECTURE_LABELS[key], key)
        self._architecture_combo.setCurrentIndex(0)
        self._architecture_combo.setToolTip(
            "Baseline CNN from scratch is the most robust default for froth "
            "images."
        )
        self._architecture_combo.currentIndexChanged.connect(
            self._on_strategy_changed
        )
        self._architecture_combo.currentIndexChanged.connect(
            self._sync_learning_rate_from_adaptation
        )
        self._add_form_row(form, "Architecture", self._architecture_combo)

        self._adaptation_combo = QComboBox()
        self._adaptation_combo.addItem("Fine-tune (recommended)", "ft")
        self._adaptation_combo.addItem("Frozen features", "fe")
        self._adaptation_combo.setToolTip(
            "Fine-tuning trains the whole net and beat frozen ImageNet "
            "features in the study. "
            "Frozen features keep the backbone fixed and were weaker, "
            "especially for Bitumen."
        )
        self._adaptation_combo.currentIndexChanged.connect(
            self._on_strategy_changed
        )
        self._adaptation_combo.currentIndexChanged.connect(
            self._sync_learning_rate_from_adaptation
        )
        self._adaptation_label = QLabel("Adaptation")
        self._adaptation_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 12px;"
            f" background: transparent;"
        )
        self._adaptation_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._adaptation_combo.setSizeAdjustPolicy(_COMBO_SIZE_ADJUST)
        self._adaptation_combo.setMinimumContentsLength(12)
        form.addRow(self._adaptation_label, self._adaptation_combo)

        self._head_combo = QComboBox()
        self._head_combo.addItem("Native linear", "native")
        self._head_combo.addItem("2-layer (C2)", "c2")
        self._head_combo.setToolTip(
            "Native is a single linear layer. C2 is the 2-layer head that "
            "helped Solids. "
            "Deeper batch-norm heads are omitted because they collapse."
        )
        self._head_combo.currentIndexChanged.connect(self._on_strategy_changed)
        self._head_label = QLabel("Head")
        self._head_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 12px;"
            f" background: transparent;"
        )
        self._head_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._head_combo.setSizeAdjustPolicy(_COMBO_SIZE_ADJUST)
        self._head_combo.setMinimumContentsLength(12)
        form.addRow(self._head_label, self._head_combo)

        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 200)
        self._epochs_spin.setValue(NUM_EPOCHS)
        self._epochs_spin.setToolTip(
            "How many full passes over the training images (100 in the "
            "study). "
            "The checkpoint with the best mean validation R² is kept even if "
            "a later epoch is worse."
        )
        self._add_form_row(form, "Epochs", self._epochs_spin)

        self._batch_size_spin = QSpinBox()
        self._batch_size_spin.setRange(1, 128)
        self._batch_size_spin.setValue(BATCH_SIZE)
        self._batch_size_spin.setToolTip(
            "Images per Adam step (32 in the study). Lower this if you run "
            "out of memory; "
            "higher is faster on GPU but needs more RAM. Last incomplete "
            "batch is still used."
        )
        self._add_form_row(form, "Batch size", self._batch_size_spin)

        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setDecimals(6)
        self._lr_spin.setRange(1e-6, 0.1)
        self._lr_spin.setSingleStep(1e-4)
        self._lr_spin.setToolTip(
            "Adam step size. The study used 0.0001 for baseline/fine-tune and "
            "0.001 when the "
            "backbone is frozen. Lower it if loss jumps around;"
            " raise it if loss barely moves. "
            "Changing Adaptation resets this to the study value for that mode."
        )
        self._add_form_row(form, "Learning rate", self._lr_spin)

        layout.addLayout(form)

        section.setStyleSheet(
            f"""
            QFrame {{ background-color: {SURFACE_COLOR};
            border: 1px solid {BORDER_COLOR}; border-radius: 3px; }}
            {LABEL_RESET_QSS}
            QComboBox {{
                background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR}; border-radius: 3px;
                padding: 6px 28px 6px 10px; min-height: 26px;
            }}
            QSpinBox, QDoubleSpinBox {{
                background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR}; border-radius: 3px;
                padding: 4px 22px 4px 8px; min-height: 26px;
            }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            """
        )
        self._on_strategy_changed()
        self._sync_learning_rate_from_adaptation()
        return section

    def _add_form_row(
        self, form: QFormLayout, label_text: str, field: QWidget
    ) -> None:
        label = QLabel(label_text)
        label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 12px;"
            f" background: transparent;"
        )
        field.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        if isinstance(field, QComboBox):
            field.setSizeAdjustPolicy(_COMBO_SIZE_ADJUST)
            field.setMinimumContentsLength(12)
        form.addRow(label, field)

    def _current_architecture(self) -> str:
        if self._architecture_combo is None:
            return "baseline"
        data = self._architecture_combo.currentData()
        return str(data) if data else "baseline"

    def _current_adaptation(self) -> str:
        if not self._is_transfer_architecture():
            return "scratch"
        if self._adaptation_combo is None:
            return "ft"
        data = self._adaptation_combo.currentData()
        return str(data) if data else "ft"

    def _current_head(self) -> str:
        if self._parent_model_meta:
            return str(self._parent_model_meta.get("head") or "native")
        if not self._is_transfer_architecture():
            return "native"
        if self._head_combo is None:
            return "native"
        data = self._head_combo.currentData()
        return str(data) if data else "native"

    def _is_transfer_architecture(
        self, architecture: Optional[str] = None
    ) -> bool:
        architecture = architecture or self._current_architecture()
        return architecture in {"resnet50", "vgg16", "resnet18"}

    def _current_batch_size(self) -> int:
        if self._batch_size_spin is None:
            return BATCH_SIZE
        return max(1, int(self._batch_size_spin.value()))

    def _current_learning_rate(
        self, adaptation: Optional[str] = None
    ) -> float:
        if self._lr_spin is None:
            return learning_rate_for_adaptation(
                adaptation or self._current_adaptation()
            )
        return float(self._lr_spin.value())

    def _sync_learning_rate_from_adaptation(self, *_args) -> None:
        """Puts the paper's Adam learning rate into the spin box for the current adaptation mode."""
        if self._lr_spin is None:
            return
        if (
            self._is_transfer_architecture()
            and self._adaptation_combo is not None
        ):
            data = self._adaptation_combo.currentData()
            mode = str(data) if data else "ft"
        else:
            mode = "scratch"
        self._lr_spin.setValue(learning_rate_for_adaptation(mode))

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
                # Remove ResNet-18 from the list if we only added it to continue an old checkpoint.
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
        index = self._architecture_combo.findData(
            current if current in TRAINABLE_ARCHITECTURES else "baseline"
        )
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
                if (
                    isinstance(data, dict)
                    and data.get("model_path") == previous_path
                ):
                    self._continue_combo.setCurrentIndex(index)
                    break
        self._continue_combo.blockSignals(False)

    def _on_continue_model_changed(self, _index: int = 0) -> None:
        data = (
            self._continue_combo.currentData()
            if self._continue_combo is not None
            else None
        )
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
                    ARCHITECTURE_LABELS.get(architecture, architecture),
                    architecture,
                )
            index = self._architecture_combo.findData(architecture)
            if index >= 0:
                self._architecture_combo.setCurrentIndex(index)
            self._architecture_combo.setEnabled(False)
            self._architecture_combo.blockSignals(False)
        if self._head_combo is not None:
            head = data.get("head", "native")
            head_index = self._head_combo.findData(head)
            if head_index >= 0:
                self._head_combo.setCurrentIndex(head_index)
            self._head_combo.setEnabled(False)
        if (
            self._model_name_edit is not None
            and not self._model_name_edit.text().strip()
        ):
            base = data.get("name") or "model"
            self._model_name_edit.setText(f"{base} retrained")
        self._on_strategy_changed()
        self._rebuild_timer.start(VAL_SPLIT_REBUILD_DEBOUNCE_MS)

    def prepare_retrain(self, metadata: Dict) -> None:
        """Ticks Continue training and selects this saved model so the user can run it on a new labels file. Pass the metadata dict from the Models page."""
        if self._continue_checkbox is None:
            return
        self._refresh_continue_combo()
        self._continue_checkbox.setChecked(True)
        target_path = metadata.get("model_path")
        if self._continue_combo is not None and target_path:
            for index in range(self._continue_combo.count()):
                data = self._continue_combo.itemData(index)
                if (
                    isinstance(data, dict)
                    and data.get("model_path") == target_path
                ):
                    self._continue_combo.setCurrentIndex(index)
                    break
            else:
                self._continue_combo.addItem(
                    metadata.get("name") or "Saved model", metadata
                )
                self._continue_combo.setCurrentIndex(
                    self._continue_combo.count() - 1
                )
        base = metadata.get("name") or "model"
        if self._model_name_edit is not None:
            self._model_name_edit.setText(f"{base} retrained")
        self._on_continue_model_changed()

    def _on_strategy_changed(self, _index: int = 0) -> None:
        is_transfer = self._is_transfer_architecture()
        continuing = bool(
            self._continue_checkbox is not None
            and self._continue_checkbox.isChecked()
        )
        show_transfer_controls = is_transfer or (
            continuing and self._is_transfer_architecture()
        )

        if self._adaptation_combo is not None:
            self._adaptation_combo.setVisible(show_transfer_controls)
            if self._adaptation_label is not None:
                self._adaptation_label.setVisible(show_transfer_controls)
        if self._head_combo is not None:
            self._head_combo.setVisible(show_transfer_controls)
            self._head_combo.setEnabled(
                show_transfer_controls and not continuing
            )
            if self._head_label is not None:
                self._head_label.setVisible(show_transfer_controls)

    # After the labels file or folder changes, rebuild the match summary.

    def showEvent(self, event) -> None:  # noqa: D401  Qt calls this when the page is shown
        super().showEvent(event)
        if (
            self._continue_checkbox is not None
            and self._continue_checkbox.isChecked()
        ):
            self._refresh_continue_combo()

    def _on_csv_selected(self, path: str) -> None:
        try:
            df = read_labels_file(path)
        except Exception as exc:  # noqa: BLE001
            # Tell the user if the labels file would not parse.
            self._show_status_error(f"Couldn't read label file: {exc}")
            return

        self._csv_path = path
        missing = [
            column for column in EXPECTED_COLUMNS if column not in df.columns
        ]

        if missing:
            self._csv_dataframe = None
            self._csv_loaded_label.setVisible(False)
            self._preview_table.setVisible(False)
            self._column_mapping_frame.setVisible(False)
            self._csv_error_label.setText(
                (
                    f"Missing columns: {', '.join(missing)}.\n"
                    f"Expected: {', '.join(EXPECTED_COLUMNS)}"
                )
            )
            self._csv_error_banner.setVisible(True)
        else:
            self._csv_dataframe = df
            self._csv_error_banner.setVisible(False)
            self._csv_loaded_label.setText(
                f"Loaded \u201c{Path(path).name}\u201d \u2014 {len(df)} "
                f"row(s)."
            )
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
        self._preview_table.setHorizontalHeaderLabels(
            [str(column) for column in preview.columns]
        )

        for row_index in range(len(preview)):
            for col_index, column in enumerate(preview.columns):
                value = preview.iloc[row_index, col_index]
                text = "" if pd.isna(value) else str(value)
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._preview_table.setItem(row_index, col_index, item)

        self._preview_table.resizeColumnsToContents()

    def _apply_image_folder(self, folder: str) -> bool:
        """Sets the photo folder for step 2. Pass the folder path. You get False if we could not read it."""
        try:
            image_count = len(collect_images(folder))
        except OSError as exc:
            self._show_status_error(f"Couldn't read folder: {exc}")
            return False

        self._image_dir = folder
        self._folder_path_label.setText(folder)
        self._folder_path_label.setVisible(True)
        count_word = "image" if image_count == 1 else "images"
        self._folder_count_label.setText(
            f"{image_count} {count_word} in this folder."
        )
        self._folder_count_label.setVisible(True)

        self._maybe_rebuild_dataset_summary()
        self._update_start_button_state()
        return True

    def _uses_target_normalisation(self) -> bool:
        """True when we should z-score the labels. Fresh runs use raw percents. Continuing an old z-scored checkpoint keeps that scale so the weights still make sense."""
        continuing = bool(
            self._continue_checkbox is not None
            and self._continue_checkbox.isChecked()
        )
        if continuing and self._parent_model_meta:
            return bool(self._parent_model_meta.get("normalise_targets", True))
        return False

    def _dataset_kwargs(self, *, normalise: Optional[bool] = None) -> Dict:
        image_size = IMAGE_SIZE
        legacy_crop = False
        if (
            self._continue_checkbox is not None
            and self._continue_checkbox.isChecked()
            and self._parent_model_meta
        ):
            image_size = image_size_from_metadata(self._parent_model_meta)
            legacy_crop = is_legacy_resnet18(self._parent_model_meta)
        if normalise is None:
            normalise = self._uses_target_normalisation()
        return {
            "csv_path": self._csv_path,
            "image_dir": self._image_dir,
            "val_fraction": VAL_FRACTION,
            "test_fraction": TEST_FRACTION,
            "normalise": normalise,
            "seed": 42,
            "split_mode": DEFAULT_SPLIT_MODE,
            "image_size": image_size,
            "legacy_crop": legacy_crop,
        }

    def _maybe_rebuild_dataset_summary(self) -> None:
        """Rematches filenames and refreshes the summary cards after the labels file, folder, or split changes."""
        if self._csv_dataframe is None or not self._image_dir:
            self._match_card.setVisible(False)
            self._step3_frame.setVisible(False)
            self._train_dataset = None
            self._val_dataset = None
            self._test_dataset = None
            return

        kwargs = self._dataset_kwargs()
        try:
            train_dataset = RegressionDataset(split="train", **kwargs)
            val_dataset = RegressionDataset(split="val", **kwargs)
            test_dataset = RegressionDataset(split="test", **kwargs)
        except Exception as exc:  # noqa: BLE001
            # Matching can fail on a bad path or a labels file that will not open.
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
            output_ranges = self._compute_full_output_ranges(
                train_dataset.matched
            )
            pan_distribution = self._compute_full_pan_distribution(
                train_dataset.matched
            )
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
                image_size=kwargs["image_size"],
            )
            self._step3_frame.setVisible(True)
        else:
            self._step3_frame.setVisible(False)

    @staticmethod
    def _compute_full_output_ranges(
        matched: List[Dict],
    ) -> Dict[str, Dict[str, float]]:
        ranges: Dict[str, Dict[str, float]] = {}
        for key, label in (
            ("water", "Water"),
            ("solids", "Solids"),
            ("bitumen", "Bitumen"),
        ):
            values = [item[key] for item in matched]
            if values:
                ranges[label] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                }
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

    # Grey out Start until labels, photos, and a model name are all there.

    def _update_start_button_state(self) -> None:
        if self._thread is not None:
            # Training is already running. The start button stays disabled until it finishes.
            return

        matched_count = (
            self._train_dataset.get_match_summary()["matched"]
            if self._train_dataset
            else 0
        )
        ready = (
            self._csv_dataframe is not None
            and self._image_dir is not None
            and matched_count > 0
            and bool(self._model_name_edit.text().strip())
        )
        self._start_button.setEnabled(ready)
        if not ready:
            self._start_button.setToolTip("Add labels and photos first")
        else:
            self._start_button.setToolTip("")

    # Status line under the start button.

    def _clear_status(self) -> None:
        self._status_label.setText("")
        self._status_label.setVisible(False)

    def _show_status_error(self, message: str) -> None:
        self._status_label.setText(message)
        self._status_label.setVisible(True)

    # Start, stop, and clean up a training thread.

    def _on_start_training(self) -> None:
        if self._thread is not None:
            return

        self._clear_status()

        model_name = self._model_name_edit.text().strip()
        if not model_name:
            self._show_status_error("Enter a model name.")
            return
        if any(character in model_name for character in ("/", "\\")):
            self._show_status_error(
                "Model name can't contain \u201c/\u201d or \u201c\\\u201d."
            )
            return

        if self._csv_dataframe is None or self._image_dir is None:
            self._show_status_error("Add labels and a photo folder first.")
            return

        continuing = bool(
            self._continue_checkbox is not None
            and self._continue_checkbox.isChecked()
        )
        parent_meta = self._parent_model_meta if continuing else None
        if continuing and not isinstance(parent_meta, dict):
            self._show_status_error(
                "Choose a saved model to continue training."
            )
            return
        if continuing:
            parent_path = parent_meta.get("model_path")
            if not parent_path or not Path(parent_path).exists():
                self._show_status_error(
                    "Couldn't find the saved model weights to continue from."
                )
                return

        kwargs = self._dataset_kwargs()

        try:
            train_dataset = RegressionDataset(split="train", **kwargs)
            val_dataset = RegressionDataset(split="val", **kwargs)
            test_dataset = RegressionDataset(split="test", **kwargs)
        except Exception as exc:  # noqa: BLE001
            # Tell the user if the dataset would not build.
            self._show_status_error(f"Couldn't prepare dataset: {exc}")
            return

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            self._show_status_error(
                "Not enough matched images. Add more, or check CSV filenames."
            )
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Decode photos in worker processes instead of holding everything in RAM.
        num_workers = _default_num_workers()
        loader_kwargs = {"num_workers": num_workers}
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
        if device.type == "cuda":
            loader_kwargs["pin_memory"] = True

        batch_size = self._current_batch_size()
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
        )
        test_loader = None
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                **loader_kwargs,
            )

        architecture = self._current_architecture()
        head = self._current_head()
        adaptation = self._current_adaptation()
        pretrained = (
            self._is_transfer_architecture(architecture) and not continuing
        )

        torch.manual_seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(42)

        try:
            if continuing:
                architecture = parent_meta.get("architecture", architecture)
                head = parent_meta.get("head", head)
                model = BitumenRegressor.from_checkpoint(
                    parent_meta["model_path"], parent_meta, device
                )
                model.train()
            else:
                model = BitumenRegressor(
                    architecture=architecture, pretrained=pretrained, head=head
                )
        except Exception as exc:  # noqa: BLE001
            # Usually a bad architecture name or a checkpoint that will not load.
            self._show_status_error(f"Couldn't load model: {exc}")
            return

        trainer = RegressionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=self._current_learning_rate(adaptation),
            num_epochs=self._epochs_spin.value()
            if self._epochs_spin is not None
            else NUM_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            output_stats=train_dataset.get_output_stats(),
            normalise_targets=bool(kwargs["normalise"]),
            patience=0,
            test_loader=test_loader,
            adaptation=adaptation,
            bin_edges=train_dataset.get_bin_edges(),
            init_output_bias=not continuing,
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
            "parent_model": (parent_meta or {}).get("name")
            if continuing
            else None,
            "parent_model_path": (parent_meta or {}).get("model_path")
            if continuing
            else None,
            "recipe": "prince_prasad_table2",
            "normalise_targets": bool(kwargs["normalise"]),
            "epochs": trainer.num_epochs,
            "batch_size": batch_size,
            "learning_rate": trainer.learning_rate,
            "weight_decay": trainer.weight_decay,
            "patience": trainer.patience,
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
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_mae_dict: dict,
        val_sum_deviation: float,
        val_r2_dict: dict,
    ) -> None:
        self._progress_panel.update_progress(
            epoch,
            train_loss,
            val_loss,
            val_mae_dict,
            val_sum_deviation,
            val_r2_dict,
        )

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
                result.final_epoch,
                result.best_val_mae,
                test_mae=result.test_mae,
                best_val_r2=result.best_val_r2,
                test_r2=result.test_r2,
            )
        else:
            self._progress_panel.show_completion(
                self._pending_model_name,
                result.best_val_mae,
                test_mae=result.test_mae,
                best_val_r2=result.best_val_r2,
                test_r2=result.test_r2,
            )

        if self.main_window is not None:
            try:
                metadata = load_model_metadata(paths["metadata_path"])
            except (OSError, ValueError):
                metadata = None
            self.main_window.set_active_model(
                str(paths["model_path"]), metadata
            )

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
            # Re-enable Start based on whether we still have a matched dataset.
            self._update_start_button_state()
            if (
                self._continue_checkbox is not None
                and self._continue_checkbox.isChecked()
            ):
                self._on_continue_model_changed()

        for widget in (
            self._csv_drop_zone,
            self._folder_drop_zone,
            self._model_name_edit,
            self._continue_checkbox,
            self._continue_combo,
            self._architecture_combo,
            self._adaptation_combo,
            self._head_combo,
            self._epochs_spin,
            self._batch_size_spin,
            self._lr_spin,
        ):
            if widget is not None:
                widget.setEnabled(not active)
