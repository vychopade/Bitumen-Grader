"""
Grade Images page.

Run the active model on sample photos and show Water / Solids / Bitumen
predictions, a sum-to-~100% check, and a rough closest-batch (Pan) guess.
"""
from __future__ import annotations

import itertools
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.components.loading_overlay import LoadingOverlay
from app.constants import OUTPUT_NAMES

from app.pages.grade_widgets import (
    PAN_GRADE_COLORS,
    PAN_GRADE_TEXT_COLORS,
    _AdaptiveImageLabel,
    _approx_output_range,
    _ClickableBanner,
    _closest_pan_grade,
    _ExportWorker,
    _GradingWorker,
    _ImageDropZone,
    _PanGradeCard,
    _QueueImage,
    _QueueItemWidget,
    _QueueList,
    _RangeBar,
)
from app.theme import (
    BACKGROUND_COLOR,
    BORDER_COLOR,
    DANGER_COLOR,
    PAGE_MARGINS,
    PAGE_SPACING,
    SUCCESS_COLOR,
    SURFACE_COLOR,
    SURFACE_HOVER_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    accent_button_qss,
    ghost_button_qss,
    LABEL_RESET_QSS,
    link_button_qss,
    sum_deviation_color,
)
from app.utils.model_io import format_created_at, format_r2_headline
from app.utils.files import unique_paths
from app.utils.image_utils import load_rgb_image
from app.utils.media import is_image_path

if TYPE_CHECKING:
    from app.main_window import MainWindow

LEFT_PANEL_WIDTH = 340


class GradePage(QWidget):
    """Grade sample images with the active model.

    Images arrive from Import's "Send to Grading", via Add Images, or by
    drop. Grade one selected image or the whole queue; results show as a
    single-image view or a batch table you can export to CSV.
    """

    def __init__(self, main_window: Optional["MainWindow"] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = main_window

        self._queue: List[_QueueImage] = []
        self._id_counter = itertools.count(1)
        self._selected_id: Optional[int] = None
        #: "single" = not-graded / one result; "batch" = Grade All summary table.
        self._view_mode = "single"
        self._batch_row_queue_ids: List[int] = []

        self._thread: Optional[QThread] = None
        self._worker: Optional[_GradingWorker] = None
        self._pending_output_stats: Dict[str, Dict[str, float]] = {}
        self._pending_grade_all = False
        self._export_thread: Optional[QThread] = None
        self._export_worker: Optional[_ExportWorker] = None

        self._model_value_label: Optional[QLabel] = None
        self._change_model_button: Optional[QPushButton] = None
        self._error_label: Optional[QLabel] = None

        self._drop_zone: Optional[_ImageDropZone] = None
        self._queue_list: Optional[_QueueList] = None
        self._queue_status_label: Optional[QLabel] = None
        self._add_images_button: Optional[QPushButton] = None
        self._grade_all_button: Optional[QPushButton] = None
        self._clear_all_button: Optional[QPushButton] = None

        self._right_stack: Optional[QStackedWidget] = None
        self._no_model_page: Optional[QWidget] = None
        self._not_graded_page: Optional[QWidget] = None
        self._single_result_page: Optional[QWidget] = None
        self._batch_results_page: Optional[QWidget] = None

        self._not_graded_preview: Optional[_AdaptiveImageLabel] = None
        self._not_graded_placeholder: Optional[QLabel] = None
        self._grade_this_button: Optional[QPushButton] = None

        self._single_preview: Optional[_AdaptiveImageLabel] = None
        self._measurement_table: Optional[QTableWidget] = None
        self._range_bars_container: Optional[QVBoxLayout] = None
        self._pan_grade_card: Optional[_PanGradeCard] = None

        self._batch_summary_labels: Dict[str, QLabel] = {}
        self._batch_table: Optional[QTableWidget] = None
        self._export_button: Optional[QPushButton] = None
        self._clear_results_button: Optional[QPushButton] = None

        self._build_ui()
        self._loading_overlay = LoadingOverlay(self)

        if self.main_window is not None:
            self.main_window.active_model_changed.connect(self._on_active_model_changed)

        self._refresh_top_bar()
        self._refresh_right_column()
        self._update_action_buttons_enabled()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(*PAGE_MARGINS)
        root.setSpacing(PAGE_SPACING)

        root.addLayout(self._build_header())

        self._error_label = QLabel("")
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet(f"color: {DANGER_COLOR}; font-size: 12px; background: transparent;")
        self._error_label.setVisible(False)
        root.addWidget(self._error_label)

        content_row = QHBoxLayout()
        content_row.setSpacing(20)
        content_row.addWidget(self._build_left_panel())
        content_row.addWidget(self._build_right_panel(), 1)
        root.addLayout(content_row, 1)

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setSpacing(12)

        title = QLabel("Grade")
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 16px; background: transparent;")
        header.addWidget(title)

        self._model_value_label = QLabel("")
        self._model_value_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; background: transparent;")
        header.addWidget(self._model_value_label, 1)

        return header

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        container.setFixedWidth(LEFT_PANEL_WIDTH)
        container.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QLabel("Queue")
        header.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; background: transparent;")
        layout.addWidget(header)

        self._drop_zone = _ImageDropZone()
        self._drop_zone.files_selected.connect(self._add_images)
        layout.addWidget(self._drop_zone)

        self._queue_list = _QueueList()
        self._queue_list.setStyleSheet(
            f"""
            QListWidget {{
                background-color: {SURFACE_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 3px;
            }}
            QListWidget::item {{ padding: 0px; border-bottom: 1px solid {BORDER_COLOR}; }}
            QListWidget::item:last {{ border-bottom: none; }}
            QListWidget::item:selected {{ background-color: {SURFACE_HOVER_COLOR}; }}
            """
        )
        self._queue_list.files_dropped.connect(self._add_images)
        self._queue_list.currentItemChanged.connect(self._on_queue_selection_changed)
        layout.addWidget(self._queue_list, 1)

        self._queue_status_label = QLabel("0 images loaded")
        self._queue_status_label.setWordWrap(True)
        self._queue_status_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;"
        )
        layout.addWidget(self._queue_status_label)

        self._grade_all_button = QPushButton("Grade all")
        self._grade_all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._grade_all_button.setStyleSheet(accent_button_qss())
        self._grade_all_button.clicked.connect(self._on_grade_all)
        layout.addWidget(self._grade_all_button)

        self._clear_all_button = QPushButton("Clear queue")
        self._clear_all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_all_button.setStyleSheet(link_button_qss())
        self._clear_all_button.clicked.connect(self._on_clear_all)
        layout.addWidget(self._clear_all_button, 0, Qt.AlignmentFlag.AlignLeft)

        return container

    def _build_right_panel(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._right_stack = QStackedWidget()
        self._no_model_page = self._build_no_model_page()
        self._not_graded_page = self._build_not_graded_page()
        self._single_result_page = self._build_single_result_page()
        self._batch_results_page = self._build_batch_results_page()

        self._right_stack.addWidget(self._no_model_page)
        self._right_stack.addWidget(self._not_graded_page)
        self._right_stack.addWidget(self._single_result_page)
        self._right_stack.addWidget(self._batch_results_page)

        layout.addWidget(self._right_stack, 1)
        return container

    def _build_no_model_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        banner = _ClickableBanner()
        banner.setObjectName("noModelBanner")
        banner.setStyleSheet(
            # Scoped to #noModelBanner rather than the bare "QFrame" type
            # selector -- QLabel is itself a QFrame subclass in Qt, so an
            # unscoped rule here would also draw this border around the
            # word-wrapped message label nested inside, not just the banner.
            f"QFrame#noModelBanner {{ background-color: {SURFACE_COLOR}; border: 1px solid {BORDER_COLOR};"
            f"border-radius: 3px; }}"
        )
        banner.clicked.connect(self._navigate_to_models)

        banner_layout = QVBoxLayout(banner)
        banner_layout.setContentsMargins(24, 20, 24, 20)

        message = QLabel("No model loaded. Open Models in the sidebar.")
        message.setWordWrap(True)
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px; background: transparent;")
        banner_layout.addWidget(message)

        layout.addWidget(banner)
        return page

    def _build_not_graded_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background-color: {SURFACE_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 3px; }}"
            f"{LABEL_RESET_QSS}"
        )
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(18, 18, 18, 18)
        frame_layout.setSpacing(14)

        self._not_graded_preview = _AdaptiveImageLabel()
        self._not_graded_preview.setStyleSheet(f"background-color: {BACKGROUND_COLOR}; border-radius: 3px;")
        frame_layout.addWidget(self._not_graded_preview, 1)

        self._not_graded_placeholder = QLabel("Select an image from the queue to preview it.")
        self._not_graded_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._not_graded_placeholder.setWordWrap(True)
        self._not_graded_placeholder.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px; background: transparent;")
        frame_layout.addWidget(self._not_graded_placeholder)

        self._grade_this_button = QPushButton("Grade this photo")
        self._grade_this_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._grade_this_button.setStyleSheet(ghost_button_qss())
        self._grade_this_button.clicked.connect(self._on_grade_this_image)
        self._grade_this_button.setEnabled(False)
        frame_layout.addWidget(self._grade_this_button, 0, Qt.AlignmentFlag.AlignHCenter)

        layout.addWidget(frame, 1)
        return page

    def _build_single_result_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        preview_frame = QFrame()
        preview_frame.setStyleSheet(
            f"QFrame {{ background-color: {SURFACE_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 3px; }}"
            f"{LABEL_RESET_QSS}"
        )
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(14, 14, 14, 14)
        self._single_preview = _AdaptiveImageLabel()
        self._single_preview.setStyleSheet(f"background-color: {BACKGROUND_COLOR}; border-radius: 3px;")
        preview_layout.addWidget(self._single_preview)
        layout.addWidget(preview_frame, 1)

        results_frame = QFrame()
        results_frame.setStyleSheet(
            f"QFrame {{ background-color: {SURFACE_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 3px; }}"
            f"{LABEL_RESET_QSS}"
        )
        results_layout = QVBoxLayout(results_frame)
        results_layout.setContentsMargins(18, 16, 18, 16)
        results_layout.setSpacing(12)

        self._measurement_table = self._build_measurement_table()
        results_layout.addWidget(self._measurement_table)

        self._range_bars_container = QVBoxLayout()
        self._range_bars_container.setSpacing(4)
        results_layout.addLayout(self._range_bars_container)

        self._pan_grade_card = _PanGradeCard()
        results_layout.addWidget(self._pan_grade_card)

        layout.addWidget(results_frame, 1)
        return page

    def _build_measurement_table(self) -> QTableWidget:
        table = QTableWidget(4, 3)
        table.setHorizontalHeaderLabels(["Measurement", "Predicted", "Training avg"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setFixedHeight(165)
        table.horizontalHeader().setStretchLastSection(True)
        table.setStyleSheet(
            f"""
            QTableWidget {{
                background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR}; border-radius: 6px; gridline-color: {BORDER_COLOR};
            }}
            QTableWidget::item {{ padding: 4px; }}
            QHeaderView::section {{
                background-color: {SURFACE_COLOR}; color: {TEXT_SECONDARY}; border: none;
                padding: 6px; font-size: 10px; font-weight: 600;
            }}
            """
        )

        row_labels = ("Water    (%)", "Solids   (%)", "Bitumen  (%)", "Sum")
        for row, label in enumerate(row_labels):
            item = QTableWidgetItem(label)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if label == "Sum":
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            table.setItem(row, 0, item)

        return table

    def _build_batch_results_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        self._build_batch_summary_card()
        layout.addWidget(self._batch_summary_card)

        self._batch_table = self._build_batch_table()
        layout.addWidget(self._batch_table, 1)

        action_row = QHBoxLayout()
        action_row.setSpacing(12)

        self._export_button = QPushButton("Export CSV")
        self._export_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_button.setStyleSheet(ghost_button_qss())
        self._export_button.clicked.connect(self._on_export_results)
        action_row.addWidget(self._export_button)

        self._clear_results_button = QPushButton("Clear results")
        self._clear_results_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_results_button.setStyleSheet(link_button_qss())
        self._clear_results_button.clicked.connect(self._on_clear_results)
        action_row.addWidget(self._clear_results_button)

        action_row.addStretch(1)
        layout.addLayout(action_row)

        return page

    def _build_batch_summary_card(self) -> None:
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background-color: {SURFACE_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 3px; }}"
            f"{LABEL_RESET_QSS}"
        )

        layout = QHBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(24)

        for title in ("Images graded", "Avg Water", "Avg Solids", "Avg Bitumen", "Avg sum off-by", "Sum warnings (>5%)"):
            self._batch_summary_labels[title] = self._build_stat_block(layout, title)

        self._batch_summary_card = card

    def _build_stat_block(self, layout: QHBoxLayout, title: str) -> QLabel:
        wrapper = QWidget()
        wrapper.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        block = QVBoxLayout(wrapper)
        block.setContentsMargins(0, 0, 0, 0)
        block.setSpacing(2)

        title_label = QLabel(title)
        title_label.setWordWrap(True)
        title_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        title_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        block.addWidget(title_label)

        value_label = QLabel("\u2014")
        value_label.setWordWrap(True)
        value_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 18px; font-weight: 700; background: transparent;"
        )
        block.addWidget(value_label)

        layout.addWidget(wrapper, 1)
        return value_label

    def _build_batch_table(self) -> QTableWidget:
        table = QTableWidget(0, 8)
        table.setHorizontalHeaderLabels(["#", "Filename", "Water", "Solids", "Bitumen", "Sum", "Sum OK", "Batch"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.horizontalHeader().setStretchLastSection(True)
        table.itemSelectionChanged.connect(self._on_batch_table_selection_changed)
        table.setStyleSheet(
            f"""
            QTableWidget {{
                background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR}; border-radius: 6px; gridline-color: {BORDER_COLOR};
            }}
            QTableWidget::item {{ padding: 4px; }}
            QTableWidget::item:selected {{ background-color: {SURFACE_HOVER_COLOR}; color: {TEXT_PRIMARY}; }}
            QHeaderView::section {{
                background-color: {SURFACE_COLOR}; color: {TEXT_SECONDARY}; border: none;
                padding: 6px; font-size: 10px; font-weight: 600;
            }}
            """
        )
        return table

    # -- Model selector / navigation ----------------------------------------

    def _on_active_model_changed(self, _active_model: Optional[Dict[str, Any]]) -> None:
        self._refresh_top_bar()
        self._refresh_right_column()
        self._update_action_buttons_enabled()

    def _refresh_top_bar(self) -> None:
        active_model = getattr(self.main_window, "active_model", None) if self.main_window else None

        if not active_model:
            self._model_value_label.setText("No model — open Models in the sidebar")
            return

        metadata = active_model.get("metadata") or {}
        name = metadata.get("name") or Path(active_model.get("path", "")).stem
        date_str = format_created_at(metadata.get("created_at"))
        r2_str = format_r2_headline(metadata)
        parts = [name]
        if r2_str:
            parts.append(r2_str)
        if date_str:
            parts.append(date_str)
        self._model_value_label.setText("  ·  ".join(parts))

    def _navigate_to_models(self) -> None:
        if self.main_window is not None:
            self.main_window.navigate_to("models")

    # -- Queue management -----------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: D401 - Qt override
        super().showEvent(event)
        self._refresh_top_bar()
        self._refresh_right_column()

    def _add_images(self, paths: List[str]) -> None:
        if not paths:
            self._error_label.setText("No JPG/PNG/TIF photos in that folder or selection.")
            self._error_label.setVisible(True)
            return

        existing_paths = {q.path for q in self._queue}
        added = unique_paths(existing_paths, (p for p in paths if is_image_path(p)))
        if not added:
            return

        for path in added:
            self._add_queue_image(path)
        self._queue_list.relayout_rows()

        self._error_label.setVisible(False)
        self._update_queue_status_label()
        self._update_action_buttons_enabled()
        if self._selected_id is None and self._queue:
            self._queue_list.setCurrentItem(self._queue[0].item)

    def _show_image_load_error(self, failed_names: List[str]) -> None:
        names = ", ".join(failed_names[:3])
        if len(failed_names) > 3:
            names += f", and {len(failed_names) - 3} more"
        count_word = "image" if len(failed_names) == 1 else "images"
        self._error_label.setText(f"Couldn't load {len(failed_names)} {count_word}: {names}")
        self._error_label.setVisible(True)

    def _add_queue_image(self, path: str) -> None:
        widget = _QueueItemWidget(Path(path).name)
        list_item = QListWidgetItem()
        self._queue_list.addItem(list_item)
        self._queue_list.setItemWidget(list_item, widget)

        queue_image = _QueueImage(id=next(self._id_counter), path=path, item=list_item, widget=widget)
        self._queue.append(queue_image)

    def _update_queue_status_label(self) -> None:
        count = len(self._queue)
        count_word = "image" if count == 1 else "images"
        self._queue_status_label.setText(f"{count} {count_word} loaded")

    def _find_queue_item(self, image_id: Optional[int]) -> Optional[_QueueImage]:
        if image_id is None:
            return None
        return next((q for q in self._queue if q.id == image_id), None)

    def _select_queue_item(self, image_id: int) -> None:
        queue_image = self._find_queue_item(image_id)
        if queue_image is not None:
            self._queue_list.setCurrentItem(queue_image.item)

    def _on_queue_selection_changed(self, current: Optional[QListWidgetItem], _previous) -> None:
        if current is None:
            self._selected_id = None
        else:
            matching = next((q for q in self._queue if q.item is current), None)
            self._selected_id = matching.id if matching is not None else None
        self._view_mode = "single"
        self._refresh_right_column()

    def _on_clear_all(self) -> None:
        self._queue_list.clear()
        self._queue = []
        self._selected_id = None
        self._view_mode = "single"
        self._error_label.setVisible(False)
        self._update_queue_status_label()
        self._update_action_buttons_enabled()
        self._refresh_right_column()

    # -- Grading ---------------------------------------------------------------

    def _on_grade_this_image(self) -> None:
        if self._thread is not None:
            return
        queue_image = self._find_queue_item(self._selected_id)
        if queue_image is None:
            return

        active_model = getattr(self.main_window, "active_model", None) if self.main_window else None
        if not active_model:
            self._refresh_right_column()
            return

        self._start_grading_job(active_model, [queue_image], grade_all=False)

    def _on_grade_all(self) -> None:
        if not self._queue or self._thread is not None:
            return

        active_model = getattr(self.main_window, "active_model", None) if self.main_window else None
        if not active_model:
            self._refresh_right_column()
            return

        self._start_grading_job(active_model, list(self._queue), grade_all=True)

    def _start_grading_job(self, active_model: Dict[str, Any], images_to_grade: List[_QueueImage], grade_all: bool) -> None:
        self._error_label.setVisible(False)

        predictor = active_model.get("predictor")
        metadata = active_model.get("metadata") or {}
        if predictor is None:
            self._error_label.setText("No model loaded.")
            self._error_label.setVisible(True)
            return

        self._pending_output_stats = metadata.get("output_stats") or {}
        self._pending_grade_all = grade_all
        payload = [(queue_image.id, queue_image.path) for queue_image in images_to_grade]

        self._loading_overlay.show_message("Grading all images\u2026" if grade_all else "Grading\u2026")
        self._set_grading_active(True)

        self._thread = QThread(self)
        self._worker = _GradingWorker(predictor, payload)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_grading_progress)
        self._worker.finished.connect(self._on_grading_finished)
        self._worker.failed.connect(self._on_grading_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._on_grading_thread_finished)

        self._thread.start()

    def _on_grading_progress(self, done: int, total: int) -> None:
        if total <= 1:
            return
        self._loading_overlay.set_message(f"Grading {done} of {total}\u2026")

    def _set_grading_active(self, active: bool) -> None:
        """Disable controls while grading is running."""
        for button in (
            self._grade_all_button,
            self._clear_all_button,
            self._grade_this_button,
        ):
            if button is not None:
                button.setEnabled(not active)
        if active:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _on_grading_finished(self, results: List[Any], failures: int) -> None:
        results_by_id = dict(results)
        for queue_image in self._queue:
            if queue_image.id in results_by_id:
                result = results_by_id[queue_image.id]
                if result is not None:
                    queue_image.result = result
                    queue_image.output_stats = self._pending_output_stats

        self._loading_overlay.hide_overlay()
        self._set_grading_active(False)

        if failures:
            count_word = "image" if failures == 1 else "images"
            self._error_label.setText(f"Couldn't grade {failures} {count_word}; other results are below.")
            self._error_label.setVisible(True)

        self._update_queue_status_label()
        self._update_action_buttons_enabled()

        if self._pending_grade_all:
            self._view_mode = "batch"
            self._refresh_batch_summary()
            self._refresh_batch_table()

        self._refresh_right_column()

    def _on_grading_failed(self, message: str) -> None:
        self._loading_overlay.hide_overlay()
        self._set_grading_active(False)
        self._handle_model_load_failure(RuntimeError(message))

    def _on_grading_thread_finished(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
        if self._worker is not None:
            self._worker.deleteLater()
        self._thread = None
        self._worker = None

    def _handle_model_load_failure(self, exc: Exception) -> None:
        """Clear the active model and show a dialog if grading blows up mid-run.

        ``set_active_model`` already checks loads up front; this is a fallback
        if a loaded model becomes unusable later (e.g. file deleted).
        """
        if self.main_window is not None:
            self.main_window.set_active_model(None)

        message = str(exc) or exc.__class__.__name__
        self._error_label.setText(f"Couldn't load model: {message}")
        self._error_label.setVisible(True)
        self._refresh_top_bar()
        self._refresh_right_column()

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Model Load Failed")
        box.setText("That model couldn't be loaded, so it's been cleared.")
        box.setInformativeText(message)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    # -- Right column: single-image view -------------------------------------

    def _refresh_right_column(self) -> None:
        active_model = getattr(self.main_window, "active_model", None) if self.main_window else None
        if not active_model:
            self._right_stack.setCurrentWidget(self._no_model_page)
            return

        if self._view_mode == "batch":
            self._right_stack.setCurrentWidget(self._batch_results_page)
            return

        queue_image = self._find_queue_item(self._selected_id)

        if queue_image is None:
            self._not_graded_preview.set_source_image(None)
            self._not_graded_placeholder.setText("Select an image from the queue to preview it.")
            self._not_graded_placeholder.setVisible(True)
            self._grade_this_button.setEnabled(False)
            self._right_stack.setCurrentWidget(self._not_graded_page)
            return

        if queue_image.result is None:
            self._not_graded_preview.set_source_image(self._preview_image(queue_image.path))
            self._not_graded_placeholder.setVisible(False)
            self._grade_this_button.setEnabled(True)
            self._right_stack.setCurrentWidget(self._not_graded_page)
            return

        self._single_preview.set_source_image(self._preview_image(queue_image.path))
        self._update_measurement_table(queue_image.result, queue_image.output_stats or {})
        self._update_range_bars(queue_image.result, queue_image.output_stats or {})
        self._update_pan_grade_card(queue_image.result, queue_image.output_stats or {})
        self._right_stack.setCurrentWidget(self._single_result_page)

    def _preview_image(self, path: str):
        try:
            return load_rgb_image(path)
        except (OSError, ValueError):
            return None

    def _update_measurement_table(self, result: Dict[str, Any], output_stats: Dict[str, Dict[str, float]]) -> None:
        training_avgs = {label: output_stats.get(label, {}).get("mean", 0.0) for label in OUTPUT_NAMES}
        training_sum = sum(training_avgs.values())

        for row, label in enumerate(OUTPUT_NAMES):
            predicted_value = result[label]["value"]
            pred_item = QTableWidgetItem(f"{predicted_value:.2f}")
            avg_item = QTableWidgetItem(f"{training_avgs[label]:.2f}")
            for item in (pred_item, avg_item):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._measurement_table.setItem(row, 1, pred_item)
            self._measurement_table.setItem(row, 2, avg_item)

        sum_deviation = result["sum_deviation"]
        color = sum_deviation_color(sum_deviation)

        sum_pred_item = QTableWidgetItem(f"{result['sum']:.2f}")
        sum_avg_item = QTableWidgetItem(f"{training_sum:.2f}")
        for item in (sum_pred_item, sum_avg_item):
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setForeground(QColor(color))
            font = item.font()
            font.setBold(True)
            item.setFont(font)
        self._measurement_table.setItem(3, 1, sum_pred_item)
        self._measurement_table.setItem(3, 2, sum_avg_item)

    def _update_range_bars(self, result: Dict[str, Any], output_stats: Dict[str, Dict[str, float]]) -> None:
        while self._range_bars_container.count():
            taken = self._range_bars_container.takeAt(0)
            widget = taken.widget()
            if widget is not None:
                widget.deleteLater()

        for label in OUTPUT_NAMES:
            stats = output_stats.get(label, {"mean": 0.0, "std": 0.0})
            low, high = _approx_output_range(stats.get("mean", 0.0), stats.get("std", 0.0))
            value = result[label]["value"]
            bar = _RangeBar(f"{label} (%)", low, high, value)
            self._range_bars_container.addWidget(bar)

    def _update_pan_grade_card(self, result: Dict[str, Any], output_stats: Dict[str, Dict[str, float]]) -> None:
        bitumen_stats = output_stats.get("Bitumen", {"mean": 0.0, "std": 0.0})
        grade = _closest_pan_grade(result["Bitumen"]["value"], bitumen_stats.get("mean", 0.0), bitumen_stats.get("std", 0.0))
        self._pan_grade_card.set_grade(grade)

    # -- Right column: batch results view -------------------------------------

    def _graded_queue_images(self) -> List[_QueueImage]:
        return [q for q in self._queue if q.result is not None]

    def _refresh_batch_summary(self) -> None:
        graded = self._graded_queue_images()
        if not graded:
            return

        count = len(graded)
        mean_water = sum(q.result["Water"]["value"] for q in graded) / count
        mean_solids = sum(q.result["Solids"]["value"] for q in graded) / count
        mean_bitumen = sum(q.result["Bitumen"]["value"] for q in graded) / count
        mean_sum_deviation = sum(q.result["sum_deviation"] for q in graded) / count
        warnings = sum(1 for q in graded if q.result["sum_deviation"] > 5.0)

        self._batch_summary_labels["Images graded"].setText(str(count))
        self._batch_summary_labels["Avg Water"].setText(f"{mean_water:.2f}%")
        self._batch_summary_labels["Avg Solids"].setText(f"{mean_solids:.2f}%")
        self._batch_summary_labels["Avg Bitumen"].setText(f"{mean_bitumen:.2f}%")
        self._batch_summary_labels["Avg sum off-by"].setText(f"{mean_sum_deviation:.2f}%")
        self._batch_summary_labels["Sum warnings (>5%)"].setText(str(warnings))

    def _refresh_batch_table(self) -> None:
        graded = self._graded_queue_images()
        table = self._batch_table
        table.setRowCount(len(graded))
        self._batch_row_queue_ids = [q.id for q in graded]

        for row, queue_image in enumerate(graded):
            result = queue_image.result
            stats = queue_image.output_stats or {}
            bitumen_stats = stats.get("Bitumen", {"mean": 0.0, "std": 0.0})
            grade = _closest_pan_grade(
                result["Bitumen"]["value"], bitumen_stats.get("mean", 0.0), bitumen_stats.get("std", 0.0)
            )

            values = [
                str(row + 1),
                Path(queue_image.path).name,
                f"{result['Water']['value']:.2f}",
                f"{result['Solids']['value']:.2f}",
                f"{result['Bitumen']['value']:.2f}",
                f"{result['sum']:.2f}",
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if col >= 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, col, item)

            sum_ok = result["sum_ok"]
            sum_ok_item = QTableWidgetItem("\u2713" if sum_ok else "\u2717")
            sum_ok_item.setFlags(sum_ok_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            sum_ok_item.setForeground(QColor(SUCCESS_COLOR if sum_ok else DANGER_COLOR))
            sum_ok_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(row, 6, sum_ok_item)

            grade_item = QTableWidgetItem(str(grade))
            grade_item.setFlags(grade_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            grade_item.setBackground(QColor(PAN_GRADE_COLORS.get(grade, TEXT_SECONDARY)))
            grade_item.setForeground(QColor(PAN_GRADE_TEXT_COLORS.get(grade, TEXT_PRIMARY)))
            grade_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            font = grade_item.font()
            font.setBold(True)
            grade_item.setFont(font)
            table.setItem(row, 7, grade_item)

    def _on_batch_table_selection_changed(self) -> None:
        selection_model = self._batch_table.selectionModel()
        if selection_model is None:
            return
        selected_rows = selection_model.selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        if 0 <= row < len(self._batch_row_queue_ids):
            self._select_queue_item(self._batch_row_queue_ids[row])

    # -- Export / clear ----------------------------------------------------------

    def _on_export_results(self) -> None:
        if self._export_thread is not None:
            return

        graded = self._graded_queue_images()
        if not graded:
            return

        default_name = f"grading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        save_path, _ = QFileDialog.getSaveFileName(self, "Export Results", default_name, "CSV Files (*.csv)")
        if not save_path:
            return

        header = ["filename", "water", "solids", "bitumen", "sum", "sum_deviation", "sum_ok", "batch"]
        rows: List[List[str]] = []
        for queue_image in graded:
            result = queue_image.result
            stats = queue_image.output_stats or {}
            bitumen_stats = stats.get("Bitumen", {"mean": 0.0, "std": 0.0})
            grade = _closest_pan_grade(
                result["Bitumen"]["value"], bitumen_stats.get("mean", 0.0), bitumen_stats.get("std", 0.0)
            )
            rows.append(
                [
                    Path(queue_image.path).name,
                    f"{result['Water']['value']:.4f}",
                    f"{result['Solids']['value']:.4f}",
                    f"{result['Bitumen']['value']:.4f}",
                    f"{result['sum']:.4f}",
                    f"{result['sum_deviation']:.4f}",
                    "true" if result["sum_ok"] else "false",
                    str(grade),
                ]
            )

        self._error_label.setVisible(False)
        self._loading_overlay.show_message("Exporting\u2026")
        if self._export_button is not None:
            self._export_button.setEnabled(False)

        self._export_thread = QThread(self)
        self._export_worker = _ExportWorker(save_path, header, rows)
        self._export_worker.moveToThread(self._export_thread)

        self._export_thread.started.connect(self._export_worker.run)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.failed.connect(self._on_export_failed)
        self._export_worker.finished.connect(self._export_thread.quit)
        self._export_worker.failed.connect(self._export_thread.quit)
        self._export_thread.finished.connect(self._on_export_thread_finished)

        self._export_thread.start()

    def _on_export_finished(self) -> None:
        self._loading_overlay.hide_overlay()
        self._error_label.setVisible(False)
        self._update_action_buttons_enabled()

    def _on_export_failed(self, message: str) -> None:
        self._loading_overlay.hide_overlay()
        self._error_label.setText(f"Couldn't export results: {message}")
        self._error_label.setVisible(True)
        self._update_action_buttons_enabled()

    def _on_export_thread_finished(self) -> None:
        if self._export_thread is not None:
            self._export_thread.deleteLater()
        if self._export_worker is not None:
            self._export_worker.deleteLater()
        self._export_thread = None
        self._export_worker = None

    def _on_clear_results(self) -> None:
        for queue_image in self._queue:
            queue_image.result = None
            queue_image.output_stats = None

        self._view_mode = "single"
        self._update_queue_status_label()
        self._update_action_buttons_enabled()
        self._refresh_right_column()

    def _update_action_buttons_enabled(self) -> None:
        active_model = getattr(self.main_window, "active_model", None) if self.main_window else None
        has_images = bool(self._queue)
        has_results = any(q.result is not None for q in self._queue)

        if self._grade_all_button is not None:
            self._grade_all_button.setEnabled(has_images and bool(active_model))
        if self._clear_all_button is not None:
            self._clear_all_button.setEnabled(has_images)
        if self._export_button is not None:
            self._export_button.setEnabled(has_results)
        if self._clear_results_button is not None:
            self._clear_results_button.setEnabled(has_results)
