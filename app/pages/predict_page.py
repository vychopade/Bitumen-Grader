"""
Prediction / Grading page.

Provides the UI for running inference with a loaded model against new
bitumen sample images and displaying the predicted grade/classification
along with confidence scores and any relevant visualizations.
"""
from __future__ import annotations

import csv
import itertools
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PIL import Image
from PyQt6.QtCore import QObject, QRectF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFontMetrics,
    QPainter,
    QResizeEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.components.image_editor import pil_to_qpixmap
from app.components.loading_overlay import LoadingOverlay
from app.ml.predictor import ModelPredictor
from app.pages.model_manager_page import ModelManagerPage
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
BAR_INACTIVE_COLOR = "#3A3D45"

LEFT_PANEL_WIDTH = 340
THUMB_SIZE = 48
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif")


class _GradingWorker(QObject):
    """Loads a ``ModelPredictor`` and grades a batch of images on a background QThread.

    Loading a checkpoint (``torch.load``) and running inference for a batch of
    images can take longer than is comfortable on the UI thread, so both steps
    run here; the main thread only touches list-widget state once results come
    back via ``finished``.
    """

    finished = pyqtSignal(object, list, int)  # predictor, [(image_id, result_or_None)], failure_count
    failed = pyqtSignal(str)

    def __init__(
        self,
        cached_predictor: Optional[ModelPredictor],
        cached_model_path: Optional[str],
        model_path: str,
        grade_labels: Optional[List[str]],
        images: List[Any],
    ):
        super().__init__()
        self._cached_predictor = cached_predictor
        self._cached_model_path = cached_model_path
        self._model_path = model_path
        self._grade_labels = grade_labels
        self._images = images

    def run(self) -> None:
        if self._cached_predictor is not None and self._cached_model_path == self._model_path:
            predictor = self._cached_predictor
        else:
            try:
                predictor = ModelPredictor(self._model_path, grade_labels=self._grade_labels)
            except Exception as exc:  # noqa: BLE001 - surface any model-load error to the UI
                self.failed.emit(str(exc))
                return

        results: List[Any] = []
        failures = 0
        for image_id, pil_image in self._images:
            try:
                result = predictor.predict(pil_image)
            except Exception:  # noqa: BLE001 - keep grading remaining images
                failures += 1
                results.append((image_id, None))
                continue
            results.append((image_id, result))

        self.finished.emit(predictor, results, failures)


class _ExportWorker(QObject):
    """Writes graded results to a CSV file on a background QThread (file I/O)."""

    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, save_path: str, header: List[str], rows: List[List[str]]):
        super().__init__()
        self._save_path = save_path
        self._header = header
        self._rows = rows

    def run(self) -> None:
        try:
            with open(self._save_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(self._header)
                writer.writerows(self._rows)
        except OSError as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit()


@dataclass
class _QueueImage:
    """A single image queued for grading, plus the list widgets representing it."""

    id: int
    path: str
    image: Image.Image
    item: QListWidgetItem
    widget: "_QueueItemWidget"
    result: Optional[Dict[str, Any]] = None


class _QueueItemWidget(QWidget):
    """Row widget for one queued image: thumbnail + filename + remove button.

    After grading, ``show_result`` reveals a badge row with the predicted
    grade and confidence, effectively turning the row into a result badge.
    """

    remove_requested = pyqtSignal()

    def __init__(self, filename: str, pil_image: Image.Image, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        self._thumb_label = QLabel()
        self._thumb_label.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        self._thumb_label.setStyleSheet(f"background-color: {BACKGROUND_COLOR}; border-radius: 4px;")
        self._thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._set_thumbnail(pil_image)
        top_row.addWidget(self._thumb_label)

        name_label = QLabel()
        metrics = QFontMetrics(name_label.font())
        name_label.setText(metrics.elidedText(filename, Qt.TextElideMode.ElideMiddle, 150))
        name_label.setToolTip(filename)
        name_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; background: transparent;")
        top_row.addWidget(name_label, 1)

        remove_button = QPushButton("Remove")
        remove_button.setCursor(Qt.CursorShape.PointingHandCursor)
        remove_button.setToolTip(f"Remove {filename} from the queue")
        remove_button.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {DANGER_COLOR}; border: none;"
            f"font-size: 10px; padding: 2px; }}"
            f"QPushButton:hover {{ text-decoration: underline; }}"
        )
        remove_button.clicked.connect(self.remove_requested.emit)
        top_row.addWidget(remove_button)

        layout.addLayout(top_row)

        self._badge_container = QWidget()
        badge_row = QHBoxLayout(self._badge_container)
        badge_row.setContentsMargins(THUMB_SIZE + 10, 0, 0, 0)
        badge_row.setSpacing(6)

        self._grade_badge = QLabel("")
        self._grade_badge.setStyleSheet(
            f"background-color: {ACCENT_COLOR}; color: #13151A; font-size: 10px; font-weight: 700;"
            f"border-radius: 8px; padding: 2px 8px;"
        )
        badge_row.addWidget(self._grade_badge)

        self._confidence_chip = QLabel("")
        self._confidence_chip.setStyleSheet(
            f"background-color: {BACKGROUND_COLOR}; color: {TEXT_SECONDARY}; font-size: 10px;"
            f"border-radius: 8px; padding: 2px 8px;"
        )
        badge_row.addWidget(self._confidence_chip)
        badge_row.addStretch(1)

        self._badge_container.setVisible(False)
        layout.addWidget(self._badge_container)

    def _set_thumbnail(self, pil_image: Image.Image) -> None:
        pixmap = pil_to_qpixmap(pil_image).scaled(
            THUMB_SIZE,
            THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumb_label.setPixmap(pixmap)

    def show_result(self, grade: str, confidence: float) -> None:
        self._grade_badge.setText(grade)
        self._confidence_chip.setText(f"{confidence * 100:.1f}%")
        self._badge_container.setVisible(True)

    def clear_result(self) -> None:
        self._badge_container.setVisible(False)


class _QueueList(QListWidget):
    """QListWidget that also accepts external image files dropped onto it."""

    files_dropped = pyqtSignal(list)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        mime_data = event.mimeData()
        if mime_data.hasUrls() and any(
            url.isLocalFile() and url.toLocalFile().lower().endswith(SUPPORTED_EXTENSIONS)
            for url in mime_data.urls()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: D401 - Qt override
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if url.isLocalFile() and url.toLocalFile().lower().endswith(SUPPORTED_EXTENSIONS)
        ]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class _AdaptiveImageLabel(QLabel):
    """QLabel that keeps a source pixmap scaled to fill its current size."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._source_pixmap = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(180)

    def set_source_image(self, pil_image: Optional[Image.Image]) -> None:
        self._source_pixmap = pil_to_qpixmap(pil_image) if pil_image is not None else None
        self._rescale()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            self.clear()
            return
        scaled = self._source_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class _ConfidenceBar(QWidget):
    """A single labeled horizontal bar showing one grade's predicted probability."""

    def __init__(self, label: str, probability: float, is_winner: bool, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._label = label
        self._probability = max(0.0, min(1.0, probability))
        self._is_winner = is_winner
        self.setFixedHeight(22)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event) -> None:  # noqa: D401 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        label_width = 92
        pct_width = 48
        spacing = 10
        track_left = label_width + spacing
        track_right = max(track_left, self.width() - pct_width - spacing)
        track_width = track_right - track_left
        track_top = (self.height() - 10) / 2

        font = painter.font()
        font.setPointSize(9)

        font.setBold(self._is_winner)
        painter.setFont(font)
        painter.setPen(QColor(TEXT_PRIMARY if self._is_winner else TEXT_SECONDARY))
        painter.drawText(
            QRectF(0, 0, label_width, self.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self._label,
        )

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(BACKGROUND_COLOR))
        painter.drawRoundedRect(QRectF(track_left, track_top, track_width, 10), 5, 5)

        fill_width = track_width * self._probability
        if fill_width > 0:
            fill_color = ACCENT_COLOR if self._is_winner else BAR_INACTIVE_COLOR
            painter.setBrush(QColor(fill_color))
            painter.drawRoundedRect(QRectF(track_left, track_top, fill_width, 10), 5, 5)

        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(TEXT_SECONDARY))
        painter.drawText(
            QRectF(track_right + spacing, 0, pct_width, self.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            f"{self._probability * 100:.1f}%",
        )

        painter.end()


class PredictPage(QWidget):
    """Page for grading bitumen sample images with the active model.

    Images populate automatically from ``main_window.grading_images`` (sent
    over from ``ImageImportPage``'s "Send to Grading" action), can be
    supplemented via "Add Images", or dropped directly onto the queue list.
    "Grade All" runs the active model's ``ModelPredictor`` against every
    queued image; results can then be inspected per-image (large preview +
    confidence bar chart) or exported to CSV.
    """

    def __init__(self, main_window: Optional["MainWindow"] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = main_window

        self._queue: List[_QueueImage] = []
        self._id_counter = itertools.count(1)
        self._selected_id: Optional[int] = None
        self._predictor: Optional[ModelPredictor] = None
        self._predictor_model_path: Optional[str] = None

        self._thread: Optional[QThread] = None
        self._worker: Optional[_GradingWorker] = None
        self._pending_model_path: Optional[str] = None
        self._export_thread: Optional[QThread] = None
        self._export_worker: Optional[_ExportWorker] = None

        self._model_value_label: Optional[QLabel] = None
        self._change_model_button: Optional[QPushButton] = None
        self._warning_banner: Optional[QFrame] = None
        self._error_label: Optional[QLabel] = None

        self._queue_list: Optional[_QueueList] = None
        self._queue_status_label: Optional[QLabel] = None
        self._add_images_button: Optional[QPushButton] = None
        self._grade_all_button: Optional[QPushButton] = None

        self._summary_card: Optional[QFrame] = None
        self._summary_total_label: Optional[QLabel] = None
        self._summary_common_label: Optional[QLabel] = None
        self._summary_confidence_label: Optional[QLabel] = None

        self._preview_label: Optional[_AdaptiveImageLabel] = None
        self._placeholder_label: Optional[QLabel] = None
        self._result_card: Optional[QWidget] = None
        self._grade_label: Optional[QLabel] = None
        self._confidence_label: Optional[QLabel] = None
        self._bars_container: Optional[QVBoxLayout] = None

        self._export_button: Optional[QPushButton] = None
        self._clear_button: Optional[QPushButton] = None
        self._shortcut_bindings: List[tuple] = []
        self._tab_order_applied = False

        self._build_ui()
        self._loading_overlay = LoadingOverlay(self)

        if self.main_window is not None:
            self.main_window.active_model_changed.connect(self._on_active_model_changed)

        self._sync_from_main_window()
        self._refresh_model_banner()
        self._update_action_buttons_enabled()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 24)
        root.setSpacing(16)

        root.addLayout(self._build_header())
        root.addWidget(self._build_model_selector_bar())

        self._warning_banner = self._build_warning_banner()
        root.addWidget(self._warning_banner)

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

    def _apply_tab_order(self) -> None:
        """Chain focus order starting from the top model-selector bar downward."""
        chain = [
            self._change_model_button,
            self._queue_list,
            self._add_images_button,
            self._grade_all_button,
            self._export_button,
            self._clear_button,
        ]
        for earlier, later in zip(chain, chain[1:]):
            if earlier is not None and later is not None:
                QWidget.setTabOrder(earlier, later)

    def _build_header(self) -> QVBoxLayout:
        header = QVBoxLayout()
        header.setSpacing(4)

        title = QLabel("Grade Images")
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 20px; font-weight: 600;")
        header.addWidget(title)

        subtitle = QLabel("Run your trained model to classify bitumen samples.")
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px;")
        header.addWidget(subtitle)

        return header

    def _build_model_selector_bar(self) -> QFrame:
        bar = QFrame()
        bar.setStyleSheet(f"QFrame {{ background-color: {SURFACE_COLOR}; border-radius: 8px; }}")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        label = QLabel("Active Model:")
        label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; background: transparent;")
        layout.addWidget(label)

        self._model_value_label = QLabel("No model selected")
        self._model_value_label.setStyleSheet(
            f"color: {ACCENT_COLOR}; font-size: 13px; font-weight: 600; background: transparent;"
        )
        layout.addWidget(self._model_value_label, 1)

        change_button = QPushButton("Change Model")
        change_button.setCursor(Qt.CursorShape.PointingHandCursor)
        change_button.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {TEXT_PRIMARY}; border: 1px solid {BORDER_COLOR};"
            f"border-radius: 6px; padding: 7px 14px; font-size: 12px; }}"
            f"QPushButton:hover {{ background-color: #2A2E36; }}"
        )
        change_button.setToolTip(shortcut_tooltip("Go to the Model Library to change the active model", "C"))
        change_button.clicked.connect(self._navigate_to_model_library)
        layout.addWidget(change_button)
        self._shortcut_bindings.append((change_button, "C"))
        self._change_model_button = change_button

        return bar

    def _build_warning_banner(self) -> QFrame:
        banner = QFrame()
        banner.setStyleSheet(
            f"QFrame {{ background-color: rgba(232, 168, 56, 30); border: 1px solid {ACCENT_COLOR};"
            f"border-radius: 8px; }}"
        )

        layout = QHBoxLayout(banner)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        link_button = QPushButton("Please load a model first. Go to Model Library \u2192")
        link_button.setObjectName("noModelWarningLink")
        link_button.setCursor(Qt.CursorShape.PointingHandCursor)
        link_button.setStyleSheet(
            f"QPushButton#noModelWarningLink {{ background: transparent; color: {ACCENT_COLOR}; border: none;"
            f"font-size: 12px; font-weight: 600; text-align: left; padding: 0px; }}"
            f"QPushButton#noModelWarningLink:hover {{ color: {ACCENT_HOVER_COLOR}; text-decoration: underline; }}"
        )
        link_button.clicked.connect(self._navigate_to_model_library)
        layout.addWidget(link_button)
        layout.addStretch(1)

        banner.setVisible(False)
        return banner

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        container.setFixedWidth(LEFT_PANEL_WIDTH)
        container.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QLabel("Image Queue")
        header.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 600; background: transparent;")
        layout.addWidget(header)

        self._queue_list = _QueueList()
        self._queue_list.setStyleSheet(
            f"""
            QListWidget {{
                background-color: {SURFACE_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 8px;
            }}
            QListWidget::item {{ border-bottom: 1px solid {BORDER_COLOR}; }}
            QListWidget::item:last {{ border-bottom: none; }}
            QListWidget::item:selected {{ background-color: #2A2E36; }}
            """
        )
        self._queue_list.files_dropped.connect(self._add_images)
        self._queue_list.currentItemChanged.connect(self._on_queue_selection_changed)
        layout.addWidget(self._queue_list, 1)

        self._queue_status_label = QLabel("No images in queue.")
        self._queue_status_label.setWordWrap(True)
        self._queue_status_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;"
        )
        layout.addWidget(self._queue_status_label)

        add_button = QPushButton("Add Images")
        add_button.setCursor(Qt.CursorShape.PointingHandCursor)
        add_button.setStyleSheet(
            f"QPushButton {{ background-color: {SURFACE_COLOR}; color: {TEXT_PRIMARY};"
            f"border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 9px 12px; font-size: 12px; }}"
            f"QPushButton:hover {{ background-color: #2A2E36; }}"
        )
        add_button.setToolTip(shortcut_tooltip("Add images to the grading queue", "A"))
        add_button.clicked.connect(self._on_add_images)
        layout.addWidget(add_button)
        self._shortcut_bindings.append((add_button, "A"))
        self._add_images_button = add_button

        self._grade_all_button = QPushButton("Grade All")
        self._grade_all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._grade_all_button.setFixedHeight(42)
        self._grade_all_button.setStyleSheet(
            f"QPushButton {{ background-color: {ACCENT_COLOR}; color: #13151A; font-weight: 700;"
            f"font-size: 13px; border: none; border-radius: 6px; }}"
            f"QPushButton:hover {{ background-color: {ACCENT_HOVER_COLOR}; }}"
            f"QPushButton:disabled {{ background-color: #4A4230; color: #8B8168; }}"
        )
        self._grade_all_button.setToolTip(shortcut_tooltip("Grade every image in the queue", "R"))
        self._grade_all_button.clicked.connect(self._on_grade_all)
        layout.addWidget(self._grade_all_button)
        self._shortcut_bindings.append((self._grade_all_button, "R"))

        return container

    def _build_right_panel(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        self._summary_card = self._build_summary_card()
        self._summary_card.setVisible(False)
        layout.addWidget(self._summary_card)

        layout.addWidget(self._build_detail_frame(), 1)

        action_row = QHBoxLayout()
        action_row.setSpacing(12)

        self._export_button = QPushButton("Export Results")
        self._export_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_button.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {TEXT_PRIMARY}; border: 1px solid {BORDER_COLOR};"
            f"border-radius: 6px; padding: 9px 16px; font-size: 12px; }}"
            f"QPushButton:hover {{ background-color: #2A2E36; }}"
            f"QPushButton:disabled {{ color: {TEXT_SECONDARY}; border: 1px solid #2A2D34; }}"
        )
        self._export_button.setToolTip(shortcut_tooltip("Export graded results to a CSV file", "E"))
        self._export_button.clicked.connect(self._on_export_results)
        action_row.addWidget(self._export_button)
        self._shortcut_bindings.append((self._export_button, "E"))

        self._clear_button = QPushButton("Clear Results")
        self._clear_button.setObjectName("clearResultsLink")
        self._clear_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_button.setStyleSheet(
            f"QPushButton#clearResultsLink {{ background: transparent; color: {TEXT_SECONDARY}; border: none;"
            f"font-size: 12px; padding: 9px 4px; }}"
            f"QPushButton#clearResultsLink:hover {{ color: {TEXT_PRIMARY}; text-decoration: underline; }}"
            f"QPushButton#clearResultsLink:disabled {{ color: #4A4D55; }}"
        )
        self._clear_button.setToolTip(shortcut_tooltip("Clear all graded results", "S"))
        self._clear_button.clicked.connect(self._on_clear_results)
        action_row.addWidget(self._clear_button)
        self._shortcut_bindings.append((self._clear_button, "S"))

        action_row.addStretch(1)
        layout.addLayout(action_row)

        return container

    def _build_summary_card(self) -> QFrame:
        card = QFrame()
        card.setStyleSheet(f"QFrame {{ background-color: {SURFACE_COLOR}; border-radius: 8px; }}")

        layout = QHBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(28)

        self._summary_total_label = self._build_stat_block(layout, "Images Graded")
        self._summary_common_label = self._build_stat_block(layout, "Most Common Grade")
        self._summary_confidence_label = self._build_stat_block(layout, "Avg Confidence")
        layout.addStretch(1)

        return card

    def _build_stat_block(self, layout: QHBoxLayout, title: str) -> QLabel:
        block = QVBoxLayout()
        block.setSpacing(2)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        block.addWidget(title_label)

        value_label = QLabel("\u2014")
        value_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 18px; font-weight: 700; background: transparent;"
        )
        block.addWidget(value_label)

        layout.addLayout(block)
        return value_label

    def _build_detail_frame(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ background-color: {SURFACE_COLOR}; border-radius: 8px; }}")

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        self._preview_label = _AdaptiveImageLabel()
        self._preview_label.setStyleSheet(f"background-color: {BACKGROUND_COLOR}; border-radius: 8px;")
        layout.addWidget(self._preview_label, 1)

        self._placeholder_label = QLabel("Select an image from the queue to preview it here.")
        self._placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder_label.setWordWrap(True)
        self._placeholder_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px; background: transparent;")
        layout.addWidget(self._placeholder_label)

        self._result_card = QWidget()
        result_layout = QVBoxLayout(self._result_card)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(10)

        self._grade_label = QLabel("")
        self._grade_label.setStyleSheet(
            f"color: {ACCENT_COLOR}; font-size: 28px; font-weight: 700; background: transparent;"
        )
        result_layout.addWidget(self._grade_label)

        self._confidence_label = QLabel("")
        self._confidence_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px; background: transparent;")
        result_layout.addWidget(self._confidence_label)

        self._bars_container = QVBoxLayout()
        self._bars_container.setSpacing(8)
        result_layout.addLayout(self._bars_container)

        self._result_card.setVisible(False)
        layout.addWidget(self._result_card)

        return frame

    # -- Model selector / navigation ----------------------------------------

    def _on_active_model_changed(self, _active_model: Optional[Dict[str, Any]]) -> None:
        self._predictor = None
        self._predictor_model_path = None
        self._refresh_model_banner()

    def _refresh_model_banner(self) -> None:
        active_model = getattr(self.main_window, "active_model", None) if self.main_window else None

        if not active_model:
            self._model_value_label.setText("No model selected")
            self._model_value_label.setStyleSheet(
                f"color: {ACCENT_COLOR}; font-size: 13px; font-weight: 600; background: transparent;"
            )
            return

        metadata = active_model.get("metadata") or {}
        name = metadata.get("name") or Path(active_model.get("path", "")).stem
        date_str = self._format_date(metadata.get("created_at"))
        text = f"{name} \u2014 {date_str}" if date_str else name

        self._model_value_label.setText(text)
        self._model_value_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600; background: transparent;"
        )
        self._warning_banner.setVisible(False)

    @staticmethod
    def _format_date(created_at: Optional[str]) -> str:
        if not created_at:
            return ""
        try:
            parsed = datetime.fromisoformat(created_at)
        except ValueError:
            return ""
        return parsed.strftime("%b %d, %Y")

    def _navigate_to_model_library(self) -> None:
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

    # -- Queue management -----------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: D401 - Qt override
        super().showEvent(event)
        self._sync_from_main_window()
        self._refresh_model_banner()
        bind_page_shortcuts(self._shortcut_bindings)
        if not self._tab_order_applied:
            self._apply_tab_order()
            self._tab_order_applied = True

    def hideEvent(self, event) -> None:  # noqa: D401 - Qt override
        super().hideEvent(event)
        unbind_page_shortcuts(self._shortcut_bindings)

    def _sync_from_main_window(self) -> None:
        """Pull queued paths from ``main_window.grading_images`` and open each from disk.

        ``ImageImportPage`` hands off bare file paths (not decoded images) so
        that sending a large batch to grading never duplicates thousands of
        already-imported images in memory; this page opens each one itself,
        exactly as ``_add_images`` already does for manually-added files.
        """
        if self.main_window is None:
            return
        incoming = getattr(self.main_window, "grading_images", None)
        if not incoming:
            return

        existing_paths = {q.path for q in self._queue}
        added_any = False
        failed_names: List[str] = []
        for entry in incoming:
            path = entry.get("path")
            if not path or path in existing_paths or not path.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            try:
                with Image.open(path) as opened:
                    opened.load()
                    image = opened.convert("RGB")
            except (OSError, ValueError):
                failed_names.append(Path(path).name)
                continue
            self._add_queue_image(path, image)
            existing_paths.add(path)
            added_any = True

        if added_any:
            self._update_queue_status_label()
            self._update_action_buttons_enabled()

        if failed_names:
            self._show_image_load_error(failed_names)

    def _on_add_images(self) -> None:
        file_filter = "Images (*.jpg *.jpeg *.png *.tif)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", file_filter)
        if paths:
            self._add_images(paths)

    def _add_images(self, paths: List[str]) -> None:
        existing_paths = {q.path for q in self._queue}
        added_any = False
        failed_names: List[str] = []
        for path in paths:
            if not path.lower().endswith(SUPPORTED_EXTENSIONS) or path in existing_paths:
                continue
            try:
                with Image.open(path) as opened:
                    opened.load()
                    image = opened.convert("RGB")
            except (OSError, ValueError):
                failed_names.append(Path(path).name)
                continue
            self._add_queue_image(path, image)
            existing_paths.add(path)
            added_any = True

        if added_any:
            self._update_queue_status_label()
            self._update_action_buttons_enabled()

        if failed_names:
            self._show_image_load_error(failed_names)
        elif added_any:
            self._error_label.setVisible(False)

    def _show_image_load_error(self, failed_names: List[str]) -> None:
        names = ", ".join(failed_names[:3])
        if len(failed_names) > 3:
            names += f", and {len(failed_names) - 3} more"
        count_word = "image" if len(failed_names) == 1 else "images"
        self._error_label.setText(f"Could not load {len(failed_names)} {count_word}: {names}")
        self._error_label.setVisible(True)

    def _add_queue_image(self, path: str, image: Image.Image) -> None:
        widget = _QueueItemWidget(Path(path).name, image)
        list_item = QListWidgetItem()
        list_item.setSizeHint(widget.sizeHint())

        self._queue_list.addItem(list_item)
        self._queue_list.setItemWidget(list_item, widget)

        queue_image = _QueueImage(id=next(self._id_counter), path=path, image=image, item=list_item, widget=widget)
        widget.remove_requested.connect(lambda q=queue_image: self._remove_queue_image(q))
        self._queue.append(queue_image)

    def _remove_queue_image(self, queue_image: _QueueImage) -> None:
        if queue_image not in self._queue:
            return
        index = self._queue.index(queue_image)
        self._queue.pop(index)

        row = self._queue_list.row(queue_image.item)
        item_widget = self._queue_list.itemWidget(queue_image.item)
        self._queue_list.takeItem(row)
        if item_widget is not None:
            item_widget.deleteLater()

        if self._selected_id == queue_image.id:
            self._selected_id = None
            if self._queue:
                self._select_queue_item(self._queue[0].id)
            else:
                self._update_detail_view()

        self._update_queue_status_label()
        self._update_action_buttons_enabled()
        self._update_summary_card()

    def _update_queue_status_label(self) -> None:
        count = len(self._queue)
        if count == 0:
            self._queue_status_label.setText("No images in queue.")
            return
        graded = sum(1 for q in self._queue if q.result is not None)
        self._queue_status_label.setText(f"{count} image(s) in queue \u2014 {graded} graded.")

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
        self._update_detail_view()

    # -- Grading ---------------------------------------------------------------

    def _on_grade_all(self) -> None:
        if not self._queue or self._thread is not None:
            return

        self._error_label.setVisible(False)

        active_model = getattr(self.main_window, "active_model", None) if self.main_window else None
        if not active_model:
            self._warning_banner.setVisible(True)
            return
        self._warning_banner.setVisible(False)

        model_path = active_model.get("path")
        metadata = active_model.get("metadata") or {}
        grade_labels = metadata.get("grade_labels")
        self._pending_model_path = model_path
        images = [(queue_image.id, queue_image.image) for queue_image in self._queue]

        self._loading_overlay.show_message("Loading model\u2026")
        self._set_grading_active(True)

        self._thread = QThread(self)
        self._worker = _GradingWorker(self._predictor, self._predictor_model_path, model_path, grade_labels, images)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_grading_finished)
        self._worker.failed.connect(self._on_grading_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._on_grading_thread_finished)

        self._thread.start()

    def _set_grading_active(self, active: bool) -> None:
        """Disable controls that would race with an in-flight grading job."""
        for button in (self._grade_all_button, self._add_images_button, self._change_model_button):
            if button is not None:
                button.setEnabled(not active)
        if active:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _on_grading_finished(self, predictor: ModelPredictor, results: List[Any], failures: int) -> None:
        self._predictor = predictor
        self._predictor_model_path = self._pending_model_path

        results_by_id = dict(results)
        for queue_image in self._queue:
            result = results_by_id.get(queue_image.id)
            if result is not None:
                queue_image.result = result
                queue_image.widget.show_result(result["grade"], result["confidence"])

        self._loading_overlay.hide_overlay()
        self._set_grading_active(False)

        if failures:
            self._error_label.setText(f"Failed to grade {failures} image(s); see remaining results below.")
            self._error_label.setVisible(True)

        self._update_queue_status_label()
        self._update_summary_card()
        self._update_action_buttons_enabled()

        if self._selected_id is None and self._queue:
            self._select_queue_item(self._queue[0].id)
        else:
            self._update_detail_view()

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
        """On a model load failure: clear the active model and surface a dialog.

        A corrupt/missing checkpoint means the previously "active" model is
        no longer usable, so we clear the app-wide state (which also updates
        the sidebar status pill and window title) rather than leaving stale
        state that would just fail again on the next attempt.
        """
        self._predictor = None
        self._predictor_model_path = None

        if self.main_window is not None:
            self.main_window.set_active_model(None)

        message = str(exc) or exc.__class__.__name__
        self._error_label.setText(f"Failed to load model: {message}")
        self._error_label.setVisible(True)
        self._refresh_model_banner()
        self._warning_banner.setVisible(True)

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Model Load Failed")
        box.setText("The selected model could not be loaded and has been cleared.")
        box.setInformativeText(message)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def _update_summary_card(self) -> None:
        graded_results = [q.result for q in self._queue if q.result is not None]
        if not graded_results:
            self._summary_card.setVisible(False)
            return

        total = len(graded_results)
        grade_counts = Counter(result["grade"] for result in graded_results)
        most_common_grade, _count = grade_counts.most_common(1)[0]
        avg_confidence = sum(result["confidence"] for result in graded_results) / total

        self._summary_total_label.setText(str(total))
        self._summary_common_label.setText(most_common_grade)
        self._summary_confidence_label.setText(f"{avg_confidence * 100:.1f}%")
        self._summary_card.setVisible(True)

    # -- Detail (single-image result) view --------------------------------------

    def _update_detail_view(self) -> None:
        queue_image = self._find_queue_item(self._selected_id)

        if queue_image is None:
            self._preview_label.set_source_image(None)
            self._placeholder_label.setText("Select an image from the queue to preview it here.")
            self._placeholder_label.setVisible(True)
            self._result_card.setVisible(False)
            return

        self._preview_label.set_source_image(queue_image.image)

        if queue_image.result is None:
            self._placeholder_label.setText('Click "Grade All" to see this image\u2019s predicted grade.')
            self._placeholder_label.setVisible(True)
            self._result_card.setVisible(False)
            return

        self._placeholder_label.setVisible(False)
        self._show_result_card(queue_image.result)

    def _show_result_card(self, result: Dict[str, Any]) -> None:
        self._grade_label.setText(result["grade"])
        self._confidence_label.setText(f"Confidence: {result['confidence'] * 100:.1f}%")

        while self._bars_container.count():
            taken = self._bars_container.takeAt(0)
            widget = taken.widget()
            if widget is not None:
                widget.deleteLater()

        for label, probability in result["all_probabilities"].items():
            bar = _ConfidenceBar(label, probability, is_winner=(label == result["grade"]))
            self._bars_container.addWidget(bar)

        self._result_card.setVisible(True)

    # -- Export / clear ----------------------------------------------------------

    def _on_export_results(self) -> None:
        if self._export_thread is not None:
            return

        graded = [q for q in self._queue if q.result is not None]
        if not graded:
            return

        default_name = f"grading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        save_path, _ = QFileDialog.getSaveFileName(self, "Export Results", default_name, "CSV Files (*.csv)")
        if not save_path:
            return

        grade_order = list(graded[0].result["all_probabilities"].keys())
        header = ["filename", "grade", "confidence"] + [f"probability_{label}" for label in grade_order]

        rows: List[List[str]] = []
        for queue_image in graded:
            probabilities = queue_image.result["all_probabilities"]
            row = [
                Path(queue_image.path).name,
                queue_image.result["grade"],
                f"{queue_image.result['confidence']:.4f}",
            ]
            row += [f"{probabilities.get(label, 0.0):.4f}" for label in grade_order]
            rows.append(row)

        self._error_label.setVisible(False)
        self._loading_overlay.show_message("Exporting results\u2026")
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
        self._error_label.setText(f"Failed to export results: {message}")
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
            queue_image.widget.clear_result()

        self._summary_card.setVisible(False)
        self._update_queue_status_label()
        self._update_action_buttons_enabled()
        self._update_detail_view()

    def _update_action_buttons_enabled(self) -> None:
        has_images = bool(self._queue)
        has_results = any(q.result is not None for q in self._queue)

        self._grade_all_button.setEnabled(has_images)
        self._export_button.setEnabled(has_results)
        self._clear_button.setEnabled(has_results)
