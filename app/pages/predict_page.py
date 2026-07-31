"""
Prediction / Grading page (regression).

Provides the UI for running inference with a loaded ``BitumenRegressor``
against new bitumen sample images and displaying the predicted Water,
Solids, and Bitumen content -- compared against the training data's
average/range, a compositional ("sum to ~100%") sanity check, and a
data-derived "closest Pan grade" indicator -- either for a single selected
image or as a batch after "Grade All".
"""
from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from PIL import Image
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFontMetrics,
    QPainter,
    QResizeEvent,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
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
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.components.image_editor import pil_to_qpixmap
from app.components.loading_overlay import LoadingOverlay
from app.ml.predictor import RegressionPredictor
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
SUCCESS_COLOR = "#3CB878"

LEFT_PANEL_WIDTH = 340
THUMB_SIZE = 64
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif")

OUTPUT_LABELS = ("Water", "Solids", "Bitumen")

#: The four Pan grades this app recognises, in increasing order, with their
#: fixed indicator colours (border/pill colour on the grade card and table).
PAN_GRADES = (3, 4, 5, 6)
PAN_GRADE_COLORS = {3: "#5B9BD5", 4: "#3CB878", 5: "#E8A838", 6: "#E5484D"}
PAN_GRADE_TEXT_COLORS = {3: "#FFFFFF", 4: "#FFFFFF", 5: "#13151A", 6: "#FFFFFF"}

#: A saved model's metadata only stores each output's training mean/std
#: (see app.ml.dataset.RegressionDataset.get_output_stats), not its literal
#: min/max -- so both the range bars and the "closest Pan grade" heuristic
#: below approximate the training range as mean +/- this many std devs,
#: clamped to the valid 0-100% span.
RANGE_STD_MULTIPLIER = 2.0


def _approx_output_range(mean: float, std: float) -> Tuple[float, float]:
    """Approximate a training output's min/max as mean +/- RANGE_STD_MULTIPLIER*std."""
    low = max(0.0, mean - RANGE_STD_MULTIPLIER * std)
    high = min(100.0, mean + RANGE_STD_MULTIPLIER * std)
    if high <= low:
        high = low + 1e-6
    return low, high


def _closest_pan_grade(bitumen_value: float, bitumen_mean: float, bitumen_std: float) -> int:
    """Bucket a predicted Bitumen value into one of the four Pan grades.

    There is no persisted mapping from Bitumen content to Pan grade in a
    saved model's metadata, so this splits the model's own approximate
    Bitumen training range into four equal bins (one per grade, low to
    high) and picks whichever bin the prediction falls into.
    """
    low, high = _approx_output_range(bitumen_mean, bitumen_std)
    span = high - low
    fraction = 0.5 if span <= 0 else (bitumen_value - low) / span
    fraction = max(0.0, min(1.0, fraction))
    index = min(int(fraction * len(PAN_GRADES)), len(PAN_GRADES) - 1)
    return PAN_GRADES[index]


class _GradingWorker(QObject):
    """Runs a already-loaded ``RegressionPredictor`` over one or more images on a background QThread.

    ``MainWindow.set_active_model`` loads the checkpoint into a
    ``RegressionPredictor`` up front (see its docstring), so this worker's
    only job is to keep the (potentially slow) inference loop itself off
    the UI thread; the main thread only touches widget state once results
    come back via ``finished``. Used for both "Grade This Image" (a
    single-item list) and "Grade All" (the full queue).
    """

    finished = pyqtSignal(list, int)  # [(image_id, result_or_None)], failure_count
    failed = pyqtSignal(str)

    def __init__(self, predictor: RegressionPredictor, images: List[Any]):
        super().__init__()
        self._predictor = predictor
        self._images = images

    def run(self) -> None:
        if self._predictor is None:
            self.failed.emit("No active model is loaded.")
            return

        results: List[Any] = []
        failures = 0
        for image_id, pil_image in self._images:
            try:
                result = self._predictor.predict(pil_image)
            except Exception:  # noqa: BLE001 - keep grading remaining images
                failures += 1
                results.append((image_id, None))
                continue
            results.append((image_id, result))

        self.finished.emit(results, failures)


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
    #: Snapshot of the active model's output_stats *at grading time*, so a
    #: later model change doesn't retroactively alter an already-graded
    #: image's training-average/range comparisons without re-grading it.
    output_stats: Optional[Dict[str, Dict[str, float]]] = None


class _QueueItemWidget(QWidget):
    """Row widget for one queued image: a 64x64 thumbnail plus its filename."""

    def __init__(self, filename: str, pil_image: Image.Image, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        self._thumb_label = QLabel()
        self._thumb_label.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        self._thumb_label.setStyleSheet(f"background-color: {BACKGROUND_COLOR}; border-radius: 4px;")
        self._thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._set_thumbnail(pil_image)
        layout.addWidget(self._thumb_label)

        name_label = QLabel()
        metrics = QFontMetrics(name_label.font())
        name_label.setText(metrics.elidedText(filename, Qt.TextElideMode.ElideMiddle, 200))
        name_label.setToolTip(filename)
        name_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; background: transparent;")
        layout.addWidget(name_label, 1)

    def _set_thumbnail(self, pil_image: Image.Image) -> None:
        pixmap = pil_to_qpixmap(pil_image).scaled(
            THUMB_SIZE,
            THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumb_label.setPixmap(pixmap)


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


class _ImageDropZone(QFrame):
    """Dashed drag-and-drop target for grading images (no embedded button -- see "Add Images")."""

    files_selected = pyqtSignal(list)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("gradeDropZone")
        self.setAcceptDrops(True)
        self.setFixedHeight(84)
        self._build_ui()
        self._apply_style(active=False)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("Drag & drop images here")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 600; background: transparent;")
        layout.addWidget(title)

        subtitle = QLabel("Supports JPG, PNG, and TIF")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; background: transparent;")
        layout.addWidget(subtitle)

    def _apply_style(self, active: bool) -> None:
        border_color = ACCENT_HOVER_COLOR if active else ACCENT_COLOR
        background = "#2A2E36" if active else SURFACE_COLOR
        self.setStyleSheet(
            f"QFrame#gradeDropZone {{ background-color: {background}; border: 2px dashed {border_color};"
            f"border-radius: 8px; }}"
        )

    @staticmethod
    def _is_supported(path: str) -> bool:
        return path.lower().endswith(SUPPORTED_EXTENSIONS)

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
            self.files_selected.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class _ClickableBanner(QFrame):
    """A QFrame that emits ``clicked`` on left-click (used for the full-width no-model banner)."""

    clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event) -> None:  # noqa: D401 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class _AdaptiveImageLabel(QLabel):
    """QLabel that keeps a source pixmap scaled to fill its current size."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._source_pixmap = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(160)

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


class _RangeBar(QWidget):
    """One output's training-range bar: track spans min..max, amber fill to the predicted value."""

    def __init__(self, label: str, low: float, high: float, value: float, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._label = label
        self._low = low
        self._high = high
        self._value = value
        self.setFixedHeight(58)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event) -> None:  # noqa: D401 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        name_height = 16
        track_height = 8
        bottom_label_height = 16
        track_top = name_height + 12
        track_left = 4.0
        track_right = self.width() - 4.0
        track_width = track_right - track_left

        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(TEXT_PRIMARY))
        painter.drawText(
            QRectF(0, 0, self.width(), name_height),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._label,
        )

        span = self._high - self._low
        fraction = 0.0 if span <= 0 else max(0.0, min(1.0, (self._value - self._low) / span))
        marker_x = track_left + fraction * track_width

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(BACKGROUND_COLOR))
        painter.drawRoundedRect(QRectF(track_left, track_top, track_width, track_height), 4, 4)

        fill_width = marker_x - track_left
        if fill_width > 0:
            painter.setBrush(QColor(ACCENT_COLOR))
            painter.drawRoundedRect(QRectF(track_left, track_top, fill_width, track_height), 4, 4)

        marker_y = track_top + track_height / 2
        painter.setBrush(QColor(TEXT_PRIMARY))
        painter.drawEllipse(QPointF(marker_x, marker_y), 5, 5)

        labels_top = track_top + track_height + 4
        value_text = f"{self._value:.2f}%"
        font.setBold(True)
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(QColor(TEXT_PRIMARY))
        value_width = 56
        value_x = min(max(marker_x - value_width / 2, track_left), track_right - value_width)
        painter.drawText(
            QRectF(value_x, labels_top, value_width, bottom_label_height),
            Qt.AlignmentFlag.AlignHCenter,
            value_text,
        )

        font.setBold(False)
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QColor(TEXT_SECONDARY))
        painter.drawText(
            QRectF(track_left, labels_top, 60, bottom_label_height),
            Qt.AlignmentFlag.AlignLeft,
            f"{self._low:.2f}%",
        )
        painter.drawText(
            QRectF(track_right - 60, labels_top, 60, bottom_label_height),
            Qt.AlignmentFlag.AlignRight,
            f"{self._high:.2f}%",
        )

        painter.end()


class _PanGradeCard(QFrame):
    """"Closest Pan grade: N" indicator, border/accent colour keyed to the grade."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._apply_frame_style(TEXT_SECONDARY)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        self._title_label = QLabel("")
        self._title_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 700; background: transparent;")
        layout.addWidget(self._title_label)

        secondary_label = QLabel("Based on Bitumen content relative to training data averages.")
        secondary_label.setWordWrap(True)
        secondary_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        layout.addWidget(secondary_label)

    def _apply_frame_style(self, color: str) -> None:
        self.setStyleSheet(
            f"QFrame {{ background-color: {BACKGROUND_COLOR}; border-radius: 6px; border-left: 4px solid {color}; }}"
        )

    def set_grade(self, grade: int) -> None:
        self._title_label.setText(f"Closest Pan grade: {grade}")
        self._apply_frame_style(PAN_GRADE_COLORS.get(grade, TEXT_SECONDARY))


class PredictPage(QWidget):
    """Page for grading bitumen sample images with the active regression model.

    Images populate automatically from ``main_window.grading_images`` (sent
    over from ``ImageImportPage``'s "Send to Grading" action), can be
    supplemented via "Add Images", or dropped directly onto the drop zone or
    the queue list. A single selected image can be graded on its own
    ("Grade This Image"), or the whole queue at once ("Grade All"); either
    action runs the active model's ``RegressionPredictor`` on a background
    QThread. Results can then be inspected per-image (comparison table,
    per-output range bars, Pan grade indicator) or, after "Grade All", as a
    batch summary + table exportable to CSV.
    """

    def __init__(self, main_window: Optional["MainWindow"] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = main_window

        self._queue: List[_QueueImage] = []
        self._id_counter = itertools.count(1)
        self._selected_id: Optional[int] = None
        #: "single" shows the not-graded/single-result page for the current
        #: selection; "batch" shows the batch summary+table (set by "Grade All").
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

        self._shortcut_bindings: List[tuple] = []
        self._tab_order_applied = False

        self._build_ui()
        self._loading_overlay = LoadingOverlay(self)

        if self.main_window is not None:
            self.main_window.active_model_changed.connect(self._on_active_model_changed)

        self._sync_from_main_window()
        self._refresh_top_bar()
        self._refresh_right_column()
        self._update_action_buttons_enabled()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 24)
        root.setSpacing(16)

        root.addLayout(self._build_header())
        root.addWidget(self._build_model_selector_bar())

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
            self._clear_all_button,
            self._grade_this_button,
            self._export_button,
            self._clear_results_button,
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

        subtitle = QLabel("Predict Water, Solids, and Bitumen content from your bitumen sample images.")
        subtitle.setWordWrap(True)
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

        self._model_value_label = QLabel("")
        self._model_value_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600; background: transparent;"
        )
        layout.addWidget(self._model_value_label, 1)

        self._change_model_button = QPushButton("Change Model")
        self._change_model_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._change_model_button.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {TEXT_PRIMARY}; border: 1px solid {BORDER_COLOR};"
            f"border-radius: 6px; padding: 7px 14px; font-size: 12px; }}"
            f"QPushButton:hover {{ background-color: #2A2E36; }}"
        )
        self._change_model_button.setToolTip(shortcut_tooltip("Go to the Model Library to change the active model", "C"))
        self._change_model_button.clicked.connect(self._navigate_to_model_library)
        layout.addWidget(self._change_model_button)
        self._shortcut_bindings.append((self._change_model_button, "C"))

        return bar

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

        self._drop_zone = _ImageDropZone()
        self._drop_zone.files_selected.connect(self._add_images)
        layout.addWidget(self._drop_zone)

        self._add_images_button = QPushButton("Add Images")
        self._add_images_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._add_images_button.setStyleSheet(
            f"QPushButton {{ background-color: {SURFACE_COLOR}; color: {TEXT_PRIMARY};"
            f"border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 9px 12px; font-size: 12px; }}"
            f"QPushButton:hover {{ background-color: #2A2E36; }}"
        )
        self._add_images_button.setToolTip(shortcut_tooltip("Add images to the grading queue", "A"))
        self._add_images_button.clicked.connect(self._on_add_images)
        layout.addWidget(self._add_images_button)
        self._shortcut_bindings.append((self._add_images_button, "A"))

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

        self._queue_status_label = QLabel("0 image(s) loaded")
        self._queue_status_label.setWordWrap(True)
        self._queue_status_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;"
        )
        layout.addWidget(self._queue_status_label)

        self._grade_all_button = QPushButton("Grade All")
        self._grade_all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._grade_all_button.setFixedHeight(46)
        self._grade_all_button.setStyleSheet(
            f"QPushButton {{ background-color: {ACCENT_COLOR}; color: #13151A; font-weight: 700;"
            f"font-size: 14px; border: none; border-radius: 6px; }}"
            f"QPushButton:hover {{ background-color: {ACCENT_HOVER_COLOR}; }}"
            f"QPushButton:disabled {{ background-color: #4A4230; color: #8B8168; }}"
        )
        self._grade_all_button.setToolTip(shortcut_tooltip("Grade every image in the queue", "R"))
        self._grade_all_button.clicked.connect(self._on_grade_all)
        layout.addWidget(self._grade_all_button)
        self._shortcut_bindings.append((self._grade_all_button, "R"))

        self._clear_all_button = QPushButton("Clear All")
        self._clear_all_button.setObjectName("clearAllLink")
        self._clear_all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_all_button.setStyleSheet(
            f"QPushButton#clearAllLink {{ background: transparent; color: {TEXT_SECONDARY}; border: none;"
            f"font-size: 12px; padding: 4px; }}"
            f"QPushButton#clearAllLink:hover {{ color: {TEXT_PRIMARY}; text-decoration: underline; }}"
            f"QPushButton#clearAllLink:disabled {{ color: #4A4D55; }}"
        )
        self._clear_all_button.setToolTip(shortcut_tooltip("Remove every image from the queue", "K"))
        self._clear_all_button.clicked.connect(self._on_clear_all)
        layout.addWidget(self._clear_all_button, 0, Qt.AlignmentFlag.AlignLeft)
        self._shortcut_bindings.append((self._clear_all_button, "K"))

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
        banner.setStyleSheet(
            f"QFrame {{ background-color: rgba(232, 168, 56, 30); border: 1px solid {ACCENT_COLOR};"
            f"border-radius: 8px; }}"
        )
        banner.clicked.connect(self._navigate_to_model_library)

        banner_layout = QVBoxLayout(banner)
        banner_layout.setContentsMargins(24, 20, 24, 20)

        message = QLabel(
            "No model loaded. Load a model from the Model Library before grading images. \u2192"
        )
        message.setWordWrap(True)
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message.setStyleSheet(f"color: {ACCENT_COLOR}; font-size: 14px; font-weight: 600; background: transparent;")
        banner_layout.addWidget(message)

        layout.addWidget(banner)
        return page

    def _build_not_graded_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ background-color: {SURFACE_COLOR}; border-radius: 8px; }}")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(18, 18, 18, 18)
        frame_layout.setSpacing(14)

        self._not_graded_preview = _AdaptiveImageLabel()
        self._not_graded_preview.setStyleSheet(f"background-color: {BACKGROUND_COLOR}; border-radius: 8px;")
        frame_layout.addWidget(self._not_graded_preview, 1)

        self._not_graded_placeholder = QLabel("Select an image from the queue to preview it here.")
        self._not_graded_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._not_graded_placeholder.setWordWrap(True)
        self._not_graded_placeholder.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px; background: transparent;")
        frame_layout.addWidget(self._not_graded_placeholder)

        self._grade_this_button = QPushButton("Grade This Image")
        self._grade_this_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._grade_this_button.setFixedHeight(42)
        self._grade_this_button.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {ACCENT_COLOR}; font-weight: 700; font-size: 13px;"
            f"border: 1px solid {ACCENT_COLOR}; border-radius: 6px; padding: 0 20px; }}"
            f"QPushButton:hover {{ background-color: rgba(232, 168, 56, 25); }}"
            f"QPushButton:disabled {{ color: {TEXT_SECONDARY}; border: 1px solid {BORDER_COLOR}; }}"
        )
        self._grade_this_button.setToolTip(shortcut_tooltip("Grade the selected image", "D"))
        self._grade_this_button.clicked.connect(self._on_grade_this_image)
        self._grade_this_button.setEnabled(False)
        frame_layout.addWidget(self._grade_this_button, 0, Qt.AlignmentFlag.AlignHCenter)
        self._shortcut_bindings.append((self._grade_this_button, "D"))

        layout.addWidget(frame, 1)
        return page

    def _build_single_result_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        preview_frame = QFrame()
        preview_frame.setStyleSheet(f"QFrame {{ background-color: {SURFACE_COLOR}; border-radius: 8px; }}")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(14, 14, 14, 14)
        self._single_preview = _AdaptiveImageLabel()
        self._single_preview.setStyleSheet(f"background-color: {BACKGROUND_COLOR}; border-radius: 8px;")
        preview_layout.addWidget(self._single_preview)
        layout.addWidget(preview_frame, 1)

        results_frame = QFrame()
        results_frame.setStyleSheet(f"QFrame {{ background-color: {SURFACE_COLOR}; border-radius: 8px; }}")
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

        self._clear_results_button = QPushButton("Clear Results")
        self._clear_results_button.setObjectName("clearResultsLink")
        self._clear_results_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_results_button.setStyleSheet(
            f"QPushButton#clearResultsLink {{ background: transparent; color: {TEXT_SECONDARY}; border: none;"
            f"font-size: 12px; padding: 9px 4px; }}"
            f"QPushButton#clearResultsLink:hover {{ color: {TEXT_PRIMARY}; text-decoration: underline; }}"
            f"QPushButton#clearResultsLink:disabled {{ color: #4A4D55; }}"
        )
        self._clear_results_button.setToolTip(shortcut_tooltip("Clear all graded results", "X"))
        self._clear_results_button.clicked.connect(self._on_clear_results)
        action_row.addWidget(self._clear_results_button)
        self._shortcut_bindings.append((self._clear_results_button, "X"))

        action_row.addStretch(1)
        layout.addLayout(action_row)

        return page

    def _build_batch_summary_card(self) -> None:
        card = QFrame()
        card.setStyleSheet(f"QFrame {{ background-color: {SURFACE_COLOR}; border-radius: 8px; }}")

        layout = QHBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(24)

        for title in ("Images graded", "Mean Water", "Mean Solids", "Mean Bitumen", "Mean sum deviation", "Sum warnings (>5%)"):
            self._batch_summary_labels[title] = self._build_stat_block(layout, title)
        layout.addStretch(1)

        self._batch_summary_card = card

    def _build_stat_block(self, layout: QHBoxLayout, title: str) -> QLabel:
        block = QVBoxLayout()
        block.setSpacing(2)

        title_label = QLabel(title)
        title_label.setWordWrap(True)
        title_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        block.addWidget(title_label)

        value_label = QLabel("\u2014")
        value_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 18px; font-weight: 700; background: transparent;"
        )
        block.addWidget(value_label)

        layout.addLayout(block)
        return value_label

    def _build_batch_table(self) -> QTableWidget:
        table = QTableWidget(0, 8)
        table.setHorizontalHeaderLabels(["#", "Filename", "Water", "Solids", "Bitumen", "Sum", "Sum OK", "Pan Grade"])
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
            QTableWidget::item:selected {{ background-color: rgba(232, 168, 56, 45); color: {TEXT_PRIMARY}; }}
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
            self._model_value_label.setText("No model loaded \u2014 go to Model Library first.")
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
        self._refresh_top_bar()
        self._refresh_right_column()
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
            self._error_label.setText("No active model is loaded.")
            self._error_label.setVisible(True)
            return

        self._pending_output_stats = metadata.get("output_stats") or {}
        self._pending_grade_all = grade_all
        payload = [(queue_image.id, queue_image.image) for queue_image in images_to_grade]

        self._loading_overlay.show_message("Grading all images\u2026" if grade_all else "Grading image\u2026")
        self._set_grading_active(True)

        self._thread = QThread(self)
        self._worker = _GradingWorker(predictor, payload)
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
        for button in (
            self._grade_all_button,
            self._add_images_button,
            self._change_model_button,
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
            self._error_label.setText(f"Failed to grade {failures} {count_word}; see remaining results below.")
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
        """On an unexpected grading failure: clear the active model and surface a dialog.

        ``MainWindow.set_active_model`` already validates a checkpoint loads
        correctly before it becomes active, so this is a defensive fallback
        for the unlikely case a previously-loaded model becomes unusable
        mid-session (e.g. its file is deleted from disk). Clearing the
        app-wide state (which also updates the sidebar status pill and
        window title) avoids leaving stale state that would just fail again.
        """
        if self.main_window is not None:
            self.main_window.set_active_model(None)

        message = str(exc) or exc.__class__.__name__
        self._error_label.setText(f"Failed to load model: {message}")
        self._error_label.setVisible(True)
        self._refresh_top_bar()
        self._refresh_right_column()

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Model Load Failed")
        box.setText("The selected model could not be loaded and has been cleared.")
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
            self._not_graded_placeholder.setText("Select an image from the queue to preview it here.")
            self._not_graded_placeholder.setVisible(True)
            self._grade_this_button.setEnabled(False)
            self._right_stack.setCurrentWidget(self._not_graded_page)
            return

        if queue_image.result is None:
            self._not_graded_preview.set_source_image(queue_image.image)
            self._not_graded_placeholder.setVisible(False)
            self._grade_this_button.setEnabled(True)
            self._right_stack.setCurrentWidget(self._not_graded_page)
            return

        self._single_preview.set_source_image(queue_image.image)
        self._update_measurement_table(queue_image.result, queue_image.output_stats or {})
        self._update_range_bars(queue_image.result, queue_image.output_stats or {})
        self._update_pan_grade_card(queue_image.result, queue_image.output_stats or {})
        self._right_stack.setCurrentWidget(self._single_result_page)

    def _update_measurement_table(self, result: Dict[str, Any], output_stats: Dict[str, Dict[str, float]]) -> None:
        training_avgs = {label: output_stats.get(label, {}).get("mean", 0.0) for label in OUTPUT_LABELS}
        training_sum = sum(training_avgs.values())

        for row, label in enumerate(OUTPUT_LABELS):
            predicted_value = result[label]["value"]
            pred_item = QTableWidgetItem(f"{predicted_value:.2f}")
            avg_item = QTableWidgetItem(f"{training_avgs[label]:.2f}")
            for item in (pred_item, avg_item):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._measurement_table.setItem(row, 1, pred_item)
            self._measurement_table.setItem(row, 2, avg_item)

        sum_deviation = result["sum_deviation"]
        if sum_deviation < 2.0:
            color = SUCCESS_COLOR
        elif sum_deviation <= 5.0:
            color = ACCENT_COLOR
        else:
            color = DANGER_COLOR

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

        for label in OUTPUT_LABELS:
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
        self._batch_summary_labels["Mean Water"].setText(f"{mean_water:.2f}%")
        self._batch_summary_labels["Mean Solids"].setText(f"{mean_solids:.2f}%")
        self._batch_summary_labels["Mean Bitumen"].setText(f"{mean_bitumen:.2f}%")
        self._batch_summary_labels["Mean sum deviation"].setText(f"{mean_sum_deviation:.2f}%")
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

        header = ["filename", "water", "solids", "bitumen", "sum", "sum_deviation", "sum_ok", "pan_grade"]
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
