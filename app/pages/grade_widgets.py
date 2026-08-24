"""Grade-page widgets: queue, drop zone, range bars, and workers."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QDragEnterEvent, QDropEvent, QFontMetrics, QImage, QPainter, QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.components.drop_zone import drop_has_accepted_files, dropped_local_paths
from app.constants import IMAGE_EXTENSIONS
from app.ml.predictor import RegressionPredictor
from app.theme import (
    ACCENT_COLOR,
    BACKGROUND_COLOR,
    DANGER_COLOR,
    SUCCESS_COLOR,
    TEXT_INVERSE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WATER_LINE_COLOR,
    drop_zone_qss,
    ghost_button_qss,
    link_button_qss,
)
from app.utils.files import pick_image_files, pick_image_folder
from app.utils.media import collect_images


def pil_to_qpixmap(image: Image.Image) -> QPixmap:
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimage = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimage.copy())

THUMB_SIZE = 64
PAN_GRADES = (3, 4, 5, 6)
PAN_GRADE_COLORS = {3: WATER_LINE_COLOR, 4: SUCCESS_COLOR, 5: ACCENT_COLOR, 6: DANGER_COLOR}
PAN_GRADE_TEXT_COLORS = {3: "#FFFFFF", 4: "#FFFFFF", 5: TEXT_INVERSE, 6: "#FFFFFF"}
RANGE_STD_MULTIPLIER = 2.0

def _approx_output_range(mean: float, std: float) -> Tuple[float, float]:
    """Rough training min/max: mean +/- RANGE_STD_MULTIPLIER*std."""
    low = max(0.0, mean - RANGE_STD_MULTIPLIER * std)
    high = min(100.0, mean + RANGE_STD_MULTIPLIER * std)
    if high <= low:
        high = low + 1e-6
    return low, high


def _closest_pan_grade(bitumen_value: float, bitumen_mean: float, bitumen_std: float) -> int:
    """Map predicted Bitumen into one of the four Pan grades.

    There's no saved Bitumen→Pan mapping, so we split the model's Bitumen
    training range into four equal bins and pick which one the prediction lands in.
    """
    low, high = _approx_output_range(bitumen_mean, bitumen_std)
    span = high - low
    fraction = 0.5 if span <= 0 else (bitumen_value - low) / span
    fraction = max(0.0, min(1.0, fraction))
    index = min(int(fraction * len(PAN_GRADES)), len(PAN_GRADES) - 1)
    return PAN_GRADES[index]


class _GradingWorker(QObject):
    """Run a loaded ``RegressionPredictor`` on one or more images off the UI thread.

    The checkpoint is already loaded by ``MainWindow.set_active_model``; this
    just keeps inference off the main thread. Used for both "Grade This Image"
    and "Grade All".
    """

    finished = pyqtSignal(list, int)  # [(image_id, result_or_None)], failure_count
    failed = pyqtSignal(str)

    def __init__(self, predictor: RegressionPredictor, images: List[Any]):
        super().__init__()
        self._predictor = predictor
        self._images = images

    def run(self) -> None:
        if self._predictor is None:
            self.failed.emit("No model loaded.")
            return

        results: List[Any] = []
        failures = 0
        for image_id, source in self._images:
            try:
                result = self._predictor.predict(source)
            except Exception:  # noqa: BLE001 - keep grading remaining images
                failures += 1
                results.append((image_id, None))
                continue
            results.append((image_id, result))

        self.finished.emit(results, failures)


class _ExportWorker(QObject):
    """Write graded results to CSV on a background thread."""

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
    """One queued photo (path on disk) plus its list-row widgets."""

    id: int
    path: str
    item: QListWidgetItem
    widget: "_QueueItemWidget"
    result: Optional[Dict[str, Any]] = None
    output_stats: Optional[Dict[str, Dict[str, float]]] = None


class _QueueItemWidget(QWidget):
    """Queue row: filename, elided to the current row width."""

    def __init__(self, filename: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._filename = filename

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        self._name_label = QLabel()
        self._name_label.setToolTip(filename)
        self._name_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._name_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; background: transparent;")
        layout.addWidget(self._name_label, 1)
        self._elide_name()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._elide_name()

    def _elide_name(self) -> None:
        metrics = QFontMetrics(self._name_label.font())
        width = max(40, self._name_label.width())
        self._name_label.setText(metrics.elidedText(self._filename, Qt.TextElideMode.ElideMiddle, width))


class _QueueList(QListWidget):
    """Queue list that also accepts dropped image files."""

    files_dropped = pyqtSignal(list)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if drop_has_accepted_files(event, IMAGE_EXTENSIONS, recurse_dirs=True):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = dropped_local_paths(event, IMAGE_EXTENSIONS, recurse_dirs=True)
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class _ImageDropZone(QFrame):
    """Drop photos or a folder here, or choose files/folder."""

    files_selected = pyqtSignal(list)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("gradeDropZone")
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._build_ui()
        self._apply_style(active=False)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("Drop photos or a folder here")
        title.setWordWrap(True)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; background: transparent;")
        layout.addWidget(title)

        subtitle = QLabel("JPG, PNG, or TIF")
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        layout.addWidget(subtitle)

        button_row = QHBoxLayout()
        button_row.setSpacing(12)
        button_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.browse_button = QPushButton("Choose files")
        self.browse_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_button.setStyleSheet(ghost_button_qss())
        self.browse_button.clicked.connect(self._browse_files)
        button_row.addWidget(self.browse_button)

        self.browse_folder_button = QPushButton("Open folder")
        self.browse_folder_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_folder_button.setStyleSheet(link_button_qss())
        self.browse_folder_button.clicked.connect(self._browse_folder)
        button_row.addWidget(self.browse_folder_button)

        layout.addLayout(button_row)

    def _apply_style(self, active: bool) -> None:
        self.setStyleSheet(drop_zone_qss("gradeDropZone", active=active))

    def _browse_files(self) -> None:
        paths = pick_image_files(self)
        if paths:
            self.files_selected.emit(paths)

    def _browse_folder(self) -> None:
        folder = pick_image_folder(self)
        if not folder:
            return
        paths = collect_images(folder)
        self.files_selected.emit(paths)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if drop_has_accepted_files(event, IMAGE_EXTENSIONS, recurse_dirs=True):
            event.acceptProposedAction()
            self._apply_style(active=True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self._apply_style(active=False)

    def dropEvent(self, event: QDropEvent) -> None:
        self._apply_style(active=False)
        paths = dropped_local_paths(event, IMAGE_EXTENSIONS, recurse_dirs=True)
        if paths:
            self.files_selected.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class _ClickableBanner(QFrame):
    """QFrame that emits ``clicked`` on left-click (no-model banner)."""

    clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event) -> None:  # noqa: D401 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class _AdaptiveImageLabel(QLabel):
    """QLabel that scales its pixmap to fit."""

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
    """Training-range bar for one output; amber fill up to the prediction."""

    def __init__(self, label: str, low: float, high: float, value: float, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._label = label
        self._low = low
        self._high = high
        self._value = value
        self.setMinimumHeight(64)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

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
        value_rect = QRectF(value_x, labels_top, value_width, bottom_label_height)
        painter.drawText(value_rect, Qt.AlignmentFlag.AlignHCenter, value_text)

        font.setBold(False)
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QColor(TEXT_SECONDARY))
        min_rect = QRectF(track_left, labels_top, 60, bottom_label_height)
        max_rect = QRectF(track_right - 60, labels_top, 60, bottom_label_height)
        overlap_pad = value_rect.adjusted(-6, 0, 6, 0)
        if not min_rect.intersects(overlap_pad):
            painter.drawText(min_rect, Qt.AlignmentFlag.AlignLeft, f"{self._low:.2f}%")
        if not max_rect.intersects(overlap_pad):
            painter.drawText(max_rect, Qt.AlignmentFlag.AlignRight, f"{self._high:.2f}%")

        painter.end()


class _PanGradeCard(QFrame):
    """Closest Pan grade indicator; border colour follows the grade."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("panGradeCard")
        self._apply_frame_style(TEXT_SECONDARY)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        self._title_label = QLabel("")
        self._title_label.setWordWrap(True)
        self._title_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 700; background: transparent;")
        layout.addWidget(self._title_label)

        secondary_label = QLabel("Based on Bitumen vs. training averages.")
        secondary_label.setWordWrap(True)
        secondary_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        layout.addWidget(secondary_label)

    def _apply_frame_style(self, color: str) -> None:
        # Scoped to #panGradeCard -- QLabel is a QFrame subclass in Qt, so a
        # bare "QFrame" selector would also draw this left-border stripe
        # around the nested title/secondary QLabels, not just the card.
        self.setStyleSheet(
            f"QFrame#panGradeCard {{ background-color: {BACKGROUND_COLOR}; border-radius: 6px;"
            f"border-left: 4px solid {color}; }}"
        )

    def set_grade(self, grade: int) -> None:
        self._title_label.setText(f"Closest Pan grade: {grade}")
        self._apply_frame_style(PAN_GRADE_COLORS.get(grade, TEXT_SECONDARY))


