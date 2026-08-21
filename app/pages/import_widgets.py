"""Import-page widgets: drop zone, thumbnails, and thumbnail worker."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from PyQt6.QtCore import QObject, QPointF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFontMetrics,
    QIcon,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import QFileDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.components.image_editor import pil_to_qimage
from app.constants import IMAGE_EXTENSIONS
from app.theme import (
    ACCENT_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    accent_button_qss,
    drop_zone_qss,
    ghost_button_qss,
)

THUMBNAIL_SIZE = 120

def _build_close_icon(color: str, size: int = 12) -> QIcon:
    """Small × icon drawn with QPainter."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)

    margin = size * 0.18
    painter.drawLine(QPointF(margin, margin), QPointF(size - margin, size - margin))
    painter.drawLine(QPointF(size - margin, margin), QPointF(margin, size - margin))
    painter.end()

    return QIcon(pixmap)


@dataclass
class _LoadedImage:
    """One imported image and its thumbnail widget.

    ``image`` is only set for the selected item (for the editor); everything
    else is just a path, so RAM stays roughly flat no matter how many you load.
    """

    id: int
    path: str
    image: Optional[Image.Image]
    thumbnail: "_Thumbnail"


class _ThumbnailWorker(QObject):
    """Open and shrink images to thumbnail QImages off the GUI thread.

    ``QImage`` is safe to build here; convert to ``QPixmap`` on the main thread.
    """

    thumbnail_ready = pyqtSignal(int, QImage)
    thumbnail_failed = pyqtSignal(int, str)
    finished = pyqtSignal()

    def __init__(self, items: List[Tuple[int, str]]):
        super().__init__()
        self._items = items

    def run(self) -> None:
        for image_id, path in self._items:
            try:
                with Image.open(path) as opened:
                    opened.load()
                    rgb = opened.convert("RGB")
            except (OSError, ValueError):
                self.thumbnail_failed.emit(image_id, Path(path).name)
                continue

            thumb = rgb.copy()
            thumb.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
            self.thumbnail_ready.emit(image_id, pil_to_qimage(thumb))

        self.finished.emit()


class _DropZone(QFrame):
    """Dashed drop area with Browse Files / Browse Folder."""

    files_selected = pyqtSignal(list)
    #: Fired when a picked folder has no supported images.
    folder_empty = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setFixedHeight(140)
        self._build_ui()
        self._apply_style(active=False)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("Drop images or a folder here")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600; background: transparent;")
        layout.addWidget(title)

        subtitle = QLabel("JPG, PNG, or TIF \u2014 files or a whole folder.")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        layout.addWidget(subtitle)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        button_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.browse_button = QPushButton("Browse Files")
        self.browse_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_button.setFixedWidth(140)
        self.browse_button.setStyleSheet(accent_button_qss())
        self.browse_button.clicked.connect(self._browse_files)
        button_row.addWidget(self.browse_button)

        self.browse_folder_button = QPushButton("Browse Folder")
        self.browse_folder_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_folder_button.setFixedWidth(140)
        self.browse_folder_button.setStyleSheet(ghost_button_qss())
        self.browse_folder_button.clicked.connect(self._browse_folder)
        button_row.addWidget(self.browse_folder_button)

        layout.addLayout(button_row)

    def _apply_style(self, active: bool) -> None:
        self.setStyleSheet(drop_zone_qss("dropZone", active=active))

    def _browse_files(self) -> None:
        file_filter = "Images (*.jpg *.jpeg *.png *.tif)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", file_filter)
        if paths:
            self.files_selected.emit(paths)

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        paths = self._collect_images_from_folder(folder)
        if paths:
            self.files_selected.emit(paths)
        else:
            self.folder_empty.emit(folder)

    @classmethod
    def _collect_images_from_folder(cls, folder: str) -> List[str]:
        """Supported image paths in ``folder`` (top level only, sorted)."""
        directory = Path(folder)
        try:
            entries = list(directory.iterdir())
        except OSError:
            return []
        paths = [
            str(entry)
            for entry in entries
            if entry.is_file() and cls._is_supported(str(entry))
        ]
        paths.sort()
        return paths

    @staticmethod
    def _is_supported(path: str) -> bool:
        return path.lower().endswith(IMAGE_EXTENSIONS)

    def _paths_from_urls(self, urls) -> List[str]:
        """Turn dropped file/folder URLs into supported image paths."""
        paths: List[str] = []
        seen: set[str] = set()
        for url in urls:
            if not url.isLocalFile():
                continue
            local_path = url.toLocalFile()
            candidate = Path(local_path)
            if candidate.is_dir():
                for image_path in self._collect_images_from_folder(local_path):
                    if image_path not in seen:
                        paths.append(image_path)
                        seen.add(image_path)
            elif self._is_supported(local_path) and local_path not in seen:
                paths.append(local_path)
                seen.add(local_path)
        return paths

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        mime_data = event.mimeData()
        if not mime_data.hasUrls():
            event.ignore()
            return

        acceptable = False
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            local_path = url.toLocalFile()
            if Path(local_path).is_dir() or self._is_supported(local_path):
                acceptable = True
                break

        if acceptable:
            event.acceptProposedAction()
            self._apply_style(active=True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: D401 - Qt override
        self._apply_style(active=False)

    def dropEvent(self, event: QDropEvent) -> None:
        self._apply_style(active=False)
        paths = self._paths_from_urls(event.mimeData().urls())
        if paths:
            self.files_selected.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class _Thumbnail(QWidget):
    """120×120 thumbnail with filename and a remove (×) button.

    Built with just a name; pixels arrive later via ``set_thumbnail_from_qimage``
    or ``set_error`` if decode fails.
    """

    clicked = pyqtSignal(int)
    remove_requested = pyqtSignal(int)

    def __init__(self, image_id: int, filename: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._image_id = image_id
        self._selected = False
        self._errored = False
        self.setFixedWidth(THUMBNAIL_SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self._image_label = QLabel("\u2026")
        self._image_label.setFixedSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._image_label)

        self._name_label = QLabel()
        self._name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; background: transparent;")
        self._set_filename(filename)
        outer.addWidget(self._name_label)

        self._update_frame_style()

        self._remove_button = QPushButton(self._image_label)
        self._remove_button.setIcon(_build_close_icon(TEXT_PRIMARY))
        self._remove_button.setIconSize(QSize(12, 12))
        self._remove_button.setFixedSize(20, 20)
        self._remove_button.move(THUMBNAIL_SIZE - 22, 2)
        self._remove_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._remove_button.setToolTip("Remove image")
        self._remove_button.setStyleSheet(
            "QPushButton { background-color: rgba(19, 21, 26, 200);"
            "border: none; border-radius: 10px; }"
            "QPushButton:hover { background-color: rgba(232, 168, 56, 230); }"
        )
        self._remove_button.clicked.connect(lambda: self.remove_requested.emit(self._image_id))

    def _set_filename(self, filename: str) -> None:
        metrics = QFontMetrics(self._name_label.font())
        elided = metrics.elidedText(filename, Qt.TextElideMode.ElideMiddle, THUMBNAIL_SIZE)
        self._name_label.setText(elided)
        self._name_label.setToolTip(filename)

    def set_thumbnail_from_qimage(self, qimage: QImage) -> None:
        self._errored = False
        self._image_label.setText("")
        pixmap = QPixmap.fromImage(qimage).scaled(
            THUMBNAIL_SIZE,
            THUMBNAIL_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(pixmap)
        self._update_frame_style()

    def set_thumbnail_image(self, pil_image: Image.Image) -> None:
        self.set_thumbnail_from_qimage(pil_to_qimage(pil_image))

    def set_error(self) -> None:
        self._errored = True
        self._image_label.setPixmap(QPixmap())
        self._image_label.setText("Couldn't\nload image")
        self._update_frame_style()

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._update_frame_style()

    def _update_frame_style(self) -> None:
        if self._errored:
            border_color = DANGER_COLOR
        elif self._selected:
            border_color = ACCENT_COLOR
        else:
            border_color = "transparent"
        text_color = DANGER_COLOR if self._errored else TEXT_SECONDARY
        self._image_label.setStyleSheet(
            f"background-color: {SURFACE_COLOR}; border: 2px solid {border_color}; border-radius: 6px;"
            f"color: {text_color}; font-size: 10px;"
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._image_id)
        super().mousePressEvent(event)


