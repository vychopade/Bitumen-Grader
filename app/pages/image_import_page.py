"""
Image Import page.

Provides the UI for uploading bitumen sample images into the application,
organizing them, and launching the image editor for cropping, flipping,
and rotating images before they are used for training or prediction.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PIL import Image
from PyQt6.QtCore import QPointF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFontMetrics,
    QIcon,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.components.image_editor import ImageEditor, pil_to_qpixmap

if TYPE_CHECKING:
    from app.main_window import MainWindow

# --------------------------------------------------------------------------
# Design tokens (kept local so this page has no dependency on MainWindow)
# --------------------------------------------------------------------------

SURFACE_COLOR = "#22252C"
ACCENT_COLOR = "#E8A838"
ACCENT_HOVER_COLOR = "#C98A20"
TEXT_PRIMARY = "#E8E9EC"
TEXT_SECONDARY = "#8B909A"
BUTTON_COLOR = "#2A2E36"

THUMBNAIL_SIZE = 120
RIGHT_PANEL_WIDTH = 300
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif")


def _build_close_icon(color: str, size: int = 12) -> QIcon:
    """Draw a small "x" glyph with QPainter (avoids relying on font glyph coverage)."""
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
    """A single imported image and the thumbnail widget representing it."""

    id: int
    path: str
    image: Image.Image
    thumbnail: "_Thumbnail"


class _DropZone(QFrame):
    """Dashed drag-and-drop target that also offers a "Browse Files" button."""

    files_selected = pyqtSignal(list)

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

        title = QLabel("Drag & drop images here")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600; background: transparent;")
        layout.addWidget(title)

        subtitle = QLabel("Supports JPG, PNG, and TIF \u2014 you can select multiple files at once.")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        layout.addWidget(subtitle)

        browse_button = QPushButton("Browse Files")
        browse_button.setCursor(Qt.CursorShape.PointingHandCursor)
        browse_button.setFixedWidth(140)
        browse_button.setStyleSheet(
            f"QPushButton {{ background-color: {ACCENT_COLOR}; color: #13151A; font-weight: 600;"
            f"border: none; border-radius: 6px; padding: 8px 16px; }}"
            f"QPushButton:hover {{ background-color: {ACCENT_HOVER_COLOR}; }}"
        )
        browse_button.clicked.connect(self._browse_files)
        layout.addWidget(browse_button, 0, Qt.AlignmentFlag.AlignHCenter)

    def _apply_style(self, active: bool) -> None:
        border_color = ACCENT_HOVER_COLOR if active else ACCENT_COLOR
        background = "#2A2E36" if active else SURFACE_COLOR
        self.setStyleSheet(
            f"QFrame#dropZone {{ background-color: {background}; border: 2px dashed {border_color};"
            f"border-radius: 8px; }}"
        )

    def _browse_files(self) -> None:
        file_filter = "Images (*.jpg *.jpeg *.png *.tif)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", file_filter)
        if paths:
            self.files_selected.emit(paths)

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


class _Thumbnail(QWidget):
    """A single 120x120 image thumbnail with a filename label and a remove (\u00d7) button."""

    clicked = pyqtSignal(int)
    remove_requested = pyqtSignal(int)

    def __init__(self, image_id: int, pil_image: Image.Image, filename: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._image_id = image_id
        self._selected = False
        self.setFixedWidth(THUMBNAIL_SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self._image_label = QLabel()
        self._image_label.setFixedSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._image_label)

        self._name_label = QLabel()
        self._name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; background: transparent;")
        self._set_filename(filename)
        outer.addWidget(self._name_label)

        self.set_thumbnail_image(pil_image)
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

    def set_thumbnail_image(self, pil_image: Image.Image) -> None:
        pixmap = pil_to_qpixmap(pil_image).scaled(
            THUMBNAIL_SIZE,
            THUMBNAIL_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(pixmap)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._update_frame_style()

    def _update_frame_style(self) -> None:
        border_color = ACCENT_COLOR if self._selected else "transparent"
        self._image_label.setStyleSheet(
            f"background-color: {SURFACE_COLOR}; border: 2px solid {border_color}; border-radius: 6px;"
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._image_id)
        super().mousePressEvent(event)


class ImageImportPage(QWidget):
    """Page for importing bitumen sample images and editing them before use.

    Loaded images are tracked locally (as PIL images plus their thumbnail
    widgets); the currently selected one is bound to the right-hand
    ``ImageEditor`` panel. "Send to Training" / "Send to Grading" hand the
    current image list off to the rest of the app: they store the payload on
    ``main_window.training_images`` / ``main_window.grading_images`` (read by
    those pages once implemented) and also emit local Qt signals for any
    listener that wants to react directly.
    """

    images_sent_to_training = pyqtSignal(list)
    images_sent_to_grading = pyqtSignal(list)

    def __init__(self, main_window: Optional["MainWindow"] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = main_window

        self._images: List[_LoadedImage] = []
        self._selected_id: Optional[int] = None
        self._id_counter = itertools.count(1)

        self._thumbnail_row: Optional[QHBoxLayout] = None
        self._empty_strip_label: Optional[QLabel] = None
        self._editor: Optional[ImageEditor] = None
        self._editor_placeholder: Optional[QLabel] = None
        self._count_label: Optional[QLabel] = None
        self._feedback_label: Optional[QLabel] = None
        self._send_training_button: Optional[QPushButton] = None
        self._send_grading_button: Optional[QPushButton] = None

        self._build_ui()
        self._refresh_action_bar()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 24)
        root.setSpacing(18)

        root.addLayout(self._build_header())

        content_row = QHBoxLayout()
        content_row.setSpacing(20)

        left_column = QVBoxLayout()
        left_column.setSpacing(16)

        self._drop_zone = _DropZone()
        self._drop_zone.files_selected.connect(self._add_images)
        left_column.addWidget(self._drop_zone)

        left_column.addWidget(self._build_thumbnail_strip())
        left_column.addStretch(1)

        content_row.addLayout(left_column, 1)
        content_row.addWidget(self._build_right_panel())

        root.addLayout(content_row, 1)
        root.addLayout(self._build_action_bar())

    def _build_header(self) -> QVBoxLayout:
        header = QVBoxLayout()
        header.setSpacing(4)

        title = QLabel("Import Images")
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 20px; font-weight: 600;")
        header.addWidget(title)

        subtitle = QLabel("Upload and prepare your bitumen samples before training or grading.")
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px;")
        header.addWidget(subtitle)

        return header

    def _build_thumbnail_strip(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(176)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")

        container = QWidget()
        container.setStyleSheet("background-color: transparent;")
        self._thumbnail_row = QHBoxLayout(container)
        self._thumbnail_row.setContentsMargins(4, 4, 4, 4)
        self._thumbnail_row.setSpacing(14)

        self._empty_strip_label = QLabel("No images loaded yet \u2014 add some above.")
        self._empty_strip_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; background: transparent;")
        self._thumbnail_row.addWidget(self._empty_strip_label)
        self._thumbnail_row.addStretch(1)

        scroll.setWidget(container)
        return scroll

    def _build_right_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("editorPanel")
        panel.setFixedWidth(RIGHT_PANEL_WIDTH)
        panel.setStyleSheet(f"QFrame#editorPanel {{ background-color: {SURFACE_COLOR}; border-radius: 8px; }}")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("Edit Image")
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 600; background: transparent;")
        layout.addWidget(title)

        self._editor_placeholder = QLabel("Select a thumbnail to edit it here.")
        self._editor_placeholder.setWordWrap(True)
        self._editor_placeholder.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 12px; background: transparent;"
        )
        layout.addWidget(self._editor_placeholder)

        self._editor = ImageEditor()
        self._editor.image_updated.connect(self._on_editor_image_updated)
        self._editor.setVisible(False)
        layout.addWidget(self._editor)

        layout.addStretch(1)
        return panel

    def _build_action_bar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)

        self._send_training_button = QPushButton("Send to Training")
        self._send_training_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._send_training_button.setStyleSheet(
            f"QPushButton {{ background-color: {ACCENT_COLOR}; color: #13151A; font-weight: 600;"
            f"border: none; border-radius: 6px; padding: 9px 18px; }}"
            f"QPushButton:hover {{ background-color: {ACCENT_HOVER_COLOR}; }}"
            f"QPushButton:disabled {{ background-color: #4A4230; color: #8B8168; }}"
        )
        self._send_training_button.clicked.connect(self._send_to_training)
        row.addWidget(self._send_training_button)

        self._send_grading_button = QPushButton("Send to Grading")
        self._send_grading_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._send_grading_button.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {ACCENT_COLOR};"
            f"border: 1px solid {ACCENT_COLOR}; border-radius: 6px; padding: 9px 18px; }}"
            f"QPushButton:hover {{ background-color: rgba(232, 168, 56, 30); }}"
            f"QPushButton:disabled {{ color: #8B909A; border: 1px solid #3A3E46; }}"
        )
        self._send_grading_button.clicked.connect(self._send_to_grading)
        row.addWidget(self._send_grading_button)

        self._feedback_label = QLabel("")
        self._feedback_label.setStyleSheet(f"color: {ACCENT_COLOR}; font-size: 12px; background: transparent;")
        row.addWidget(self._feedback_label)

        row.addStretch(1)

        self._count_label = QLabel("0 image(s) loaded")
        self._count_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; background: transparent;")
        row.addWidget(self._count_label)

        return row

    # -- Image list management ----------------------------------------------

    def _add_images(self, paths: List[str]) -> None:
        added = 0
        for path in paths:
            if not path.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            try:
                with Image.open(path) as opened:
                    opened.load()
                    image = opened.convert("RGB")
            except (OSError, ValueError):
                continue

            image_id = next(self._id_counter)
            thumbnail = _Thumbnail(image_id, image, Path(path).name)
            thumbnail.clicked.connect(self._select_image)
            thumbnail.remove_requested.connect(self._remove_image)

            insert_index = self._thumbnail_row.count() - 1
            self._thumbnail_row.insertWidget(max(insert_index, 0), thumbnail)
            self._images.append(_LoadedImage(id=image_id, path=path, image=image, thumbnail=thumbnail))
            added += 1

        if added:
            if self._empty_strip_label is not None:
                self._empty_strip_label.setVisible(False)
            if self._selected_id is None:
                self._select_image(self._images[0].id)

        self._refresh_action_bar()

    def _remove_image(self, image_id: int) -> None:
        index = next((i for i, item in enumerate(self._images) if item.id == image_id), None)
        if index is None:
            return

        item = self._images.pop(index)
        item.thumbnail.setParent(None)
        item.thumbnail.deleteLater()

        if self._selected_id == image_id:
            self._selected_id = None
            if self._images:
                self._select_image(self._images[0].id)
            else:
                self._clear_editor()

        if not self._images and self._empty_strip_label is not None:
            self._empty_strip_label.setVisible(True)

        self._refresh_action_bar()

    def _select_image(self, image_id: int) -> None:
        self._selected_id = image_id
        for item in self._images:
            item.thumbnail.set_selected(item.id == image_id)

        item = self._find_image(image_id)
        if item is None or self._editor is None or self._editor_placeholder is None:
            return

        self._editor_placeholder.setVisible(False)
        self._editor.setVisible(True)
        self._editor.set_image(item.image)

    def _clear_editor(self) -> None:
        if self._editor is None or self._editor_placeholder is None:
            return
        self._editor.clear()
        self._editor.setVisible(False)
        self._editor_placeholder.setVisible(True)

    def _find_image(self, image_id: int) -> Optional[_LoadedImage]:
        return next((item for item in self._images if item.id == image_id), None)

    def _on_editor_image_updated(self, pil_image: Image.Image) -> None:
        if self._selected_id is None:
            return
        item = self._find_image(self._selected_id)
        if item is None:
            return
        item.image = pil_image
        item.thumbnail.set_thumbnail_image(pil_image)

    # -- Action bar ----------------------------------------------------------

    def _refresh_action_bar(self) -> None:
        count = len(self._images)
        if self._count_label is not None:
            self._count_label.setText(f"{count} image(s) loaded")
        if self._send_training_button is not None:
            self._send_training_button.setEnabled(count > 0)
        if self._send_grading_button is not None:
            self._send_grading_button.setEnabled(count > 0)
        if self._feedback_label is not None:
            self._feedback_label.setText("")

    def _build_payload(self) -> List[Dict[str, Any]]:
        return [{"path": item.path, "image": item.image.copy()} for item in self._images]

    def _send_to_training(self) -> None:
        if not self._images:
            return
        payload = self._build_payload()
        if self.main_window is not None:
            self.main_window.training_images = payload
        self.images_sent_to_training.emit(payload)
        self._show_feedback(f"Sent {len(payload)} image(s) to Training.")

    def _send_to_grading(self) -> None:
        if not self._images:
            return
        payload = self._build_payload()
        if self.main_window is not None:
            self.main_window.grading_images = payload
        self.images_sent_to_grading.emit(payload)
        self._show_feedback(f"Sent {len(payload)} image(s) to Grading.")

    def _show_feedback(self, message: str) -> None:
        if self._feedback_label is not None:
            self._feedback_label.setText(message)
