"""
Import Images page.

Add sample photos, edit them (crop/flip/rotate), then send paths to
training or grading. Thumbnails load in batches on a background thread
so big imports don't freeze the UI.
"""
from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from PIL import Image
from PyQt6.QtCore import QObject, QPointF, QSize, Qt, QThread, pyqtSignal
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

from app.components.image_editor import ImageEditor, pil_to_qimage
from app.constants import IMAGE_EXTENSIONS
from app.theme import (
    ACCENT_COLOR,
    ACCENT_TINT_HOVER,
    BORDER_COLOR,
    DANGER_COLOR,
    PAGE_MARGINS,
    PAGE_SPACING,
    SURFACE_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    accent_button_qss,
    drop_zone_qss,
    ghost_button_qss,
)
from app.utils.shortcuts import bind_page_shortcuts, shortcut_tooltip, unbind_page_shortcuts

if TYPE_CHECKING:
    from app.main_window import MainWindow

THUMBNAIL_SIZE = 120
RIGHT_PANEL_WIDTH = 300
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS

#: Thumbnails built per batch. The rest wait behind "Load More" so huge
#: imports don't create thousands of widgets at once.
BATCH_SIZE = 100

#: Above this many images, show a one-line "large dataset" notice.
LARGE_DATASET_THRESHOLD = 2000

#: Temp folder for edited images so everything still works from a path on disk.
_EDIT_CACHE_DIR = Path(gettempdir()) / "bitumengrader_edited_images"


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
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        for image_id, path in self._items:
            if self._stop_requested:
                break
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
        return path.lower().endswith(SUPPORTED_EXTENSIONS)

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


class ImageImportPage(QWidget):
    """Import sample photos and edit them before training or grading.

    Tracked by path; only the selected image is fully decoded. Thumbnails
    load in ``BATCH_SIZE`` batches. "Send to Training" / "Send to Grading"
    store path lists on ``main_window`` and switch to that page.
    """

    def __init__(self, main_window: Optional["MainWindow"] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = main_window

        self._images: List[_LoadedImage] = []
        self._pending_paths: List[str] = []
        self._selected_id: Optional[int] = None
        self._id_counter = itertools.count(1)

        self._thumbnail_thread: Optional[QThread] = None
        self._thumbnail_worker: Optional[_ThumbnailWorker] = None
        self._batch_failed_names: List[str] = []

        self._thumbnail_row: Optional[QHBoxLayout] = None
        self._empty_strip_label: Optional[QLabel] = None
        self._load_more_button: Optional[QPushButton] = None
        self._dataset_warning_banner: Optional[QFrame] = None
        self._editor: Optional[ImageEditor] = None
        self._editor_placeholder: Optional[QLabel] = None
        self._count_label: Optional[QLabel] = None
        self._feedback_label: Optional[QLabel] = None
        self._send_training_button: Optional[QPushButton] = None
        self._send_grading_button: Optional[QPushButton] = None
        self._shortcut_bindings: List[tuple] = []
        self._tab_order_applied = False

        self._build_ui()
        self._refresh_action_bar()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(*PAGE_MARGINS)
        root.setSpacing(PAGE_SPACING)

        root.addLayout(self._build_header())

        self._dataset_warning_banner = self._build_dataset_warning_banner()
        root.addWidget(self._dataset_warning_banner)

        content_row = QHBoxLayout()
        content_row.setSpacing(20)

        left_column = QVBoxLayout()
        left_column.setSpacing(16)

        self._drop_zone = _DropZone()
        self._drop_zone.files_selected.connect(self._add_images)
        self._drop_zone.folder_empty.connect(self._on_folder_empty)
        self._drop_zone.browse_button.setToolTip(shortcut_tooltip("Browse for image files", "B"))
        self._drop_zone.browse_folder_button.setToolTip(shortcut_tooltip("Browse for an image folder", "F"))
        self._shortcut_bindings.append((self._drop_zone.browse_button, "B"))
        self._shortcut_bindings.append((self._drop_zone.browse_folder_button, "F"))
        left_column.addWidget(self._drop_zone)

        left_column.addWidget(self._build_thumbnail_strip())
        left_column.addStretch(1)

        content_row.addLayout(left_column, 1)
        content_row.addWidget(self._build_right_panel())

        root.addLayout(content_row, 1)
        root.addLayout(self._build_action_bar())

    def _apply_tab_order(self) -> None:
        """Focus order: drop zone → editor → send buttons."""
        chain = [
            self._drop_zone.browse_button,
            self._drop_zone.browse_folder_button,
            self._editor.crop_button,
            self._editor.flip_h_button,
            self._editor.flip_v_button,
            self._editor.rotate_ccw_button,
            self._editor.rotate_cw_button,
            self._editor.reset_button,
            self._editor.apply_button,
            self._send_training_button,
            self._send_grading_button,
        ]
        for earlier, later in zip(chain, chain[1:]):
            if earlier is not None and later is not None:
                QWidget.setTabOrder(earlier, later)

    def showEvent(self, event) -> None:  # noqa: D401 - Qt override
        super().showEvent(event)
        bind_page_shortcuts(self._shortcut_bindings)
        if not self._tab_order_applied:
            # Deferred until we're under the main window (setTabOrder needs a shared window).
            self._apply_tab_order()
            self._tab_order_applied = True

    def hideEvent(self, event) -> None:  # noqa: D401 - Qt override
        super().hideEvent(event)
        unbind_page_shortcuts(self._shortcut_bindings)

    def _build_header(self) -> QVBoxLayout:
        header = QVBoxLayout()
        header.setSpacing(4)

        title = QLabel("Import Images")
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 20px; font-weight: 600;")
        header.addWidget(title)

        subtitle = QLabel("Add photos, then edit them before training or grading.")
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px;")
        header.addWidget(subtitle)

        return header

    def _build_dataset_warning_banner(self) -> QFrame:
        banner = QFrame()
        banner.setObjectName("datasetWarningBanner")
        banner.setStyleSheet(
            # Scoped to #datasetWarningBanner -- QLabel is a QFrame subclass
            # in Qt, so a bare "QFrame" selector would also draw this border
            # around the nested label, not just the banner.
            f"QFrame#datasetWarningBanner {{ background-color: {ACCENT_TINT_HOVER}; border: 1px solid {ACCENT_COLOR};"
            f"border-radius: 8px; }}"
        )
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(14, 10, 14, 10)

        label = QLabel("Large dataset \u2014 thumbnails load in batches.")
        label.setStyleSheet(f"color: {ACCENT_COLOR}; font-size: 12px; font-weight: 600; background: transparent;")
        layout.addWidget(label)

        banner.setVisible(False)
        return banner

    def _build_thumbnail_strip(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(176)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")

        container = QWidget()
        container.setStyleSheet("background-color: transparent;")
        self._thumbnail_row = QHBoxLayout(container)
        self._thumbnail_row.setContentsMargins(4, 4, 4, 4)
        self._thumbnail_row.setSpacing(14)

        self._empty_strip_label = QLabel("No images yet \u2014 add some above.")
        self._empty_strip_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; background: transparent;")
        self._thumbnail_row.addWidget(self._empty_strip_label)

        self._load_more_button = QPushButton("Load More")
        self._load_more_button.setFixedSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        self._load_more_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._load_more_button.setStyleSheet(
            f"QPushButton {{ background-color: {SURFACE_COLOR}; color: {ACCENT_COLOR}; font-weight: 600;"
            f"font-size: 11px; border: 2px dashed {ACCENT_COLOR}; border-radius: 6px; }}"
            f"QPushButton:hover {{ background-color: {ACCENT_TINT_HOVER}; }}"
            f"QPushButton:disabled {{ color: {TEXT_SECONDARY}; border: 2px dashed {BORDER_COLOR}; }}"
        )
        self._load_more_button.clicked.connect(self._on_load_more_clicked)
        self._load_more_button.setVisible(False)
        self._thumbnail_row.addWidget(self._load_more_button)

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

        self._editor_placeholder = QLabel("Select a thumbnail to edit.")
        self._editor_placeholder.setWordWrap(True)
        self._editor_placeholder.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 12px; background: transparent;"
        )
        layout.addWidget(self._editor_placeholder)

        self._editor = ImageEditor()
        self._editor.image_updated.connect(self._on_editor_image_updated)
        self._editor.setVisible(False)
        layout.addWidget(self._editor)

        editor_shortcuts = (
            (self._editor.crop_button, "C", "Crop the image"),
            (self._editor.flip_h_button, "H", "Flip Horizontal"),
            (self._editor.flip_v_button, "V", "Flip Vertical"),
            (self._editor.rotate_cw_button, "W", "Rotate 90\u00b0 CW"),
            (self._editor.rotate_ccw_button, "Q", "Rotate 90\u00b0 CCW"),
            (self._editor.reset_button, "O", "Reset to original"),
            (self._editor.apply_button, "P", "Apply Changes"),
        )
        for button, letter, description in editor_shortcuts:
            button.setToolTip(shortcut_tooltip(description, letter))
            self._shortcut_bindings.append((button, letter))

        layout.addStretch(1)
        return panel

    def _build_action_bar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)

        self._send_training_button = QPushButton("Send to Training")
        self._send_training_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._send_training_button.setStyleSheet(accent_button_qss(extra="padding: 9px 18px;"))
        self._send_training_button.setToolTip(shortcut_tooltip("Send loaded images to Train Model", "S"))
        self._send_training_button.clicked.connect(self._send_to_training)
        row.addWidget(self._send_training_button)
        self._shortcut_bindings.append((self._send_training_button, "S"))

        self._send_grading_button = QPushButton("Send to Grading")
        self._send_grading_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._send_grading_button.setStyleSheet(ghost_button_qss(extra="padding: 9px 18px;"))
        self._send_grading_button.setToolTip(shortcut_tooltip("Send loaded images to Grade Images", "D"))
        self._send_grading_button.clicked.connect(self._send_to_grading)
        row.addWidget(self._send_grading_button)
        self._shortcut_bindings.append((self._send_grading_button, "D"))

        self._feedback_label = QLabel("")
        self._feedback_label.setWordWrap(True)
        self._feedback_label.setStyleSheet(f"color: {ACCENT_COLOR}; font-size: 12px; background: transparent;")
        row.addWidget(self._feedback_label)

        row.addStretch(1)

        self._count_label = QLabel("0 images loaded")
        self._count_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; background: transparent;")
        row.addWidget(self._count_label)

        return row

    # -- Image list management (paginated + threaded thumbnails) ------------

    def _all_known_paths(self) -> List[str]:
        """Every path added this session, including ones still waiting on thumbnails."""
        return [item.path for item in self._images] + list(self._pending_paths)

    def _add_images(self, paths: List[str]) -> None:
        already_known = {item.path for item in self._images} | set(self._pending_paths)
        new_paths: List[str] = []
        for path in paths:
            if not path.lower().endswith(SUPPORTED_EXTENSIONS) or path in already_known:
                continue
            new_paths.append(path)
            already_known.add(path)

        if not new_paths:
            return

        self._pending_paths.extend(new_paths)
        self._load_next_batch()
        self._refresh_action_bar()
        self._update_dataset_warning()

    def _on_folder_empty(self, folder: str) -> None:
        self._show_feedback(
            f"No JPG/PNG/TIF images found in \u201c{Path(folder).name}\u201d.",
            danger=True,
        )
    def _on_load_more_clicked(self) -> None:
        self._load_next_batch()

    def _load_next_batch(self) -> None:
        """Take up to ``BATCH_SIZE`` pending paths and thumbnail them in the background."""
        if not self._pending_paths or self._thumbnail_thread is not None:
            self._update_load_more_button()
            return

        batch = self._pending_paths[:BATCH_SIZE]
        del self._pending_paths[:BATCH_SIZE]

        items: List[Tuple[int, str]] = []
        for path in batch:
            image_id = next(self._id_counter)
            thumbnail = _Thumbnail(image_id, Path(path).name)
            thumbnail.clicked.connect(self._select_image)
            thumbnail.remove_requested.connect(self._remove_image)

            # Widgets always stay in order [thumbnails..., load_more_button, stretch];
            # insert new thumbnails just before those two trailing items.
            insert_index = max(self._thumbnail_row.count() - 2, 0)
            self._thumbnail_row.insertWidget(insert_index, thumbnail)

            self._images.append(_LoadedImage(id=image_id, path=path, image=None, thumbnail=thumbnail))
            items.append((image_id, path))

        if self._empty_strip_label is not None:
            self._empty_strip_label.setVisible(False)
        if self._selected_id is None and self._images:
            self._select_image(self._images[0].id)

        self._start_thumbnail_worker(items)
        self._update_load_more_button()
        self._refresh_action_bar()

    def _start_thumbnail_worker(self, items: List[Tuple[int, str]]) -> None:
        self._batch_failed_names = []
        self._thumbnail_thread = QThread(self)
        self._thumbnail_worker = _ThumbnailWorker(items)
        self._thumbnail_worker.moveToThread(self._thumbnail_thread)

        self._thumbnail_thread.started.connect(self._thumbnail_worker.run)
        self._thumbnail_worker.thumbnail_ready.connect(self._on_thumbnail_ready)
        self._thumbnail_worker.thumbnail_failed.connect(self._on_thumbnail_failed)
        self._thumbnail_worker.finished.connect(self._thumbnail_thread.quit)
        self._thumbnail_thread.finished.connect(self._on_thumbnail_thread_finished)

        self._thumbnail_thread.start()

    def _on_thumbnail_ready(self, image_id: int, qimage: QImage) -> None:
        item = self._find_image(image_id)
        if item is not None:
            item.thumbnail.set_thumbnail_from_qimage(qimage)

    def _on_thumbnail_failed(self, image_id: int, filename: str) -> None:
        item = self._find_image(image_id)
        if item is not None:
            item.thumbnail.set_error()
        self._batch_failed_names.append(filename)

    def _on_thumbnail_thread_finished(self) -> None:
        if self._thumbnail_thread is not None:
            self._thumbnail_thread.deleteLater()
        if self._thumbnail_worker is not None:
            self._thumbnail_worker.deleteLater()
        self._thumbnail_thread = None
        self._thumbnail_worker = None

        if self._batch_failed_names:
            self._show_load_error(self._batch_failed_names)
            self._batch_failed_names = []

        self._update_load_more_button()

    def _update_load_more_button(self) -> None:
        if self._load_more_button is None:
            return
        remaining = len(self._pending_paths)
        loading = self._thumbnail_thread is not None

        self._load_more_button.setVisible(remaining > 0 or loading)
        if loading:
            self._load_more_button.setEnabled(False)
            self._load_more_button.setText("Loading\u2026")
        else:
            self._load_more_button.setEnabled(True)
            shown = min(BATCH_SIZE, remaining)
            self._load_more_button.setText(f"Load {shown} More\n({remaining} left)")

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

        if not self._images and not self._pending_paths and self._empty_strip_label is not None:
            self._empty_strip_label.setVisible(True)

        self._refresh_action_bar()
        self._update_dataset_warning()

    def _select_image(self, image_id: int) -> None:
        previous_id = self._selected_id
        self._selected_id = image_id
        for item in self._images:
            item.thumbnail.set_selected(item.id == image_id)

        item = self._find_image(image_id)
        if item is None or self._editor is None or self._editor_placeholder is None:
            return

        if item.image is None:
            try:
                with Image.open(item.path) as opened:
                    opened.load()
                    item.image = opened.convert("RGB")
            except (OSError, ValueError):
                self._show_feedback(f"Couldn't open \u201c{Path(item.path).name}\u201d for editing.", danger=True)
                # Keep the previous selection if this one won't open.
                self._selected_id = previous_id
                for other in self._images:
                    other.thumbnail.set_selected(other.id == previous_id)
                return

        self._editor_placeholder.setVisible(False)
        self._editor.setVisible(True)
        self._editor.set_image(item.image)

        # Drop the previous selection's decoded copy so RAM doesn't grow
        # just from clicking around.
        if previous_id is not None and previous_id != image_id:
            previous_item = self._find_image(previous_id)
            if previous_item is not None:
                previous_item.image = None

    def _clear_editor(self) -> None:
        if self._editor is None or self._editor_placeholder is None:
            return
        self._editor.clear()
        self._editor.setVisible(False)
        self._editor_placeholder.setVisible(True)

    def _find_image(self, image_id: int) -> Optional[_LoadedImage]:
        return next((item for item in self._images if item.id == image_id), None)

    def _persist_edited_image(self, item: _LoadedImage, pil_image: Image.Image) -> str:
        """Save the edited image to a temp file and return the new path.

        Training/grading re-read from ``item.path``, so edits have to land
        on disk — not only in a PIL object — or they'd be lost.
        """
        _EDIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(item.path).stem
        filename = f"{stem}_{item.id}_{uuid.uuid4().hex[:8]}.png"
        new_path = _EDIT_CACHE_DIR / filename
        pil_image.save(new_path, format="PNG")
        return str(new_path)

    def _on_editor_image_updated(self, pil_image: Image.Image) -> None:
        if self._selected_id is None:
            return
        item = self._find_image(self._selected_id)
        if item is None:
            return

        try:
            new_path = self._persist_edited_image(item, pil_image)
        except OSError as exc:
            self._show_feedback(f"Couldn't save edit to disk: {exc}", danger=True)
            return

        item.path = new_path
        item.image = pil_image
        item.thumbnail.set_thumbnail_image(pil_image)

    # -- Action bar ----------------------------------------------------------

    def _refresh_action_bar(self) -> None:
        loaded = len(self._images)
        pending = len(self._pending_paths)
        total = loaded + pending

        if self._count_label is not None:
            if pending:
                self._count_label.setText(f"{total} images loaded ({pending} thumbnails pending)")
            else:
                self._count_label.setText(f"{total} images loaded")

        if self._send_training_button is not None:
            self._send_training_button.setEnabled(total > 0)
        if self._send_grading_button is not None:
            self._send_grading_button.setEnabled(total > 0)

    def _update_dataset_warning(self) -> None:
        if self._dataset_warning_banner is None:
            return
        total = len(self._images) + len(self._pending_paths)
        self._dataset_warning_banner.setVisible(total > LARGE_DATASET_THRESHOLD)

    def _build_path_payload(self) -> List[Dict[str, Any]]:
        """Path-only list for training/grading (avoids duplicating decoded images)."""
        return [{"path": path} for path in self._all_known_paths()]

    def _send_to_training(self) -> None:
        if not self._images and not self._pending_paths:
            return
        payload = self._build_path_payload()
        if self.main_window is not None:
            self.main_window.training_images = payload
            self.main_window.navigate_to("train")
        self._show_feedback(f"Sent {len(payload)} images to Training.")

    def _send_to_grading(self) -> None:
        if not self._images and not self._pending_paths:
            return
        payload = self._build_path_payload()
        if self.main_window is not None:
            self.main_window.grading_images = payload
            self.main_window.navigate_to("grade")
        self._show_feedback(f"Sent {len(payload)} images to Grading.")

    def _show_feedback(self, message: str, *, danger: bool = False) -> None:
        if self._feedback_label is None:
            return
        color = DANGER_COLOR if danger else ACCENT_COLOR
        self._feedback_label.setStyleSheet(f"color: {color}; font-size: 12px; background: transparent;")
        self._feedback_label.setText(message)

    def _show_load_error(self, failed_names: List[str]) -> None:
        names = ", ".join(failed_names[:3])
        if len(failed_names) > 3:
            names += f", and {len(failed_names) - 3} more"
        count_word = "image" if len(failed_names) == 1 else "images"
        self._show_feedback(f"Couldn't load {len(failed_names)} {count_word}: {names}", danger=True)
