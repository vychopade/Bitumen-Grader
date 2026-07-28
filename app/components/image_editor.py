"""
Image editor widget.

Reusable widget providing basic image editing operations needed before
training/prediction: cropping, flipping (horizontal/vertical), and rotating
an image, with a live preview. Edits are non-destructive (applied to an
in-memory working copy) until the user presses "Apply Changes", at which
point ``image_updated`` is emitted with the final Pillow image.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from PIL import Image
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QIcon,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# --------------------------------------------------------------------------
# Design tokens (kept local so this component has no dependency on MainWindow)
# --------------------------------------------------------------------------

SURFACE_COLOR = "#22252C"
BORDER_COLOR = "#33373F"
ACCENT_COLOR = "#E8A838"
ACCENT_HOVER_COLOR = "#C98A20"
TEXT_PRIMARY = "#E8E9EC"
TEXT_SECONDARY = "#8B909A"
WARNING_COLOR = "#F5C518"
DANGER_COLOR = "#E5484D"
BUTTON_COLOR = "#2A2E36"
BUTTON_HOVER_COLOR = "#33373F"

PREVIEW_SIZE = 260
CROP_DIALOG_MAX = 480
MIN_CROP_SIZE = 24
HANDLE_SIZE = 10


def pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """Convert a PIL image to a QPixmap via QImage.

    Copies the underlying QImage buffer so the returned QPixmap remains
    valid after the source ``bytes`` object is garbage collected.
    """
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimage = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimage.copy())


def _build_icon(kind: str, color: str, size: int = 16) -> QIcon:
    """Draw a small flat line-icon (crop / flip / rotate) with QPainter primitives."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    pen = QPen(QColor(color))
    pen.setWidthF(1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)

    margin = size * 0.15

    if kind == "flip_h":
        mid_x = size / 2
        painter.drawLine(QPointF(mid_x, margin), QPointF(mid_x, size - margin))
        painter.drawLine(QPointF(margin, size / 2), QPointF(mid_x - size * 0.12, size / 2))
        painter.drawPolyline(
            QPolygonF(
                [
                    QPointF(margin + size * 0.08, size / 2 - size * 0.12),
                    QPointF(margin, size / 2),
                    QPointF(margin + size * 0.08, size / 2 + size * 0.12),
                ]
            )
        )
        painter.drawLine(QPointF(mid_x + size * 0.12, size / 2), QPointF(size - margin, size / 2))
        painter.drawPolyline(
            QPolygonF(
                [
                    QPointF(size - margin - size * 0.08, size / 2 - size * 0.12),
                    QPointF(size - margin, size / 2),
                    QPointF(size - margin - size * 0.08, size / 2 + size * 0.12),
                ]
            )
        )
    elif kind == "flip_v":
        mid_y = size / 2
        painter.drawLine(QPointF(margin, mid_y), QPointF(size - margin, mid_y))
        painter.drawLine(QPointF(size / 2, margin), QPointF(size / 2, mid_y - size * 0.12))
        painter.drawPolyline(
            QPolygonF(
                [
                    QPointF(size / 2 - size * 0.12, margin + size * 0.08),
                    QPointF(size / 2, margin),
                    QPointF(size / 2 + size * 0.12, margin + size * 0.08),
                ]
            )
        )
        painter.drawLine(QPointF(size / 2, mid_y + size * 0.12), QPointF(size / 2, size - margin))
        painter.drawPolyline(
            QPolygonF(
                [
                    QPointF(size / 2 - size * 0.12, size - margin - size * 0.08),
                    QPointF(size / 2, size - margin),
                    QPointF(size / 2 + size * 0.12, size - margin - size * 0.08),
                ]
            )
        )
    elif kind in ("rotate_cw", "rotate_ccw"):
        rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
        if kind == "rotate_cw":
            painter.drawArc(rect, 45 * 16, 250 * 16)
            arrow_center = QPointF(rect.right(), rect.center().y() - rect.height() * 0.25)
            arrow = QPolygonF(
                [
                    QPointF(arrow_center.x() - size * 0.14, arrow_center.y() - size * 0.10),
                    QPointF(arrow_center.x() + size * 0.06, arrow_center.y()),
                    QPointF(arrow_center.x() - size * 0.14, arrow_center.y() + size * 0.10),
                ]
            )
        else:
            painter.drawArc(rect, (180 - 45) * 16, -250 * 16)
            arrow_center = QPointF(rect.left(), rect.center().y() - rect.height() * 0.25)
            arrow = QPolygonF(
                [
                    QPointF(arrow_center.x() + size * 0.14, arrow_center.y() - size * 0.10),
                    QPointF(arrow_center.x() - size * 0.06, arrow_center.y()),
                    QPointF(arrow_center.x() + size * 0.14, arrow_center.y() + size * 0.10),
                ]
            )
        painter.setBrush(QColor(color))
        painter.drawPolygon(arrow)
    elif kind == "crop":
        corner = size * 0.32
        # Four L-shaped corner brackets, evoking a crop-tool icon.
        painter.drawLine(QPointF(margin, margin), QPointF(margin + corner, margin))
        painter.drawLine(QPointF(margin, margin), QPointF(margin, margin + corner))
        painter.drawLine(QPointF(size - margin - corner, margin), QPointF(size - margin, margin))
        painter.drawLine(QPointF(size - margin, margin), QPointF(size - margin, margin + corner))
        painter.drawLine(QPointF(margin, size - margin - corner), QPointF(margin, size - margin))
        painter.drawLine(QPointF(margin, size - margin), QPointF(margin + corner, size - margin))
        painter.drawLine(QPointF(size - margin, size - margin - corner), QPointF(size - margin, size - margin))
        painter.drawLine(QPointF(size - margin - corner, size - margin), QPointF(size - margin, size - margin))

    painter.end()
    return QIcon(pixmap)


class _CropArea(QWidget):
    """Interactive crop-rectangle overlay drawn on top of a scaled image preview."""

    def __init__(self, pil_image: Image.Image, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._source_image = pil_image

        source_pixmap = pil_to_qpixmap(pil_image)
        self._display_pixmap = source_pixmap.scaled(
            CROP_DIALOG_MAX,
            CROP_DIALOG_MAX,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setFixedSize(self._display_pixmap.size())

        width, height = self._display_pixmap.width(), self._display_pixmap.height()
        inset_x, inset_y = width * 0.1, height * 0.1
        self._crop_rect = QRectF(inset_x, inset_y, width - 2 * inset_x, height - 2 * inset_y)

        self._drag_mode: Optional[str] = None
        self._drag_origin = QPointF()
        self._rect_origin = QRectF()

        self.setMouseTracking(True)

    def crop_box(self) -> tuple:
        """Map the on-screen crop rectangle back to source-image pixel coordinates."""
        scale_x = self._source_image.width / self._display_pixmap.width()
        scale_y = self._source_image.height / self._display_pixmap.height()

        left = int(round(self._crop_rect.left() * scale_x))
        top = int(round(self._crop_rect.top() * scale_y))
        right = int(round(self._crop_rect.right() * scale_x))
        bottom = int(round(self._crop_rect.bottom() * scale_y))

        left = max(0, min(left, self._source_image.width - 1))
        top = max(0, min(top, self._source_image.height - 1))
        right = max(left + 1, min(right, self._source_image.width))
        bottom = max(top + 1, min(bottom, self._source_image.height))
        return (left, top, right, bottom)

    def _handle_rects(self) -> dict:
        rect = self._crop_rect
        half = HANDLE_SIZE / 2
        return {
            "nw": QRectF(rect.left() - half, rect.top() - half, HANDLE_SIZE, HANDLE_SIZE),
            "ne": QRectF(rect.right() - half, rect.top() - half, HANDLE_SIZE, HANDLE_SIZE),
            "sw": QRectF(rect.left() - half, rect.bottom() - half, HANDLE_SIZE, HANDLE_SIZE),
            "se": QRectF(rect.right() - half, rect.bottom() - half, HANDLE_SIZE, HANDLE_SIZE),
        }

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        for name, rect in self._handle_rects().items():
            if rect.contains(pos):
                self._drag_mode = name
                self._drag_origin = pos
                self._rect_origin = QRectF(self._crop_rect)
                return
        if self._crop_rect.contains(pos):
            self._drag_mode = "move"
            self._drag_origin = pos
            self._rect_origin = QRectF(self._crop_rect)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_mode is None:
            return

        pos = event.position()
        delta = pos - self._drag_origin
        bounds = QRectF(0, 0, self.width(), self.height())
        rect = QRectF(self._rect_origin)

        if self._drag_mode == "move":
            rect.moveLeft(self._rect_origin.left() + delta.x())
            rect.moveTop(self._rect_origin.top() + delta.y())
            if rect.left() < bounds.left():
                rect.moveLeft(bounds.left())
            if rect.top() < bounds.top():
                rect.moveTop(bounds.top())
            if rect.right() > bounds.right():
                rect.moveRight(bounds.right())
            if rect.bottom() > bounds.bottom():
                rect.moveBottom(bounds.bottom())
        else:
            if "n" in self._drag_mode:
                new_top = min(self._rect_origin.top() + delta.y(), rect.bottom() - MIN_CROP_SIZE)
                rect.setTop(max(bounds.top(), new_top))
            if "s" in self._drag_mode:
                new_bottom = max(self._rect_origin.bottom() + delta.y(), rect.top() + MIN_CROP_SIZE)
                rect.setBottom(min(bounds.bottom(), new_bottom))
            if "w" in self._drag_mode:
                new_left = min(self._rect_origin.left() + delta.x(), rect.right() - MIN_CROP_SIZE)
                rect.setLeft(max(bounds.left(), new_left))
            if "e" in self._drag_mode:
                new_right = max(self._rect_origin.right() + delta.x(), rect.left() + MIN_CROP_SIZE)
                rect.setRight(min(bounds.right(), new_right))

        self._crop_rect = rect
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_mode = None

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.drawPixmap(0, 0, self._display_pixmap)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 140))
        painter.drawRect(QRectF(0, 0, self.width(), self.height()))

        crop_pixel_rect = self._crop_rect.toRect()
        painter.drawPixmap(crop_pixel_rect, self._display_pixmap, crop_pixel_rect)

        pen = QPen(QColor(ACCENT_COLOR))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self._crop_rect)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(ACCENT_COLOR))
        for handle_rect in self._handle_rects().values():
            painter.drawRect(handle_rect)

        painter.end()


class _CropDialog(QDialog):
    """Modal dialog that lets the user drag a resizable rectangle to crop an image."""

    def __init__(self, pil_image: Image.Image, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Crop Image")
        self.setModal(True)

        self._source_image = pil_image
        self._result_image: Optional[Image.Image] = None
        self._crop_area = _CropArea(pil_image)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        hint = QLabel("Drag the corner handles to resize, or drag inside the box to move it.")
        hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        layout.addWidget(hint)

        layout.addWidget(self._crop_area, 0, Qt.AlignmentFlag.AlignCenter)

        button_row = QHBoxLayout()
        button_row.addStretch(1)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_row.addWidget(cancel_button)

        confirm_button = QPushButton("Confirm")
        confirm_button.setObjectName("confirmButton")
        confirm_button.clicked.connect(self._on_confirm)
        button_row.addWidget(confirm_button)

        layout.addLayout(button_row)

        self.setStyleSheet(
            f"""
            QDialog {{ background-color: {SURFACE_COLOR}; }}
            QLabel {{ background-color: transparent; }}
            QPushButton {{
                background-color: {BUTTON_COLOR};
                color: {TEXT_PRIMARY};
                border: none;
                border-radius: 6px;
                padding: 8px 18px;
            }}
            QPushButton:hover {{ background-color: {BUTTON_HOVER_COLOR}; }}
            QPushButton#confirmButton {{
                background-color: {ACCENT_COLOR};
                color: #13151A;
                font-weight: 600;
            }}
            QPushButton#confirmButton:hover {{ background-color: {ACCENT_HOVER_COLOR}; }}
            """
        )

    def _on_confirm(self) -> None:
        self._result_image = self._source_image.crop(self._crop_area.crop_box())
        self.accept()

    def cropped_image(self) -> Optional[Image.Image]:
        return self._result_image


class ImageEditor(QWidget):
    """Non-destructive image editor: preview, crop/flip/rotate, apply/reset.

    Call ``set_image(pil_image)`` to load an image. Flip/rotate/crop
    operations mutate an in-memory working copy and update the preview
    immediately, but nothing is emitted until ``Apply Changes`` is pressed
    (at which point ``image_updated`` fires with the final image). A yellow
    dot + "Unsaved edits" label appear whenever there are pending changes
    that have not yet been applied.
    """

    #: Emitted with the edited PIL.Image.Image when "Apply Changes" is pressed.
    image_updated = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._original_image: Optional[Image.Image] = None
        self._working_image: Optional[Image.Image] = None
        self._dirty = False

        self._preview_label: Optional[QLabel] = None
        self._status_dot: Optional[QLabel] = None
        self._status_text: Optional[QLabel] = None
        self._apply_button: Optional[QPushButton] = None
        self._action_buttons: List[QPushButton] = []

        self._build_ui()
        self._update_controls_enabled()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self._preview_label = QLabel("No image selected")
        self._preview_label.setFixedSize(PREVIEW_SIZE, PREVIEW_SIZE)
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setWordWrap(True)
        self._preview_label.setStyleSheet(
            f"background-color: {SURFACE_COLOR}; color: {TEXT_SECONDARY};"
            f"border: 1px solid {BORDER_COLOR}; border-radius: 8px; font-size: 12px;"
        )
        layout.addWidget(self._preview_label, 0, Qt.AlignmentFlag.AlignHCenter)

        layout.addLayout(self._build_status_row())

        self.crop_button = QPushButton("  Crop")
        self.crop_button.setIcon(_build_icon("crop", TEXT_PRIMARY))
        self.crop_button.setIconSize(QSize(16, 16))
        self.crop_button.clicked.connect(self._open_crop_dialog)
        layout.addWidget(self.crop_button)
        self._action_buttons.append(self.crop_button)

        flip_row = QHBoxLayout()
        flip_row.setSpacing(8)
        self.flip_h_button = self._build_icon_button("flip_h", "Flip Horizontal", self._flip_horizontal)
        self.flip_v_button = self._build_icon_button("flip_v", "Flip Vertical", self._flip_vertical)
        flip_row.addWidget(self.flip_h_button)
        flip_row.addWidget(self.flip_v_button)
        layout.addLayout(flip_row)
        self._action_buttons.extend((self.flip_h_button, self.flip_v_button))

        rotate_row = QHBoxLayout()
        rotate_row.setSpacing(8)
        self.rotate_ccw_button = self._build_icon_button("rotate_ccw", "Rotate 90\u00b0 CCW", self._rotate_ccw)
        self.rotate_cw_button = self._build_icon_button("rotate_cw", "Rotate 90\u00b0 CW", self._rotate_cw)
        rotate_row.addWidget(self.rotate_ccw_button)
        rotate_row.addWidget(self.rotate_cw_button)
        layout.addLayout(rotate_row)
        self._action_buttons.extend((self.rotate_ccw_button, self.rotate_cw_button))

        self.reset_button = QPushButton("Reset to Original")
        self.reset_button.setObjectName("resetLink")
        self.reset_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_button.setStyleSheet(
            f"QPushButton#resetLink {{ background: transparent; color: {ACCENT_COLOR}; border: none;"
            f"text-decoration: underline; padding: 2px 0px; text-align: left; }}"
            f"QPushButton#resetLink:hover {{ color: {ACCENT_HOVER_COLOR}; }}"
        )
        self.reset_button.clicked.connect(self._reset_to_original)
        layout.addWidget(self.reset_button)
        self._action_buttons.append(self.reset_button)

        layout.addStretch(1)

        self._apply_button = QPushButton("Apply Changes")
        self._apply_button.setStyleSheet(
            f"QPushButton {{ background-color: {ACCENT_COLOR}; color: #13151A; font-weight: 600;"
            f"border: none; border-radius: 6px; padding: 10px 16px; }}"
            f"QPushButton:hover {{ background-color: {ACCENT_HOVER_COLOR}; }}"
            f"QPushButton:disabled {{ background-color: #4A4230; color: #8B8168; }}"
        )
        self._apply_button.clicked.connect(self._apply_changes)
        layout.addWidget(self._apply_button)
        self.apply_button = self._apply_button

    def _build_status_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        self._status_dot = QLabel()
        self._status_dot.setFixedSize(8, 8)
        row.addWidget(self._status_dot)

        self._status_text = QLabel("")
        self._status_text.setStyleSheet(f"color: {WARNING_COLOR}; font-size: 11px; background: transparent;")
        row.addWidget(self._status_text)
        row.addStretch(1)

        self._set_dirty(False)
        return row

    def _build_icon_button(self, icon_kind: str, tooltip: str, on_click: Callable[[], None]) -> QPushButton:
        button = QPushButton()
        button.setIcon(_build_icon(icon_kind, TEXT_PRIMARY))
        button.setIconSize(QSize(18, 18))
        button.setToolTip(tooltip)
        button.setFixedHeight(36)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button.clicked.connect(on_click)
        return button

    # -- Public API ----------------------------------------------------------

    def set_image(self, pil_image: Image.Image) -> None:
        """Load a new image into the editor, discarding any previous edits."""
        self._original_image = pil_image.copy()
        self._working_image = pil_image.copy()
        self._set_dirty(False)
        self._update_preview()
        self._update_controls_enabled()

    def current_image(self) -> Optional[Image.Image]:
        """Return the current working image (may include unapplied edits)."""
        return self._working_image

    def has_pending_changes(self) -> bool:
        """Whether there are edits that haven't been applied yet."""
        return self._dirty

    def clear(self) -> None:
        """Reset the editor to its empty ("no image loaded") state."""
        self._original_image = None
        self._working_image = None
        self._set_dirty(False)
        self._update_preview()
        self._update_controls_enabled()

    # -- Transform handlers ----------------------------------------------

    def _flip_horizontal(self) -> None:
        self._apply_transform(lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT))

    def _flip_vertical(self) -> None:
        self._apply_transform(lambda img: img.transpose(Image.Transpose.FLIP_TOP_BOTTOM))

    def _rotate_cw(self) -> None:
        self._apply_transform(lambda img: img.rotate(-90, expand=True))

    def _rotate_ccw(self) -> None:
        self._apply_transform(lambda img: img.rotate(90, expand=True))

    def _apply_transform(self, transform: Callable[[Image.Image], Image.Image]) -> None:
        if self._working_image is None:
            return
        try:
            transformed = transform(self._working_image)
        except Exception as exc:  # noqa: BLE001 - never let a transform crash the app
            self._show_transform_error(str(exc))
            return
        self._working_image = transformed
        self._set_dirty(True)
        self._update_preview()

    def _open_crop_dialog(self) -> None:
        if self._working_image is None:
            return
        try:
            dialog = _CropDialog(self._working_image, self)
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
        except Exception as exc:  # noqa: BLE001 - never let the crop dialog crash the app
            self._show_transform_error(str(exc))
            return
        if accepted:
            cropped = dialog.cropped_image()
            if cropped is not None:
                self._working_image = cropped
                self._set_dirty(True)
                self._update_preview()

    def _show_transform_error(self, message: str) -> None:
        if self._status_dot is not None and self._status_text is not None:
            self._status_dot.setStyleSheet(f"background-color: {DANGER_COLOR}; border-radius: 4px;")
            self._status_text.setStyleSheet(f"color: {DANGER_COLOR}; font-size: 11px; background: transparent;")
            self._status_text.setText(f"Edit failed: {message}")

    def _reset_to_original(self) -> None:
        if self._original_image is None:
            return
        self._working_image = self._original_image.copy()
        self._set_dirty(False)
        self._update_preview()

    def _apply_changes(self) -> None:
        if self._working_image is None:
            return
        self._set_dirty(False)
        self.image_updated.emit(self._working_image.copy())

    # -- Internal helpers --------------------------------------------------

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        if self._status_dot is not None and self._status_text is not None:
            if dirty:
                self._status_dot.setStyleSheet(f"background-color: {WARNING_COLOR}; border-radius: 4px;")
                self._status_text.setText("Unsaved edits")
            else:
                self._status_dot.setStyleSheet("background-color: transparent; border-radius: 4px;")
                self._status_text.setText("")
        if self._apply_button is not None:
            self._apply_button.setEnabled(dirty and self._working_image is not None)

    def _update_preview(self) -> None:
        if self._preview_label is None:
            return
        if self._working_image is None:
            self._preview_label.setPixmap(QPixmap())
            self._preview_label.setText("No image selected")
            return

        pixmap = pil_to_qpixmap(self._working_image)
        scaled = pixmap.scaled(
            PREVIEW_SIZE,
            PREVIEW_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview_label.setText("")
        self._preview_label.setPixmap(scaled)

    def _update_controls_enabled(self) -> None:
        enabled = self._working_image is not None
        for button in self._action_buttons:
            button.setEnabled(enabled)
        if self._apply_button is not None:
            self._apply_button.setEnabled(enabled and self._dirty)
