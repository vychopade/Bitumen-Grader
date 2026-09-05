"""A dim overlay with a spinner and a short message for blocking work like loading a model or exporting. Parent it to a page and call show_message or hide_overlay."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QEvent, QObject, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import QWidget

from app.theme import ACCENT_COLOR, BACKGROUND_COLOR, TEXT_PRIMARY

OVERLAY_COLOR = QColor(BACKGROUND_COLOR)
OVERLAY_COLOR.setAlpha(0x99)

SPINNER_RADIUS = 16
SPINNER_ARC_SPAN = 270
TICK_INTERVAL_MS = 16
DEGREES_PER_TICK = 6


class LoadingOverlay(QWidget):
    """A dim overlay with a spinning arc and a status line. Parent it to the page you want to block, then call show_message while the work runs."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self._message = ""
        self._angle = 0

        self._timer = QTimer(self)
        self._timer.setInterval(TICK_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)

        if parent is not None:
            parent.installEventFilter(self)

        self.hide()

    def show_message(self, message: str = "Loading\u2026") -> None:
        """Covers the parent and starts the spinner. Pass the status text to show under the arc."""
        self._message = message
        if self.parentWidget() is not None:
            self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.show()
        self._timer.start()

    def set_message(self, message: str) -> None:
        """Changes the status text without restarting the spinner. Pass the new message."""
        self._message = message
        self.update()

    def hide_overlay(self) -> None:
        """Hides the overlay and stops the spinner."""
        self._timer.stop()
        self.hide()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if (
            watched is self.parentWidget()
            and event.type() == QEvent.Type.Resize
            and self.isVisible()
        ):
            self.setGeometry(self.parentWidget().rect())
        return super().eventFilter(watched, event)

    def _tick(self) -> None:
        self._angle = (self._angle + DEGREES_PER_TICK) % 360
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: D401  Qt calls this whenever the overlay needs a redraw
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), OVERLAY_COLOR)

        center_x, center_y = self.width() / 2, self.height() / 2

        painter.save()
        painter.translate(center_x, center_y)
        painter.rotate(self._angle)
        pen = QPen(QColor(ACCENT_COLOR))
        pen.setWidthF(3.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(
            QRectF(
                -SPINNER_RADIUS,
                -SPINNER_RADIUS,
                SPINNER_RADIUS * 2,
                SPINNER_RADIUS * 2,
            ),
            0,
            SPINNER_ARC_SPAN * 16,
        )
        painter.restore()

        if self._message:
            painter.setPen(QColor(TEXT_PRIMARY))
            font = painter.font()
            font.setPointSize(11)
            font.setWeight(font.Weight.DemiBold)
            painter.setFont(font)
            text_rect = QRectF(
                0, center_y + SPINNER_RADIUS + 14, self.width(), 30
            )
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                self._message,
            )

        painter.end()
