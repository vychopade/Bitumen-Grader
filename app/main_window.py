"""
Main application window for BitumenGrader.

Defines the top-level QMainWindow that hosts the left navigation sidebar
and switches between the app's pages (Import Images, Train Model, Grade
Images, Model Library) in a QStackedWidget. Also owns the app-wide
``active_model`` state (currently loaded model path + metadata), which is
handed to child pages that need it and re-broadcast via the
``active_model_changed`` signal whenever it changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QPointF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QGuiApplication,
    QIcon,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.pages.image_import_page import ImageImportPage
from app.pages.model_manager_page import ModelManagerPage
from app.pages.predict_page import PredictPage
from app.pages.train_page import TrainPage

# --------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------

BACKGROUND_COLOR = "#1A1C20"
SIDEBAR_COLOR = "#13151A"
SURFACE_COLOR = "#22252C"
ACCENT_COLOR = "#E8A838"
ACCENT_HOVER_COLOR = "#C98A20"
TEXT_PRIMARY = "#E8E9EC"
TEXT_SECONDARY = "#8B909A"
SEPARATOR_COLOR = "#2A2D34"
NAV_HOVER_COLOR = "#1D2027"

SIDEBAR_WIDTH = 220
WINDOW_MIN_WIDTH = 1100
WINDOW_MIN_HEIGHT = 720
WINDOW_TITLE_BASE = "BitumenGrader"

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_LOGO_PATH = _ASSETS_DIR / "logo.png"

_FALLBACK_FONT_FAMILIES = ("Segoe UI", "Helvetica Neue", "Arial", "Roboto", "Ubuntu")

#: (key, label, icon kind, page class) for each sidebar nav entry, in display order.
_NAV_ITEMS = [
    ("import", "Import Images", "import", ImageImportPage),
    ("train", "Train Model", "train", TrainPage),
    ("grade", "Grade Images", "grade", PredictPage),
    ("library", "Model Library", "library", ModelManagerPage),
]

#: Alt+<letter> keyboard shortcuts for sidebar nav items. These are always
#: bound (the sidebar is always visible), so pages must avoid reusing these
#: letters for their own Alt+key shortcuts.
NAV_SHORTCUT_LETTERS = {"import": "I", "train": "T", "grade": "G", "library": "L"}


def _resolve_font_family() -> str:
    """Return "Inter" if installed on the system, else a common sans-serif fallback."""
    families = set(QFontDatabase.families())
    if "Inter" in families:
        return "Inter"
    for fallback in _FALLBACK_FONT_FAMILIES:
        if fallback in families:
            return fallback
    return QFont().defaultFamily()


def _build_nav_icon(kind: str, color: str, size: int = 18) -> QIcon:
    """Draw a small flat line-icon for a sidebar nav item.

    Icons are rendered with QPainter primitives (no external image assets)
    so they stay crisp and match the current text color exactly.
    """
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

    margin = size * 0.18

    if kind == "import":
        # Upward arrow over a tray: an "upload images" glyph.
        mid_x = size / 2
        painter.drawLine(QPointF(mid_x, size - margin), QPointF(mid_x, margin))
        arrow = QPolygonF(
            [
                QPointF(mid_x - size * 0.22, margin + size * 0.28),
                QPointF(mid_x, margin),
                QPointF(mid_x + size * 0.22, margin + size * 0.28),
            ]
        )
        painter.drawPolyline(arrow)
        painter.drawLine(QPointF(margin, size - margin), QPointF(size - margin, size - margin))
    elif kind == "train":
        # Play glyph inside a circle: "run a training routine".
        center = QPointF(size / 2, size / 2)
        painter.drawEllipse(center, size / 2 - margin * 0.4, size / 2 - margin * 0.4)
        triangle = QPolygonF(
            [
                QPointF(size * 0.40, size * 0.32),
                QPointF(size * 0.40, size * 0.68),
                QPointF(size * 0.72, size * 0.50),
            ]
        )
        painter.setBrush(QColor(color))
        painter.drawPolygon(triangle)
    elif kind == "grade":
        # Magnifying glass: "inspect / grade a sample".
        radius = size * 0.28
        center = QPointF(size * 0.42, size * 0.42)
        painter.drawEllipse(center, radius, radius)
        handle_start = QPointF(center.x() + radius * 0.75, center.y() + radius * 0.75)
        handle_end = QPointF(size - margin * 0.6, size - margin * 0.6)
        painter.drawLine(handle_start, handle_end)
    elif kind == "library":
        # Stacked layers: "collection of saved models".
        for index, y in enumerate((size * 0.28, size * 0.5, size * 0.72)):
            inset = index * size * 0.06
            painter.drawLine(QPointF(margin + inset, y), QPointF(size - margin - inset, y))

    painter.end()
    return QIcon(pixmap)


class _Sidebar(QWidget):
    """Fixed-width left navigation sidebar: brand block, nav items, status pill."""

    nav_selected = pyqtSignal(int)

    def __init__(self, font_family: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(SIDEBAR_WIDTH)

        self._font_family = font_family
        self._nav_buttons: List[QPushButton] = []
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._status_dot: Optional[QLabel] = None
        self._status_label: Optional[QLabel] = None

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 24, 0, 16)
        layout.setSpacing(0)

        layout.addLayout(self._build_brand_block())
        layout.addSpacing(28)

        for index, (key, label, icon_kind, _page_cls) in enumerate(_NAV_ITEMS):
            button = self._build_nav_button(label, icon_kind, index, key)
            layout.addWidget(button)
            self._nav_buttons.append(button)

        layout.addStretch(1)
        layout.addWidget(self._build_status_pill())

    def _build_brand_block(self) -> QVBoxLayout:
        wrapper = QVBoxLayout()
        wrapper.setContentsMargins(20, 0, 20, 0)
        wrapper.setSpacing(10)

        logo_label = QLabel()
        logo_label.setFixedSize(40, 40)
        pixmap = QPixmap(str(_LOGO_PATH)) if _LOGO_PATH.exists() else QPixmap()
        if not pixmap.isNull() and min(pixmap.width(), pixmap.height()) >= 8:
            scaled = pixmap.scaled(
                40,
                40,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            logo_label.setPixmap(scaled)
        else:
            logo_label.setText("BG")
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            logo_font = QFont(self._font_family, 14)
            logo_font.setWeight(QFont.Weight.DemiBold)
            logo_label.setFont(logo_font)
            logo_label.setStyleSheet(
                f"background-color: {ACCENT_COLOR}; color: {SIDEBAR_COLOR}; border-radius: 8px;"
            )
        wrapper.addWidget(logo_label)

        name_label = QLabel("BitumenGrader")
        name_font = QFont(self._font_family, 15)
        name_font.setWeight(QFont.Weight.DemiBold)
        name_label.setFont(name_font)
        name_label.setStyleSheet(f"color: {TEXT_PRIMARY};")
        wrapper.addWidget(name_label)

        version_label = QLabel("v1.0")
        version_label.setFont(QFont(self._font_family, 11))
        version_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        wrapper.addWidget(version_label)

        return wrapper

    def _build_nav_button(self, label: str, icon_kind: str, index: int, key: str) -> QPushButton:
        button = QPushButton(f"  {label}")
        button.setObjectName("navItem")
        button.setCheckable(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setIcon(_build_nav_icon(icon_kind, TEXT_SECONDARY))
        button.setIconSize(QSize(18, 18))
        button.setFixedHeight(44)
        button.setFont(QFont(self._font_family, 13))
        button.clicked.connect(lambda _checked, i=index: self._on_nav_clicked(i))

        shortcut_letter = NAV_SHORTCUT_LETTERS.get(key)
        if shortcut_letter:
            button.setShortcut(QKeySequence(f"Alt+{shortcut_letter}"))
            button.setToolTip(f"{label} (Alt+{shortcut_letter})")

        self._button_group.addButton(button, index)
        return button

    def _build_status_pill(self) -> QWidget:
        pill_wrapper = QWidget()
        outer = QVBoxLayout(pill_wrapper)
        outer.setContentsMargins(20, 8, 20, 0)

        pill = QFrame()
        pill.setObjectName("statusPill")
        row = QHBoxLayout(pill)
        row.setContentsMargins(10, 7, 10, 7)
        row.setSpacing(8)

        self._status_dot = QLabel()
        self._status_dot.setFixedSize(8, 8)
        row.addWidget(self._status_dot)

        self._status_label = QLabel()
        self._status_label.setFont(QFont(self._font_family, 11))
        row.addWidget(self._status_label, 1)

        outer.addWidget(pill)
        self.set_active_model_label(None)
        return pill_wrapper

    def _on_nav_clicked(self, index: int) -> None:
        self.nav_selected.emit(index)

    def set_active_index(self, index: int) -> None:
        """Programmatically mark the nav button at ``index`` as the active/checked one."""
        if 0 <= index < len(self._nav_buttons):
            self._nav_buttons[index].setChecked(True)

    def set_active_model_label(self, display_name: Optional[str]) -> None:
        """Update the bottom status pill to show ``display_name`` or "No model loaded"."""
        if self._status_dot is None or self._status_label is None:
            return

        if display_name:
            self._status_dot.setStyleSheet(f"background-color: {ACCENT_COLOR}; border-radius: 4px;")
            metrics = QFontMetrics(self._status_label.font())
            elided = metrics.elidedText(display_name, Qt.TextElideMode.ElideRight, 130)
            self._status_label.setText(elided)
            self._status_label.setToolTip(display_name)
            self._status_label.setStyleSheet(f"color: {TEXT_PRIMARY};")
        else:
            self._status_dot.setStyleSheet(f"background-color: {TEXT_SECONDARY}; border-radius: 4px;")
            self._status_label.setText("No model loaded")
            self._status_label.setToolTip("")
            self._status_label.setStyleSheet(f"color: {TEXT_SECONDARY};")


class MainWindow(QMainWindow):
    """Top-level BitumenGrader window: sidebar navigation + stacked content pages."""

    #: Emitted whenever ``active_model`` changes, with the new value (a dict
    #: with "path"/"metadata" keys, or None if no model is loaded).
    active_model_changed = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        if _LOGO_PATH.exists():
            self.setWindowIcon(QIcon(str(_LOGO_PATH)))

        self._font_family = _resolve_font_family()
        self._apply_base_font()

        #: App-wide active model state: {"path": str, "metadata": dict} or None.
        self.active_model: Optional[Dict[str, Any]] = None

        self._pages: List[QWidget] = []
        self._stack = QStackedWidget()
        self._sidebar = _Sidebar(self._font_family)
        self._sidebar.nav_selected.connect(self._on_nav_selected)

        self._build_layout()
        self._update_window_title()

        self._sidebar.set_active_index(0)
        self._stack.setCurrentIndex(0)

        self.resize(1280, 800)
        self.center_on_screen()

    def _apply_base_font(self) -> None:
        body_font = QFont(self._font_family, 13)
        body_font.setWeight(QFont.Weight.Normal)
        self.setFont(body_font)

    def _build_layout(self) -> None:
        central = QWidget()
        central.setObjectName("centralWidget")

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._sidebar)

        separator = QFrame()
        separator.setObjectName("sidebarSeparator")
        separator.setFixedWidth(1)
        separator.setFrameShape(QFrame.Shape.NoFrame)
        root_layout.addWidget(separator)

        self._stack.setObjectName("contentStack")
        for _key, _label, _icon_kind, page_cls in _NAV_ITEMS:
            page = page_cls(main_window=self)
            self._stack.addWidget(page)
            self._pages.append(page)
        root_layout.addWidget(self._stack, 1)

        self.setCentralWidget(central)

    def _on_nav_selected(self, index: int) -> None:
        self._stack.setCurrentIndex(index)

    def set_active_model(self, model_path: Optional[str], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Set (or clear) the app-wide active model and notify sidebar + pages.

        Args:
            model_path: Path to the ``.pt`` checkpoint to make active, or
                ``None`` to clear the active model.
            metadata: The model's metadata dict (as returned by
                ``app.utils.model_io.load_model_metadata``), used to display
                a friendly name in the sidebar status pill.
        """
        if model_path is None:
            self.active_model = None
            self._sidebar.set_active_model_label(None)
        else:
            self.active_model = {"path": model_path, "metadata": metadata or {}}
            display_name = (metadata or {}).get("name") or Path(model_path).stem
            self._sidebar.set_active_model_label(display_name)

        self._update_window_title()
        self.active_model_changed.emit(self.active_model)

    def _update_window_title(self) -> None:
        """Refresh the window title to reflect the current active model (if any)."""
        if self.active_model:
            metadata = self.active_model.get("metadata") or {}
            name = metadata.get("name") or Path(self.active_model.get("path", "")).stem
            self.setWindowTitle(f"{WINDOW_TITLE_BASE} \u2014 {name}")
        else:
            self.setWindowTitle(f"{WINDOW_TITLE_BASE} \u2014 No Model Loaded")

    def center_on_screen(self) -> None:
        """Move this window to the center of its current (or primary) screen."""
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        available_geometry = screen.availableGeometry()
        frame_geometry = self.frameGeometry()
        frame_geometry.moveCenter(available_geometry.center())
        self.move(frame_geometry.topLeft())
