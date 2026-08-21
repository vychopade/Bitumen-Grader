"""
Main window for BitumenGrader.

Sidebar + stacked pages (Import, Train, Grade, Model Library). Owns the
app-wide ``active_model`` (path, metadata, ready ``RegressionPredictor``)
and broadcasts ``active_model_changed`` when it changes.
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
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.ml.predictor import RegressionPredictor
from app.pages import GradePage, ImportPage, LibraryPage, TrainPage
from app.paths import ASSETS_DIR
from app.theme import ACCENT_COLOR, TEXT_PRIMARY, TEXT_SECONDARY

SIDEBAR_WIDTH = 220
WINDOW_MIN_WIDTH = 1100
WINDOW_MIN_HEIGHT = 720
WINDOW_TITLE_BASE = "BitumenGrader"

_LOGO_PATH = ASSETS_DIR / "logo.png"

_FALLBACK_FONT_FAMILIES = ("Segoe UI", "Helvetica Neue", "Arial", "Roboto", "Ubuntu")

#: (key, label, icon kind, page class) for each sidebar item.
_NAV_ITEMS = [
    ("import", "Import Images", "import", ImportPage),
    ("train", "Train Model", "train", TrainPage),
    ("grade", "Grade Images", "grade", GradePage),
    ("library", "Model Library", "library", LibraryPage),
]

#: Alt+letter shortcuts for sidebar nav. Pages should not reuse these letters.
NAV_SHORTCUT_LETTERS = {"import": "I", "train": "T", "grade": "G", "library": "L"}


def _resolve_font_family() -> str:
    """Prefer Inter if installed; otherwise a common sans-serif."""
    families = set(QFontDatabase.families())
    if "Inter" in families:
        return "Inter"
    for fallback in _FALLBACK_FONT_FAMILIES:
        if fallback in families:
            return fallback
    return QFont().defaultFamily()


def _build_nav_icon(kind: str, color: str, size: int = 18) -> QIcon:
    """Small flat sidebar icon drawn with QPainter (no image assets)."""
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
        # Upload arrow over a tray.
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
        # Play triangle in a circle.
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
        # Magnifying glass.
        radius = size * 0.28
        center = QPointF(size * 0.42, size * 0.42)
        painter.drawEllipse(center, radius, radius)
        handle_start = QPointF(center.x() + radius * 0.75, center.y() + radius * 0.75)
        handle_end = QPointF(size - margin * 0.6, size - margin * 0.6)
        painter.drawLine(handle_start, handle_end)
    elif kind == "library":
        # Stacked lines = saved models.
        for index, y in enumerate((size * 0.28, size * 0.5, size * 0.72)):
            inset = index * size * 0.06
            painter.drawLine(QPointF(margin + inset, y), QPointF(size - margin - inset, y))

    painter.end()
    return QIcon(pixmap)


class _Sidebar(QWidget):
    """Left nav: brand area, page links, status pill."""

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
        self._status_secondary_label: Optional[QLabel] = None

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 24, 0, 16)
        layout.setSpacing(0)

        for index, (key, label, icon_kind, _page_cls) in enumerate(_NAV_ITEMS):
            button = self._build_nav_button(label, icon_kind, index, key)
            layout.addWidget(button)
            self._nav_buttons.append(button)

        layout.addStretch(1)
        layout.addWidget(self._build_status_pill())

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
        pill_layout = QVBoxLayout(pill)
        pill_layout.setContentsMargins(10, 7, 10, 7)
        pill_layout.setSpacing(3)

        name_row = QHBoxLayout()
        name_row.setContentsMargins(0, 0, 0, 0)
        name_row.setSpacing(8)

        self._status_dot = QLabel()
        self._status_dot.setFixedSize(8, 8)
        name_row.addWidget(self._status_dot)

        self._status_label = QLabel()
        self._status_label.setFont(QFont(self._font_family, 11))
        name_row.addWidget(self._status_label, 1)
        pill_layout.addLayout(name_row)

        self._status_secondary_label = QLabel()
        self._status_secondary_label.setFont(QFont(self._font_family, 9))
        self._status_secondary_label.setWordWrap(True)
        self._status_secondary_label.setStyleSheet(f"color: {TEXT_SECONDARY}; padding-left: 16px;")
        pill_layout.addWidget(self._status_secondary_label)

        outer.addWidget(pill)
        self.set_active_model_label(None, None)
        return pill_wrapper

    def _on_nav_clicked(self, index: int) -> None:
        self.nav_selected.emit(index)

    def set_active_index(self, index: int) -> None:
        """Mark the nav button at ``index`` as checked."""
        if 0 <= index < len(self._nav_buttons):
            self._nav_buttons[index].setChecked(True)

    def set_active_model_label(
        self, display_name: Optional[str], best_val_mae: Optional[Dict[str, float]] = None
    ) -> None:
        """Update the bottom status pill.

        With a model: name + best val MAE. Without: "No model loaded".
        """
        if self._status_dot is None or self._status_label is None or self._status_secondary_label is None:
            return

        if display_name:
            self._status_dot.setStyleSheet(f"background-color: {ACCENT_COLOR}; border-radius: 4px;")
            metrics = QFontMetrics(self._status_label.font())
            elided = metrics.elidedText(display_name, Qt.TextElideMode.ElideRight, 130)
            self._status_label.setText(elided)
            self._status_label.setToolTip(display_name)
            self._status_label.setStyleSheet(f"color: {TEXT_PRIMARY};")

            mae = best_val_mae or {}
            mae_text = (
                f"Water \u00b1{mae.get('Water', 0.0):.2f}  "
                f"Solids \u00b1{mae.get('Solids', 0.0):.2f}  "
                f"Bitumen \u00b1{mae.get('Bitumen', 0.0):.2f}"
            )
            self._status_secondary_label.setText(mae_text)
            self._status_secondary_label.setToolTip(mae_text)
            self._status_secondary_label.setVisible(True)
        else:
            self._status_dot.setStyleSheet(f"background-color: {ACCENT_COLOR}; border-radius: 4px;")
            self._status_label.setText("No model loaded")
            self._status_label.setToolTip("")
            self._status_label.setStyleSheet(f"color: {ACCENT_COLOR}; font-weight: 600;")
            self._status_secondary_label.setVisible(False)


class MainWindow(QMainWindow):
    """BitumenGrader shell: sidebar + stacked pages."""

    #: Fired when ``active_model`` changes (dict with path/metadata, or None).
    active_model_changed = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        if _LOGO_PATH.exists():
            self.setWindowIcon(QIcon(str(_LOGO_PATH)))

        self._font_family = _resolve_font_family()
        self._apply_base_font()

        #: Active model: {"path", "metadata", "predictor"} or None.
        self.active_model: Optional[Dict[str, Any]] = None
        #: Path payloads from Import → Train / Grade. Consumed when those pages show.
        self.training_images: Optional[List[Dict[str, Any]]] = None
        self.grading_images: Optional[List[Dict[str, Any]]] = None

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
        self.navigate_to_index(index)

    def navigate_to_index(self, index: int) -> None:
        """Show the sidebar page at ``index`` and mark that nav item active."""
        if index < 0 or index >= self._stack.count():
            return
        self._stack.setCurrentIndex(index)
        self._sidebar.set_active_index(index)

    def navigate_to(self, page_key: str) -> None:
        """Show a sidebar page by key (``import``, ``train``, ``grade``, ``library``)."""
        keys = [item[0] for item in _NAV_ITEMS]
        try:
            self.navigate_to_index(keys.index(page_key))
        except ValueError:
            return

    def page_for(self, page_key: str) -> Optional[QWidget]:
        """Return the stacked page for ``page_key``, or ``None``."""
        keys = [item[0] for item in _NAV_ITEMS]
        try:
            index = keys.index(page_key)
        except ValueError:
            return None
        if index < 0 or index >= len(self._pages):
            return None
        return self._pages[index]

    def set_active_model(self, model_path: Optional[str], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Set or clear the active model; update sidebar + pages.

        Loads the checkpoint into a ``RegressionPredictor`` up front so Grade
        and Library can rely on ``active_model`` already having a usable predictor.

        Args:
            model_path: Path to the ``.pt`` file, or ``None`` to clear.
            metadata: Model metadata (name, MAE, ``output_stats``, etc.).

        Load failures show a dialog and leave the previous model unchanged.
        """
        if model_path is None:
            self.active_model = None
            self._sidebar.set_active_model_label(None)
            self._update_window_title()
            self.active_model_changed.emit(self.active_model)
            return

        metadata = metadata or {}
        try:
            predictor = RegressionPredictor(model_path, metadata)
        except Exception as exc:  # noqa: BLE001 - surface any load failure to the user
            QMessageBox.critical(
                self,
                "Model Load Failed",
                f"Couldn't load \u201c{metadata.get('name') or Path(model_path).stem}\u201d:\n{exc}",
            )
            return

        self.active_model = {"path": model_path, "metadata": metadata, "predictor": predictor}
        display_name = metadata.get("name") or Path(model_path).stem
        self._sidebar.set_active_model_label(display_name, metadata.get("best_val_mae"))

        self._update_window_title()
        self.active_model_changed.emit(self.active_model)

    def _update_window_title(self) -> None:
        """Window title includes the active model name, if any."""
        if self.active_model:
            metadata = self.active_model.get("metadata") or {}
            name = metadata.get("name") or Path(self.active_model.get("path", "")).stem
            self.setWindowTitle(f"{WINDOW_TITLE_BASE} \u2014 {name}")
        else:
            self.setWindowTitle(f"{WINDOW_TITLE_BASE} \u2014 No Model Loaded")

    def center_on_screen(self) -> None:
        """Center this window on the current (or primary) screen."""
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        available_geometry = screen.availableGeometry()
        frame_geometry = self.frameGeometry()
        frame_geometry.moveCenter(available_geometry.center())
        self.move(frame_geometry.topLeft())
