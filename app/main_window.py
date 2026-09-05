"""The main BitumenGrader window. Sidebar on the left, Train, Grade, and Models stacked on the right. This class owns the loaded model and tells the pages when it changes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import (
    QFont,
    QFontDatabase,
    QFontMetrics,
    QGuiApplication,
    QIcon,
    QKeySequence,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.ml.predictor import RegressionPredictor
from app.pages import GradePage, ModelsPage, TrainPage
from app.paths import ASSETS_DIR
from app.theme import TEXT_PRIMARY
from app.utils.model_io import format_r2_headline

SIDEBAR_WIDTH = 148
WINDOW_MIN_WIDTH = 1100
WINDOW_MIN_HEIGHT = 720
WINDOW_TITLE_BASE = "BitumenGrader"

_LOGO_PATH = ASSETS_DIR / "logo.png"

_FALLBACK_FONT_FAMILIES = (
    "Segoe UI",
    "Helvetica Neue",
    ".AppleSystemUIFont",
    "Arial",
    "Ubuntu",
)

# Each sidebar button: key, label, and the page class to open.
_NAV_ITEMS = [
    ("train", "Train", TrainPage),
    ("grade", "Grade", GradePage),
    ("models", "Models", ModelsPage),
]

# Alt plus this letter jumps to that sidebar page.
NAV_SHORTCUT_LETTERS = {"train": "T", "grade": "G", "models": "M"}


def _resolve_font_family() -> str:
    """Picks a plain system font so the UI does not look like a marketing site. Inter was too polished for a lab tool. You get a family name that is actually installed."""
    families = set(QFontDatabase.families())
    for fallback in _FALLBACK_FONT_FAMILIES:
        if fallback in families:
            return fallback
    return QFont().defaultFamily()


class _Sidebar(QWidget):
    """Left sidebar with the page names and a link that shows which model is loaded."""

    nav_selected = pyqtSignal(int)
    models_requested = pyqtSignal()

    def __init__(self, font_family: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(SIDEBAR_WIDTH)

        self._font_family = font_family
        self._nav_buttons: List[QPushButton] = []
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._status_button: Optional[QPushButton] = None

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 16, 0, 8)
        layout.setSpacing(0)

        for index, (key, label, _page_cls) in enumerate(_NAV_ITEMS):
            button = self._build_nav_button(label, index, key)
            layout.addWidget(button)
            self._nav_buttons.append(button)

        layout.addStretch(1)

        self._status_button = QPushButton("No model")
        self._status_button.setObjectName("statusLink")
        self._status_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._status_button.setFont(QFont(self._font_family, 11))
        self._status_button.setToolTip("Open saved models")
        self._status_button.clicked.connect(self.models_requested.emit)
        layout.addWidget(self._status_button)

    def _build_nav_button(
        self, label: str, index: int, key: str
    ) -> QPushButton:
        button = QPushButton(label)
        button.setObjectName("navItem")
        button.setCheckable(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFixedHeight(36)
        button.setFont(QFont(self._font_family, 13))
        button.clicked.connect(
            lambda _checked, i=index: self._on_nav_clicked(i)
        )

        shortcut_letter = NAV_SHORTCUT_LETTERS.get(key)
        if shortcut_letter:
            button.setShortcut(QKeySequence(f"Alt+{shortcut_letter}"))
            button.setToolTip(f"{label}  Alt+{shortcut_letter}")

        self._button_group.addButton(button, index)
        return button

    def _on_nav_clicked(self, index: int) -> None:
        self.nav_selected.emit(index)

    def set_active_index(self, index: int) -> None:
        """Checks the sidebar button for this page index so it looks selected. Pass the 0-based index."""
        if 0 <= index < len(self._nav_buttons):
            self._nav_buttons[index].setChecked(True)

    def set_active_model_label(
        self, display_name: Optional[str], r2_headline: str = ""
    ) -> None:
        """Shows the loaded model name at the bottom of the sidebar, or No model if nothing is loaded. Pass the display name and an optional R squared headline for the tooltip."""
        if self._status_button is None:
            return
        if display_name:
            metrics = QFontMetrics(self._status_button.font())
            elided = metrics.elidedText(
                display_name, Qt.TextElideMode.ElideRight, SIDEBAR_WIDTH - 28
            )
            self._status_button.setText(elided)
            tooltip = f"{display_name} — click to change"
            if r2_headline:
                tooltip = f"{display_name}  ·  {r2_headline} — click to change"
            self._status_button.setToolTip(tooltip)
            self._status_button.setStyleSheet(f"color: {TEXT_PRIMARY};")
        else:
            self._status_button.setText("No model")
            self._status_button.setToolTip("Open saved models")
            self._status_button.setStyleSheet("")


class MainWindow(QMainWindow):
    """The outer window: sidebar plus the stacked Train, Grade, and Models pages."""

    # Fires when the loaded model changes. The payload is the active_model dict, or None.
    active_model_changed = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        if _LOGO_PATH.exists():
            self.setWindowIcon(QIcon(str(_LOGO_PATH)))

        self._font_family = _resolve_font_family()
        self._apply_base_font()

        # Loaded model dict with path, metadata, and predictor, or None.
        self.active_model: Optional[Dict[str, Any]] = None

        self._pages: List[QWidget] = []
        self._stack = QStackedWidget()
        self._sidebar = _Sidebar(self._font_family)
        self._sidebar.nav_selected.connect(self._on_nav_selected)
        self._sidebar.models_requested.connect(
            lambda: self.navigate_to("models")
        )

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
        for _key, _label, page_cls in _NAV_ITEMS:
            page = page_cls(main_window=self)
            self._stack.addWidget(page)
            self._pages.append(page)
        root_layout.addWidget(self._stack, 1)

        self.setCentralWidget(central)

    def _on_nav_selected(self, index: int) -> None:
        self.navigate_to_index(index)

    def navigate_to_index(self, index: int) -> None:
        """Switches to the sidebar page at this index and lights up that nav button. Pass 0, 1, or 2."""
        if index < 0 or index >= self._stack.count():
            return
        self._stack.setCurrentIndex(index)
        self._sidebar.set_active_index(index)

    def navigate_to(self, page_key: str) -> None:
        """Opens a page by name. Pass train, grade, or models."""
        keys = [item[0] for item in _NAV_ITEMS]
        try:
            self.navigate_to_index(keys.index(page_key))
        except ValueError:
            return

    def page_for(self, page_key: str) -> Optional[QWidget]:
        """Looks up a stacked page by name. Pass train, grade, or models. You get the widget, or None if the key is unknown."""
        keys = [item[0] for item in _NAV_ITEMS]
        try:
            index = keys.index(page_key)
        except ValueError:
            return None
        if index < 0 or index >= len(self._pages):
            return None
        return self._pages[index]

    def set_active_model(
        self,
        model_path: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Loads a checkpoint into a predictor and tells the sidebar and pages. Pass the .pt path and its metadata, or None to clear. If the file will not load we show a dialog and leave the previous model alone."""
        if model_path is None:
            self.active_model = None
            self._sidebar.set_active_model_label(None)
            self._update_window_title()
            self.active_model_changed.emit(self.active_model)
            return

        metadata = metadata or {}
        try:
            predictor = RegressionPredictor(model_path, metadata)
        except Exception as exc:  # noqa: BLE001
            # Show why the checkpoint failed so the previous model stays loaded.
            QMessageBox.critical(
                self,
                "Model Load Failed",
                (
                    "Couldn't load \u201c"
                    f"{metadata.get('name') or Path(model_path).stem}"
                    f"\u201d:\n{exc}"
                ),
            )
            return

        self.active_model = {
            "path": model_path,
            "metadata": metadata,
            "predictor": predictor,
        }
        display_name = metadata.get("name") or Path(model_path).stem
        self._sidebar.set_active_model_label(
            display_name, format_r2_headline(metadata)
        )

        self._update_window_title()
        self.active_model_changed.emit(self.active_model)

    def _update_window_title(self) -> None:
        """Puts the loaded model name in the window title, or says no model is loaded."""
        if self.active_model:
            metadata = self.active_model.get("metadata") or {}
            name = (
                metadata.get("name")
                or Path(self.active_model.get("path", "")).stem
            )
            self.setWindowTitle(f"{WINDOW_TITLE_BASE} \u2014 {name}")
        else:
            self.setWindowTitle(f"{WINDOW_TITLE_BASE} \u2014 No Model Loaded")

    def center_on_screen(self) -> None:
        """Moves this window onto the center of the current screen, or the primary screen if we cannot tell."""
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        available_geometry = screen.availableGeometry()
        frame_geometry = self.frameGeometry()
        frame_geometry.moveCenter(available_geometry.center())
        self.move(frame_geometry.topLeft())
