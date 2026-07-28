"""
Model Manager page.

Provides the UI for browsing, loading, and deleting saved models stored in
the models/ directory. Displays each saved model using the reusable
ModelCard component, in a responsive card grid with a name search filter,
and calls ``main_window.set_active_model(...)`` when the user loads one.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.components.model_card import ModelCard
from app.utils.model_io import list_saved_models
from app.utils.shortcuts import bind_page_shortcuts, shortcut_tooltip, unbind_page_shortcuts

if TYPE_CHECKING:
    from app.main_window import MainWindow

# --------------------------------------------------------------------------
# Design tokens (kept local so this page has no dependency on MainWindow)
# --------------------------------------------------------------------------

SURFACE_COLOR = "#22252C"
BORDER_COLOR = "#33373F"
ACCENT_COLOR = "#E8A838"
ACCENT_HOVER_COLOR = "#C98A20"
TEXT_PRIMARY = "#E8E9EC"
TEXT_SECONDARY = "#8B909A"
DANGER_COLOR = "#E5484D"

_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

CARD_MIN_WIDTH = 340
CARD_SPACING = 16
MIN_COLUMNS = 2

_EMPTY_STATE_SVG = b"""<?xml version="1.0" encoding="UTF-8"?>
<svg width="120" height="120" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <circle cx="60" cy="60" r="56" fill="none" stroke="#2E3138" stroke-width="2"/>
  <rect x="34" y="50" width="52" height="34" rx="6" fill="none" stroke="#8B909A" stroke-width="2.5"/>
  <line x1="34" y1="60" x2="86" y2="60" stroke="#8B909A" stroke-width="2.5"/>
  <circle cx="60" cy="72" r="6" fill="none" stroke="#E8A838" stroke-width="2.5"/>
  <path d="M60 50 L60 30" stroke="#8B909A" stroke-width="2.5" stroke-linecap="round"/>
  <path d="M50 38 L60 28 L70 38" fill="none" stroke="#8B909A" stroke-width="2.5"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""


class _CardsGrid(QWidget):
    """Container that lays out ModelCard widgets in a responsive grid.

    Recomputes the number of columns (minimum ``MIN_COLUMNS``) to fit as
    many ``CARD_MIN_WIDTH``-wide cards as possible on each resize, and
    re-flows the current card list whenever the column count changes.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._layout = QGridLayout(self)
        self._layout.setSpacing(CARD_SPACING)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._cards: List[QWidget] = []
        self._columns = MIN_COLUMNS

    def set_cards(self, cards: List[QWidget]) -> None:
        outgoing = [card for card in self._cards if card not in cards]
        for card in outgoing:
            card.setVisible(False)

        self._cards = cards
        self._apply_layout(self._compute_columns())

    def resizeEvent(self, event) -> None:  # noqa: D401 - Qt override
        super().resizeEvent(event)
        columns = self._compute_columns()
        if columns != self._columns:
            self._apply_layout(columns)

    def _compute_columns(self) -> int:
        width = self.width()
        if width <= 0:
            return MIN_COLUMNS
        return max(MIN_COLUMNS, (width + CARD_SPACING) // (CARD_MIN_WIDTH + CARD_SPACING))

    def _apply_layout(self, columns: int) -> None:
        self._columns = columns

        while self._layout.count():
            self._layout.takeAt(0)

        for index, card in enumerate(self._cards):
            row, col = divmod(index, columns)
            card.setVisible(True)
            self._layout.addWidget(card, row, col, Qt.AlignmentFlag.AlignTop)

        for col in range(columns):
            self._layout.setColumnStretch(col, 1)

        # Push a trailing stretch row so a scroll-area-resized container's
        # leftover vertical space collapses below the cards instead of being
        # distributed into (and stretching) the card rows themselves.
        num_rows = (len(self._cards) + columns - 1) // columns if self._cards else 0
        self._layout.setRowStretch(num_rows, 1)


class ModelManagerPage(QWidget):
    """Page for browsing, loading, and deleting saved models.

    Scans ``models/`` (via ``model_io.list_saved_models``) every time the
    page becomes visible, rendering one ``ModelCard`` per saved model.
    """

    def __init__(self, main_window: Optional["MainWindow"] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = main_window
        self._cards: List[ModelCard] = []

        self._search_edit: Optional[QLineEdit] = None
        self._load_error_label: Optional[QLabel] = None
        self._empty_state: Optional[QWidget] = None
        self._no_match_label: Optional[QLabel] = None
        self._scroll: Optional[QScrollArea] = None
        self._cards_grid: Optional[_CardsGrid] = None
        self._shortcut_bindings: List[tuple] = []
        self._train_link_button: Optional[QPushButton] = None
        self._tab_order_applied = False

        self._build_ui()

        if self.main_window is not None:
            self.main_window.active_model_changed.connect(self._on_active_model_changed)

        self._reload_models()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 24)
        root.setSpacing(16)

        root.addLayout(self._build_header())
        root.addWidget(self._build_search_bar())

        self._load_error_label = QLabel("")
        self._load_error_label.setWordWrap(True)
        self._load_error_label.setStyleSheet(f"color: {DANGER_COLOR}; font-size: 12px; background: transparent;")
        self._load_error_label.setVisible(False)
        root.addWidget(self._load_error_label)

        body = QWidget()
        body.setStyleSheet("background: transparent;")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self._empty_state = self._build_empty_state()
        body_layout.addWidget(self._empty_state)

        self._no_match_label = QLabel("No models match your search.")
        self._no_match_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_match_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px; padding: 40px;")
        self._no_match_label.setVisible(False)
        body_layout.addWidget(self._no_match_label)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self._cards_grid = _CardsGrid()
        self._scroll.setWidget(self._cards_grid)
        body_layout.addWidget(self._scroll)

        root.addWidget(body, 1)

    def _build_header(self) -> QVBoxLayout:
        header = QVBoxLayout()
        header.setSpacing(4)

        title = QLabel("Model Library")
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 20px; font-weight: 600;")
        header.addWidget(title)

        subtitle = QLabel("Load, review, and manage your saved models.")
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px;")
        header.addWidget(subtitle)

        return header

    def _build_search_bar(self) -> QLineEdit:
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search models by name\u2026")
        self._search_edit.setFixedHeight(38)
        self._search_edit.setStyleSheet(
            f"QLineEdit {{ background-color: {SURFACE_COLOR}; color: {TEXT_PRIMARY};"
            f"border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 6px 12px; font-size: 13px; }}"
            f"QLineEdit:focus {{ border: 1px solid {ACCENT_COLOR}; }}"
        )
        self._search_edit.textChanged.connect(self._on_search_text_changed)
        return self._search_edit

    def _build_empty_state(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 60, 0, 60)
        layout.setSpacing(16)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        illustration = QSvgWidget()
        illustration.load(QByteArray(_EMPTY_STATE_SVG))
        illustration.setFixedSize(120, 120)
        layout.addWidget(illustration, 0, Qt.AlignmentFlag.AlignHCenter)

        message = QLabel("No models yet. Train your first model to get started.")
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message.setWordWrap(True)
        message.setFixedWidth(320)
        message.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 14px; background: transparent;")
        layout.addWidget(message, 0, Qt.AlignmentFlag.AlignHCenter)

        train_link = QPushButton("Train a Model")
        train_link.setObjectName("trainModelLink")
        train_link.setCursor(Qt.CursorShape.PointingHandCursor)
        train_link.setStyleSheet(
            f"QPushButton#trainModelLink {{ background: transparent; color: {ACCENT_COLOR}; border: none;"
            f"font-size: 13px; font-weight: 600; text-decoration: underline; padding: 4px; }}"
            f"QPushButton#trainModelLink:hover {{ color: {ACCENT_HOVER_COLOR}; }}"
        )
        train_link.setToolTip(shortcut_tooltip("Go to Train Model", "M"))
        train_link.clicked.connect(self._navigate_to_train_page)
        layout.addWidget(train_link, 0, Qt.AlignmentFlag.AlignHCenter)
        self._shortcut_bindings.append((train_link, "M"))
        self._train_link_button = train_link

        return container

    # -- Data loading / filtering --------------------------------------------

    def showEvent(self, event) -> None:  # noqa: D401 - Qt override
        super().showEvent(event)
        self._reload_models()
        bind_page_shortcuts(self._shortcut_bindings)
        if not self._tab_order_applied and self._search_edit is not None and self._train_link_button is not None:
            QWidget.setTabOrder(self._search_edit, self._train_link_button)
            self._tab_order_applied = True

    def hideEvent(self, event) -> None:  # noqa: D401 - Qt override
        super().hideEvent(event)
        unbind_page_shortcuts(self._shortcut_bindings)

    def _reload_models(self) -> None:
        try:
            metadata_list = list_saved_models(_MODELS_DIR)
        except OSError as exc:
            self._load_error_label.setText(f"Could not read the models folder: {exc}")
            self._load_error_label.setVisible(True)
            metadata_list = []
        else:
            self._load_error_label.setVisible(False)

        for card in self._cards:
            card.deleteLater()
        self._cards = []

        for metadata in metadata_list:
            card = ModelCard(metadata, is_active=self._is_active_model(metadata))
            card.load_requested.connect(self._on_load_requested)
            card.delete_requested.connect(self._on_delete_requested)
            self._cards.append(card)

        self._apply_search_filter()

    def _is_active_model(self, metadata: Dict[str, Any]) -> bool:
        active_model = getattr(self.main_window, "active_model", None) if self.main_window else None
        if not active_model:
            return False
        return active_model.get("path") == metadata.get("model_path")

    def _on_active_model_changed(self, _active_model: Optional[Dict[str, Any]]) -> None:
        for card in self._cards:
            card.set_active(self._is_active_model(card.metadata))

    def _on_search_text_changed(self, _text: str) -> None:
        self._apply_search_filter()

    def _apply_search_filter(self) -> None:
        query = self._search_edit.text().strip().lower()
        if query:
            visible = [card for card in self._cards if query in (card.metadata.get("name") or "").lower()]
        else:
            visible = list(self._cards)

        self._cards_grid.set_cards(visible)

        has_models = bool(self._cards)
        has_visible = bool(visible)

        self._empty_state.setVisible(not has_models)
        self._no_match_label.setVisible(has_models and not has_visible)
        self._scroll.setVisible(has_visible)

    # -- Card actions --------------------------------------------------------

    def _on_load_requested(self, metadata: Dict[str, Any]) -> None:
        if self.main_window is None:
            return
        model_path = metadata.get("model_path")
        if not model_path:
            return
        self.main_window.set_active_model(model_path, metadata)
        self._navigate_to_grade_page()

    def _on_delete_requested(self, metadata: Dict[str, Any]) -> None:
        model_path = metadata.get("model_path")
        metadata_path = metadata.get("metadata_path")

        failures: List[str] = []
        for path_str in (model_path, metadata_path):
            if path_str:
                try:
                    Path(path_str).unlink(missing_ok=True)
                except OSError as exc:
                    failures.append(f"{Path(path_str).name}: {exc}")

        if self.main_window is not None:
            active_model = getattr(self.main_window, "active_model", None)
            if active_model and active_model.get("path") == model_path:
                self.main_window.set_active_model(None)

        self._reload_models()

        if failures:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Delete Incomplete")
            box.setText("Some model files could not be deleted.")
            box.setInformativeText("\n".join(failures))
            box.setStandardButtons(QMessageBox.StandardButton.Ok)
            box.exec()

    # -- Navigation (local imports below avoid a circular import, since
    #    TrainPage/PredictPage import ModelManagerPage for their own
    #    "go to library" navigation). ------------------------------------

    def _navigate_to_train_page(self) -> None:
        from app.pages.train_page import TrainPage

        self._navigate_to_page_of_type(TrainPage)

    def _navigate_to_grade_page(self) -> None:
        from app.pages.predict_page import PredictPage

        self._navigate_to_page_of_type(PredictPage)

    def _navigate_to_page_of_type(self, page_type: type) -> None:
        if self.main_window is None:
            return
        stack = getattr(self.main_window, "_stack", None)
        sidebar = getattr(self.main_window, "_sidebar", None)
        pages = getattr(self.main_window, "_pages", [])

        target_index = next((i for i, page in enumerate(pages) if isinstance(page, page_type)), None)
        if target_index is None or stack is None:
            return

        stack.setCurrentIndex(target_index)
        if sidebar is not None and hasattr(sidebar, "set_active_index"):
            sidebar.set_active_index(target_index)
