"""
Model Manager page.

Provides the UI for browsing, loading, renaming, and deleting saved models
stored in the models/ directory. Displays each saved model using the
reusable model_card component and allows selecting the active model used
for prediction.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from app.main_window import MainWindow


class ModelManagerPage(QWidget):
    """Page for browsing and managing saved models.

    Currently a minimal placeholder so it can be wired into MainWindow's
    QStackedWidget; listing saved models (via ``model_io.list_saved_models``)
    as ModelCard widgets and calling ``main_window.set_active_model(...)``
    when the user selects one will be implemented in a later pass.
    """

    def __init__(self, main_window: Optional["MainWindow"] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = main_window
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(6)

        title = QLabel("Model Library")
        title.setStyleSheet("color: #E8E9EC; font-size: 20px; font-weight: 600;")
        layout.addWidget(title)

        subtitle = QLabel("Browse, load, and manage your saved models.")
        subtitle.setStyleSheet("color: #8B909A; font-size: 13px;")
        layout.addWidget(subtitle)

        layout.addStretch(1)
