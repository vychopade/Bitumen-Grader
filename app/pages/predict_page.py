"""
Prediction / Grading page.

Provides the UI for running inference with a loaded model against new
bitumen sample images and displaying the predicted grade/classification
along with confidence scores and any relevant visualizations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from app.main_window import MainWindow


class PredictPage(QWidget):
    """Page for grading bitumen sample images with the active model.

    Currently a minimal placeholder so it can be wired into MainWindow's
    QStackedWidget; the image picker, ModelPredictor integration, and
    confidence/probability visualizations will be implemented in a later
    pass. Reads ``main_window.active_model`` (and listens for
    ``main_window.active_model_changed``) to know which saved model to run
    inference with.
    """

    def __init__(self, main_window: Optional["MainWindow"] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = main_window
        self._build_ui()

        if self.main_window is not None:
            self.main_window.active_model_changed.connect(self._on_active_model_changed)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(6)

        title = QLabel("Grade Images")
        title.setStyleSheet("color: #E8E9EC; font-size: 20px; font-weight: 600;")
        layout.addWidget(title)

        subtitle = QLabel("Run inference on new bitumen sample images with the active model.")
        subtitle.setStyleSheet("color: #8B909A; font-size: 13px;")
        layout.addWidget(subtitle)

        layout.addStretch(1)

    def _on_active_model_changed(self, active_model: Optional[dict]) -> None:
        """Placeholder hook for reacting to MainWindow's active model changing."""
        pass
