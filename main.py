"""
Entry point for the BitumenGrader desktop application.

Responsible for bootstrapping the PyQt6 QApplication, applying any global
application settings (styling, icons, high-DPI policy, etc.), instantiating
the MainWindow, and starting the Qt event loop. Also installs a global
exception hook so unhandled errors surface as a dialog instead of silently
crashing the app.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from types import TracebackType
from typing import Optional, Type

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

APP_NAME = "BitumenGrader"
APP_VERSION = "1.0.0"
ORGANIZATION_NAME = "BitumenGrader"
ORGANIZATION_DOMAIN = "bitumengrader.local"

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_STYLESHEET_PATH = _ASSETS_DIR / "style.qss"
_MODELS_DIR = Path(__file__).resolve().parent / "models"


def _ensure_models_directory() -> None:
    """Create the top-level models/ directory on first launch if missing.

    The Model Library, Train, and Predict pages all expect this directory to
    exist once the window is up (e.g. to scan it for saved models); creating
    it here means a fresh checkout works immediately instead of only after
    the first model is trained.
    """
    try:
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def _load_stylesheet() -> str:
    """Read assets/style.qss, substituting the {ASSETS_DIR} asset-path token.

    Absolute paths are used for asset url()s referenced from the stylesheet
    (e.g. the checkbox checkmark glyph) because Qt resolves relative url()s
    in an application-wide style sheet against the process's current working
    directory, not the .qss file's location -- an absolute path keeps this
    working regardless of where the app is launched from.
    """
    if not _STYLESHEET_PATH.exists():
        return ""
    try:
        text = _STYLESHEET_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""
    return text.replace("{ASSETS_DIR}", _ASSETS_DIR.as_posix())


def _install_exception_hook(app: QApplication) -> None:
    """Route unhandled exceptions to a QMessageBox instead of a hard crash.

    Keeps the previous excepthook so KeyboardInterrupt/SystemExit and any
    outer tooling still see the exception after the user dismisses the
    dialog (the app continues running rather than force-exiting, since a
    single unexpected error in one page shouldn't take down the whole tool).
    """
    previous_hook = sys.excepthook

    def _handle_exception(
        exc_type: Type[BaseException],
        exc_value: BaseException,
        exc_traceback: Optional[TracebackType],
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            previous_hook(exc_type, exc_value, exc_traceback)
            return

        details = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print(details, file=sys.stderr)

        try:
            box = QMessageBox()
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle(f"{APP_NAME} \u2014 Unexpected Error")
            box.setText("An unexpected error occurred. You can continue using the app, but some state may be inconsistent.")
            box.setInformativeText(str(exc_value) or exc_type.__name__)
            box.setDetailedText(details)
            box.setStandardButtons(QMessageBox.StandardButton.Ok)
            box.exec()
        except Exception:  # noqa: BLE001 - never let the error handler itself crash the app
            pass

    sys.excepthook = _handle_exception


def main() -> int:
    if hasattr(Qt.ApplicationAttribute, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(ORGANIZATION_NAME)
    app.setOrganizationDomain(ORGANIZATION_DOMAIN)

    stylesheet = _load_stylesheet()
    if stylesheet:
        app.setStyleSheet(stylesheet)

    _install_exception_hook(app)
    _ensure_models_directory()

    # Imported after high-DPI attributes are set and before any QWidget is
    # constructed, per Qt's recommended startup order.
    from app.main_window import MainWindow

    window = MainWindow()
    window.show()
    window.center_on_screen()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
