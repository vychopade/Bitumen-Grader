"""Start the BitumenGrader PyQt6 app."""
from __future__ import annotations

import sys
import traceback
from types import TracebackType
from typing import Optional, Type

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

from app.paths import ASSETS_DIR, MODELS_DIR

APP_NAME = "BitumenGrader"
APP_VERSION = "1.0.0"
ORGANIZATION_NAME = "BitumenGrader"
ORGANIZATION_DOMAIN = "bitumengrader.local"

_STYLESHEET_PATH = ASSETS_DIR / "style.qss"


def _ensure_models_directory() -> None:
    """Make sure models/ exists so a fresh checkout can run."""
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def _load_stylesheet() -> str:
    """Load style.qss and fill in {ASSETS_DIR} for absolute asset urls.

    Qt resolves relative url()s against the cwd, not the .qss file, so we
    use absolute paths so it works no matter where you launch from.
    """
    if not _STYLESHEET_PATH.exists():
        return ""
    try:
        text = _STYLESHEET_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""
    return text.replace("{ASSETS_DIR}", ASSETS_DIR.as_posix())


def _install_exception_hook(app: QApplication) -> None:
    """Show unhandled errors in a dialog instead of silently dying.

    Still calls the previous hook afterward. App keeps running so one bad
    page error doesn't kill the whole tool.
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
        except Exception:  # noqa: BLE001 - don't let the error handler crash too
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

    # Import after high-DPI attrs, before any QWidget (Qt startup order).
    from app.main_window import MainWindow

    window = MainWindow()
    window.show()
    window.center_on_screen()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
