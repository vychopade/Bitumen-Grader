"""Opens the BitumenGrader desktop window."""

from __future__ import annotations

import sys
import traceback
from types import TracebackType
from typing import Optional, Type

from PyQt6.QtWidgets import QApplication, QMessageBox

from app.paths import APP_NAME, ASSETS_DIR, MODELS_DIR

APP_VERSION = "1.0.0"
ORGANIZATION_NAME = "BitumenGrader"
ORGANIZATION_DOMAIN = "bitumengrader.local"

_STYLESHEET_PATH = ASSETS_DIR / "style.qss"


def _ensure_models_directory() -> None:
    """Creates the models folder if it is missing so a fresh checkout can save checkpoints."""
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def _load_stylesheet() -> str:
    """Reads style.qss and replaces {ASSETS_DIR} with a real path. Qt looks up url() against the working directory, so baking in an absolute path means the logo and checkbox image still load no matter where you launch from."""
    if not _STYLESHEET_PATH.exists():
        return ""
    try:
        text = _STYLESHEET_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""
    return text.replace("{ASSETS_DIR}", ASSETS_DIR.as_posix())


def _install_exception_hook() -> None:
    """Pops up a dialog for unexpected errors instead of dying silently. The old exception hook still runs, and the window stays open so one bad page does not kill the whole tool."""
    previous_hook = sys.excepthook

    def _handle_exception(
        exc_type: Type[BaseException],
        exc_value: BaseException,
        exc_traceback: Optional[TracebackType],
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            previous_hook(exc_type, exc_value, exc_traceback)
            return

        details = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
        sys.stderr.write(details)

        try:
            box = QMessageBox()
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle(f"{APP_NAME} \u2014 Unexpected Error")
            box.setText(
                "An unexpected error occurred. You can continue using the "
                "app, but some state may be inconsistent."
            )
            box.setInformativeText(str(exc_value) or exc_type.__name__)
            box.setDetailedText(details)
            box.setStandardButtons(QMessageBox.StandardButton.Ok)
            box.exec()
        except Exception:  # noqa: BLE001
            # If the dialog itself fails, swallow it so we do not crash twice.
            pass

    sys.excepthook = _handle_exception


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(ORGANIZATION_NAME)
    app.setOrganizationDomain(ORGANIZATION_DOMAIN)

    stylesheet = _load_stylesheet()
    if stylesheet:
        app.setStyleSheet(stylesheet)

    _install_exception_hook()
    _ensure_models_directory()

    # Widgets need a QApplication already running, so import the window here.
    from app.main_window import MainWindow

    window = MainWindow()
    window.show()
    window.center_on_screen()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
