"""
Entry point for the BitumenGrader desktop application.

Responsible for bootstrapping the PyQt6 QApplication, applying any global
application settings (styling, icons, high-DPI policy, etc.), instantiating
the MainWindow, and starting the Qt event loop.
"""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from app.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("BitumenGrader")
    app.setOrganizationName("BitumenGrader")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
