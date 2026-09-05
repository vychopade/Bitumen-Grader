"""Project directories used by more than one module."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "BitumenGrader"


def _is_frozen() -> bool:
    """True when running as a PyInstaller bundle, not from source."""
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """Read-only files shipped with the app (stylesheet, logo)."""
    if _is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    """Writable location for saved models.

    A source checkout uses the project folder so existing checkpoints stay
    visible. A packaged app uses the OS application-data directory.
    """
    if not _is_frozen():
        return Path(__file__).resolve().parent.parent
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    if sys.platform == "win32":
        root = os.environ.get("APPDATA")
        base = Path(root) if root else Path.home() / "AppData" / "Roaming"
        return base / APP_NAME
    return Path.home() / f".{APP_NAME.lower()}"


PROJECT_ROOT = bundle_dir()
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = user_data_dir() / "models"
