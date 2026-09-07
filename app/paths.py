"""Where the app keeps its stylesheet and logo, and where it writes saved models."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "BitumenGrader"


def _is_frozen() -> bool:
    """True for a packaged build, false when running from source."""
    return bool(getattr(sys, "frozen", False))


def _app_dir() -> Path:
    return Path(__file__).resolve().parent


def bundle_dir() -> Path:
    """Folder that ships with the app (stylesheet, logo, checkbox image)."""
    if _is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return _app_dir()


def user_data_dir() -> Path:
    """Writable folder for checkpoints. From source that is the project root so existing models stay put. Packaged builds use the OS app-data directory."""
    if not _is_frozen():
        return _app_dir().parent
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    if sys.platform == "win32":
        root = os.environ.get("APPDATA")
        base = Path(root) if root else Path.home() / "AppData" / "Roaming"
        return base / APP_NAME
    return Path.home() / f".{APP_NAME.lower()}"


ASSETS_DIR = bundle_dir() / "assets"
MODELS_DIR = user_data_dir() / "models"
