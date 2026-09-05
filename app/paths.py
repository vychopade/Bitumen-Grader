"""Where the app finds its stylesheet and where it writes saved models."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "BitumenGrader"


def _is_frozen() -> bool:
    """True when this is a packaged build, not a source checkout."""
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """Folder that holds read-only files we ship with the app, like the stylesheet and logo. You do not pass anything in. You get a Path back."""
    if _is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    """Folder we are allowed to write checkpoints into. From source that is the project root so existing models stay visible. A packaged build uses the OS app-data directory instead."""
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
