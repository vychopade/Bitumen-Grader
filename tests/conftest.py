"""
Pytest configuration for the tests/ directory.

Ensures the project root (which contains the ``app`` package) is importable
regardless of the current working directory pytest was invoked from.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
