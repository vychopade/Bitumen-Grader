"""Small display helpers used by more than one page."""
from __future__ import annotations

from datetime import datetime
from typing import Optional


def format_created_at(created_at: Optional[str], fmt: str = "%b %d, %Y") -> str:
    """Format a model metadata ISO timestamp, or '' if missing/invalid."""
    if not created_at:
        return ""
    try:
        parsed = datetime.fromisoformat(created_at)
    except ValueError:
        return ""
    return parsed.strftime(fmt)
