"""Load training labels from CSV/txt or Excel."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

# CSV/txt first — that's what most people use.
SUPPORTED_EXTENSIONS = (".csv", ".txt", ".xlsx", ".xls")

_EXCEL_EXTENSIONS = (".xlsx", ".xls")


def read_labels_file(path: Union[str, Path]) -> pd.DataFrame:
    """Read a labels file; Excel if .xlsx/.xls, otherwise CSV."""
    suffix = Path(path).suffix.lower()
    if suffix in _EXCEL_EXTENSIONS:
        return pd.read_excel(path)
    return pd.read_csv(path)
