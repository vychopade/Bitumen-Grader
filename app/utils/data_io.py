"""Reads the labels table, Excel if the path ends in xlsx or xls, otherwise CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

_EXCEL_EXTENSIONS = (".xlsx", ".xls")


def read_labels_file(path: Union[str, Path]) -> pd.DataFrame:
    """Open a labels table. Excel paths go through read_excel; everything else is CSV."""
    suffix = Path(path).suffix.lower()
    if suffix in _EXCEL_EXTENSIONS:
        return pd.read_excel(path)
    return pd.read_csv(path)
