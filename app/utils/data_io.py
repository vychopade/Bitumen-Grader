"""
Tabular label-file loading helpers.

Training labels (filenames plus Water/Solids/Bitumen/Pan values) can be
supplied as either a CSV/plain-text file or an Excel workbook; both the
"Load CSV File" step on the Train page and ``RegressionDataset`` read
through ``read_labels_file`` so the two stay in sync on which formats are
accepted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

#: File extensions accepted for training label files, in the order surfaced
#: to the user (CSV/plain-text first, since that is the common case).
SUPPORTED_EXTENSIONS = (".csv", ".txt", ".xlsx", ".xls")

_EXCEL_EXTENSIONS = (".xlsx", ".xls")


def read_labels_file(path: Union[str, Path]) -> pd.DataFrame:
    """Read a training label file into a ``DataFrame``, dispatching on extension.

    ``.xlsx``/``.xls`` files are read as Excel workbooks (first sheet);
    everything else (``.csv``, ``.txt``) is read as delimited text.

    Args:
        path: Path to the label file.

    Returns:
        The parsed ``DataFrame``.

    Raises:
        Exception: Whatever ``pandas`` raises for a malformed/unreadable
            file (e.g. ``ParserError``, ``ValueError``, ``OSError``) --
            callers are expected to catch broadly and show a friendly
            message, since the underlying failure modes vary by format.
    """
    suffix = Path(path).suffix.lower()
    if suffix in _EXCEL_EXTENSIONS:
        return pd.read_excel(path)
    return pd.read_csv(path)
