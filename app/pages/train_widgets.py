"""Train-page widgets: CSV drop zone and dataset summary cards."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.constants import IMAGE_EXTENSIONS, LABEL_EXTENSIONS, OUTPUT_NAMES
from app.ml.recipe import IMAGE_SIZE
from app.theme import (
    ACCENT_COLOR,
    DANGER_COLOR,
    SUCCESS_COLOR,
    SURFACE_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    card_qss,
    drop_zone_qss,
    ghost_button_qss,
)
from app.utils.files import (
    drop_has_accepted_files,
    dropped_local_paths,
    pick_image_folder,
    pick_labels_file,
)

MAX_UNMATCHED_PREVIEW = 200


def _drop_zone_label(text: str, *, primary: bool) -> QLabel:
    """Centered wrapping label that shrinks to the drop-zone width."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setMinimumWidth(0)
    label.setSizePolicy(
        QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
    )
    color = TEXT_PRIMARY if primary else TEXT_SECONDARY
    size = 13 if primary else 11
    label.setStyleSheet(
        f"color: {color}; font-size: {size}px; background: transparent;"
    )
    return label


class _DropZoneFrame(QFrame):
    """Dashed target whose height follows wrapped title/subtitle text."""

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        layout = self.layout()
        if layout is None:
            return super().heightForWidth(width)
        return layout.heightForWidth(width)


class _CsvDropZone(_DropZoneFrame):
    """Dashed drop area for the training CSV, plus a Browse button."""

    file_selected = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("csvDropZone")
        self.setAcceptDrops(True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        self._build_ui()
        self._apply_style(active=False)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        title = _drop_zone_label("Drop a labels file", primary=True)
        layout.addWidget(title)

        subtitle = _drop_zone_label(
            ".csv, .txt, .xlsx, or .xls", primary=False
        )
        layout.addWidget(subtitle)

        self.browse_button = QPushButton("Choose file")
        self.browse_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_button.setStyleSheet(ghost_button_qss())
        self.browse_button.clicked.connect(self._browse_file)
        layout.addWidget(self.browse_button, 0, Qt.AlignmentFlag.AlignHCenter)

    def _apply_style(self, active: bool) -> None:
        self.setStyleSheet(drop_zone_qss("csvDropZone", active=active))

    def _browse_file(self) -> None:
        path = pick_labels_file(self)
        if path:
            self.file_selected.emit(path)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if drop_has_accepted_files(event, LABEL_EXTENSIONS):
            event.acceptProposedAction()
            self._apply_style(active=True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self._apply_style(active=False)

    def dropEvent(self, event: QDropEvent) -> None:
        self._apply_style(active=False)
        paths = dropped_local_paths(event, LABEL_EXTENSIONS)
        if paths:
            self.file_selected.emit(paths[0])
            event.acceptProposedAction()
        else:
            event.ignore()


class _FolderDropZone(_DropZoneFrame):
    """Drop a photo folder (or photos) here, or choose a folder."""

    folder_selected = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("folderDropZone")
        self.setAcceptDrops(True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        self._build_ui()
        self._apply_style(active=False)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        title = _drop_zone_label("Drop a photo folder", primary=True)
        layout.addWidget(title)

        subtitle = _drop_zone_label(
            "JPG, PNG, or TIF — nested folders included", primary=False
        )
        layout.addWidget(subtitle)

        self.browse_button = QPushButton("Choose folder")
        self.browse_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_button.setStyleSheet(ghost_button_qss())
        self.browse_button.clicked.connect(self._browse_folder)
        layout.addWidget(self.browse_button, 0, Qt.AlignmentFlag.AlignHCenter)

    def _apply_style(self, active: bool) -> None:
        self.setStyleSheet(drop_zone_qss("folderDropZone", active=active))

    def _browse_folder(self) -> None:
        folder = pick_image_folder(self)
        if folder:
            self.folder_selected.emit(folder)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if drop_has_accepted_files(event, IMAGE_EXTENSIONS, recurse_dirs=True):
            event.acceptProposedAction()
            self._apply_style(active=True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self._apply_style(active=False)

    def dropEvent(self, event: QDropEvent) -> None:
        self._apply_style(active=False)
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            event.ignore()
            return
        for url in mime.urls():
            if url.isLocalFile() and Path(url.toLocalFile()).is_dir():
                self.folder_selected.emit(url.toLocalFile())
                event.acceptProposedAction()
                return
        paths = dropped_local_paths(event, IMAGE_EXTENSIONS, recurse_dirs=True)
        if paths:
            self.folder_selected.emit(str(Path(paths[0]).parent))
            event.acceptProposedAction()
        else:
            event.ignore()


class _MatchSummaryCard(QFrame):
    """Match count plus expandable unmatched / invalid-row lists."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet(card_qss(inset=True))

        self._unmatched_count = 0
        self._invalid_count = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        self._summary_label = QLabel("")
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        self._invalid_summary_label = QLabel("")
        self._invalid_summary_label.setWordWrap(True)
        self._invalid_summary_label.setStyleSheet(
            f"color: {DANGER_COLOR}; font-size: 12px; font-weight: 600;"
            f" background: transparent;"
        )
        self._invalid_summary_label.setVisible(False)
        layout.addWidget(self._invalid_summary_label)

        self._tip_label = QLabel("")
        self._tip_label.setWordWrap(True)
        self._tip_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px;"
            f" background: transparent;"
        )
        self._tip_label.setVisible(False)
        layout.addWidget(self._tip_label)

        self.toggle_button = QPushButton("")
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.setStyleSheet(
            f"QPushButton {{ background: transparent;"
            f" color: {TEXT_SECONDARY}; border: none;"
            f"text-decoration: underline; font-size: 11px; text-align: left;"
            f" padding: 0px; }}"
            f"QPushButton:hover {{ color: {TEXT_PRIMARY}; }}"
        )
        self.toggle_button.clicked.connect(self._toggle_unmatched)
        self.toggle_button.setVisible(False)
        layout.addWidget(self.toggle_button, 0, Qt.AlignmentFlag.AlignLeft)

        self._unmatched_label = QLabel("")
        self._unmatched_label.setWordWrap(True)
        self._unmatched_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px;"
            f" background: transparent;"
        )
        self._unmatched_label.setVisible(False)
        layout.addWidget(self._unmatched_label)

        self._invalid_toggle_button = QPushButton("")
        self._invalid_toggle_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        self._invalid_toggle_button.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {DANGER_COLOR};"
            f" border: none;"
            f"text-decoration: underline; font-size: 11px; text-align: left;"
            f" padding: 0px; }}"
            f"QPushButton:hover {{ color: {TEXT_PRIMARY}; }}"
        )
        self._invalid_toggle_button.clicked.connect(self._toggle_invalid)
        self._invalid_toggle_button.setVisible(False)
        layout.addWidget(
            self._invalid_toggle_button, 0, Qt.AlignmentFlag.AlignLeft
        )

        self._invalid_rows_label = QLabel("")
        self._invalid_rows_label.setWordWrap(True)
        self._invalid_rows_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px;"
            f" background: transparent;"
        )
        self._invalid_rows_label.setVisible(False)
        layout.addWidget(self._invalid_rows_label)

    def update_summary(self, match_summary: Dict) -> None:
        matched = match_summary["matched"]
        total = match_summary["total_csv_rows"]
        rate_pct = match_summary["match_rate"] * 100

        self._summary_label.setText(f"{matched} of {total} images matched")
        if rate_pct > 90:
            color = SUCCESS_COLOR
        elif rate_pct >= 50:
            color = ACCENT_COLOR
        else:
            color = DANGER_COLOR
        self._summary_label.setStyleSheet(
            f"color: {color}; font-size: 15px; font-weight: 700;"
            f" background: transparent;"
        )

        if rate_pct < 50:
            self._tip_label.setText(
                "Check that CSV filenames match your image files. Matching "
                "works "
                "with or without the file extension."
            )
            self._tip_label.setVisible(True)
        else:
            self._tip_label.setVisible(False)

        unmatched_files: List[str] = list(
            match_summary.get("unmatched_files", [])
        )
        self._unmatched_count = len(unmatched_files)
        if unmatched_files:
            preview = unmatched_files[:MAX_UNMATCHED_PREVIEW]
            text = ", ".join(preview)
            if len(unmatched_files) > len(preview):
                text += f", and {len(unmatched_files) - len(preview)} more"
            self._unmatched_label.setText(text)
            self.toggle_button.setVisible(True)
        else:
            self.toggle_button.setVisible(False)

        self._unmatched_label.setVisible(False)
        self.toggle_button.setText(
            f"Show unmatched filenames ({self._unmatched_count}) \u25be"
        )

        invalid_rows: List[Dict] = list(match_summary.get("invalid_rows", []))
        self._invalid_count = len(invalid_rows)
        if invalid_rows:
            word = "row" if self._invalid_count == 1 else "rows"
            self._invalid_summary_label.setText(
                f"{self._invalid_count} {word} skipped \u2014 couldn't read "
                f"Water/Solids/Bitumen/Pan (batch)."
            )
            self._invalid_summary_label.setVisible(True)

            preview = invalid_rows[:MAX_UNMATCHED_PREVIEW]
            text = "\n".join(
                f"\u201c{entry['image']}\u201d \u2014 {entry['reason']}"
                for entry in preview
            )
            if len(invalid_rows) > len(preview):
                text += f"\n\u2026and {len(invalid_rows) - len(preview)} more"
            self._invalid_rows_label.setText(text)
            self._invalid_toggle_button.setVisible(True)
        else:
            self._invalid_summary_label.setVisible(False)
            self._invalid_toggle_button.setVisible(False)

        self._invalid_rows_label.setVisible(False)
        self._invalid_toggle_button.setText(
            f"Show invalid rows ({self._invalid_count}) \u25be"
        )

    def _toggle_invalid(self) -> None:
        showing = not self._invalid_rows_label.isVisible()
        self._invalid_rows_label.setVisible(showing)
        verb = "Hide" if showing else "Show"
        arrow = "\u25b4" if showing else "\u25be"
        self._invalid_toggle_button.setText(
            f"{verb} invalid rows ({self._invalid_count}) {arrow}"
        )

    def _toggle_unmatched(self) -> None:
        showing = not self._unmatched_label.isVisible()
        self._unmatched_label.setVisible(showing)
        verb = "Hide" if showing else "Show"
        arrow = "\u25b4" if showing else "\u25be"
        self.toggle_button.setText(
            f"{verb} unmatched filenames ({self._unmatched_count}) {arrow}"
        )


class _DatasetSummaryCard(QFrame):
    """Sample counts, output ranges, and pan-grade bars."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet(card_qss(inset=True))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        self._counts_label = QLabel("")
        self._counts_label.setWordWrap(True)
        self._counts_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; background: transparent;"
        )
        layout.addWidget(self._counts_label)

        ranges_title = QLabel("Output ranges:")
        ranges_title.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; font-weight: 600;"
            f" background: transparent;"
        )
        layout.addWidget(ranges_title)

        self._ranges_label = QLabel("")
        self._ranges_label.setWordWrap(True)
        self._ranges_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; background: transparent;"
        )
        layout.addWidget(self._ranges_label)

        pan_title = QLabel("Batches:")
        pan_title.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; font-weight: 600;"
            f" background: transparent;"
        )
        layout.addWidget(pan_title)

        self._pan_container = QVBoxLayout()
        self._pan_container.setSpacing(6)
        layout.addLayout(self._pan_container)

        self._campaigns_title = QLabel("Held-out campaigns:")
        self._campaigns_title.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; font-weight: 600;"
            f" background: transparent;"
        )
        self._campaigns_title.setVisible(False)
        layout.addWidget(self._campaigns_title)

        self._campaigns_label = QLabel("")
        self._campaigns_label.setWordWrap(True)
        self._campaigns_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; background: transparent;"
        )
        self._campaigns_label.setVisible(False)
        layout.addWidget(self._campaigns_label)

    def update_summary(
        self,
        total: int,
        train_count: int,
        val_count: int,
        test_count: int,
        val_fraction: float,
        test_fraction: float,
        output_ranges: Dict[str, Dict[str, float]],
        pan_distribution: Dict[int, int],
        split_mode: str = "random",
        split_campaigns: Optional[Dict[str, List[str]]] = None,
        split_fallback_reason: Optional[str] = None,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        train_pct = round((1 - val_fraction - test_fraction) * 100)
        val_pct = round(val_fraction * 100)
        test_pct = round(test_fraction * 100)
        self._counts_label.setText(
            f"Matched: {total}\n"
            f"Train: {train_count} ({train_pct}%)\n"
            f"Validation: {val_count} ({val_pct}%)\n"
            f"Test: {test_count} ({test_pct}%)\n"
            f"Photos resized to {image_size}\u00d7{image_size} before training"
        )

        lines = []
        for label in OUTPUT_NAMES:
            values = output_ranges.get(
                label, {"min": 0.0, "max": 0.0, "mean": 0.0}
            )
            lines.append(
                f"{label}: {values['min']:.2f} \u2013 {values['max']:.2f} % "
                f"(mean {values['mean']:.2f})"
            )
        self._ranges_label.setText("\n".join(lines))

        while self._pan_container.count():
            item = self._pan_container.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        max_count = max(pan_distribution.values()) if pan_distribution else 1
        for grade in sorted(pan_distribution.keys()):
            count = pan_distribution[grade]
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            label = QLabel(f"Batch {grade}: {count}")
            label.setMinimumWidth(96)
            label.setStyleSheet(
                f"color: {TEXT_SECONDARY}; font-size: 11px;"
                f" background: transparent;"
            )
            row_layout.addWidget(label)

            bar = QProgressBar()
            bar.setRange(0, max_count)
            bar.setValue(count)
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            bar.setStyleSheet(
                f"QProgressBar {{ background-color: {SURFACE_COLOR};"
                f" border-radius: 5px; border: none; }}"
                f"QProgressBar::chunk {{ background-color: {ACCENT_COLOR};"
                f" border-radius: 5px; }}"
            )
            row_layout.addWidget(bar, 1)

            self._pan_container.addWidget(row_widget)

        show_campaigns = split_mode == "experiment" or bool(
            split_fallback_reason
        )
        if show_campaigns:
            campaigns = split_campaigns or {}
            lines = []
            for key, title in (
                ("train", "Train"),
                ("val", "Val"),
                ("test", "Test"),
            ):
                names = campaigns.get(key) or []
                lines.append(f"{title}: {', '.join(names) if names else '—'}")
            if split_fallback_reason:
                lines.append(split_fallback_reason)
            self._campaigns_label.setText("\n".join(lines))
            self._campaigns_title.setVisible(True)
            self._campaigns_label.setVisible(True)
        else:
            self._campaigns_title.setVisible(False)
            self._campaigns_label.setVisible(False)
