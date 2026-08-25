import math
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from app.constants import IMAGE_EXTENSIONS
from app.ml.recipe import CLS_BINS, TEST_FRACTION, VAL_FRACTION
from app.utils.data_io import read_labels_file
from app.utils.media import collect_images
from app.utils.image_utils import IMAGE_SIZE, build_eval_transforms, build_train_transforms, prepare_image, standardize_to_model_size


# Optional CSV columns used as an experiment/campaign id (Case 2 split).
CAMPAIGN_COLUMN_CANDIDATES = ("Campaign", "Experiment", "Run")
# IMG_0032_N12.jpg → N12; IMG_0027_11.jpg → N11
_CAMPAIGN_N_RE = re.compile(r"_N(\d+)", re.IGNORECASE)
_CAMPAIGN_TRAILING_RE = re.compile(r"_(\d+)$")
_PARENTHETICAL_RE = re.compile(r"\s*\(.*\)\s*$")


def parse_campaign_id(filename: str, row=None) -> str:
    """Campaign / flotation-run id for experiment-holdout splits.

    Prefers an explicit Campaign/Experiment/Run column, then ``_N12`` in the
    filename, then a trailing ``_11``. Returns ``unknown`` if none match.
    """
    if row is not None:
        for column in CAMPAIGN_COLUMN_CANDIDATES:
            has_column = column in row.index if hasattr(row, "index") else column in row
            if not has_column:
                continue
            raw = row[column]
            if raw is None:
                continue
            try:
                if isinstance(raw, float) and math.isnan(raw):
                    continue
            except TypeError:
                pass
            text = str(raw).strip()
            if text and text.lower() not in {"nan", "none"}:
                return text

    stem = Path(str(filename)).stem
    stem = _PARENTHETICAL_RE.sub("", stem).strip()
    match = _CAMPAIGN_N_RE.search(stem)
    if match:
        return f"N{match.group(1)}"
    match = _CAMPAIGN_TRAILING_RE.search(stem)
    if match:
        return f"N{match.group(1)}"
    return "unknown"


class RegressionDataset(Dataset):
    """Images + (Water, Solids, Bitumen) targets from a labels file.

    Matches CSV/Excel rows to files in ``image_dir``, shuffles with a fixed
    seed, and splits into train/val/test. Norm stats always come from train
    only, even when this instance is the val or test split.

    ``split_mode``:
        ``random`` — Case 1: shuffled image-level split (interpolation).
        ``experiment`` — Case 2: hold out entire flotation campaigns.
    """

    EXPECTED_COLUMNS = ["Image", "Pan", "Water", "Solids", "Bitumen"]
    EXTENSION_CANDIDATES = IMAGE_EXTENSIONS

    def __init__(
        self,
        csv_path,
        image_dir,
        split="train",
        val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION,
        normalise=False,
        seed=42,
        split_mode="random",
        image_size=IMAGE_SIZE,
        legacy_crop=False,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
        if split_mode not in {"random", "experiment"}:
            raise ValueError(f"split_mode must be 'random' or 'experiment', got {split_mode!r}")
        if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1.0:
            raise ValueError(
                f"val_fraction ({val_fraction}) + test_fraction ({test_fraction}) "
                "must be >= 0 and leave room for a non-empty train split"
            )

        self.csv_path = csv_path
        self.image_dir = Path(image_dir)
        self.split = split
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.normalise = normalise
        self.seed = seed
        self.split_mode = split_mode
        self.image_size = int(image_size)
        self.legacy_crop = bool(legacy_crop)
        self.split_fallback_reason = None
        self.split_campaigns = {"train": [], "val": [], "test": []}

        df = read_labels_file(csv_path)
        missing_columns = [column for column in self.EXPECTED_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV is missing expected columns: {missing_columns}")

        by_name, by_stem = self._index_images(self.image_dir)

        self.matched = []
        self.unmatched = []
        # Image found, but Water/Solids/Bitumen/Pan wasn't a number.
        # Each entry: {"image": str, "reason": str}.
        self.invalid_rows = []

        for _, row in df.iterrows():
            image_value = str(row["Image"]).strip()
            matched_rel = None

            if image_value in by_name:
                matched_rel = by_name[image_value]
            elif Path(image_value).name in by_name:
                matched_rel = by_name[Path(image_value).name]
            elif Path(image_value).stem in by_stem:
                matched_rel = by_stem[Path(image_value).stem]
            else:
                for extension in self.EXTENSION_CANDIDATES:
                    candidate = image_value + extension
                    if candidate in by_name:
                        matched_rel = by_name[candidate]
                        break
                    candidate_name = Path(image_value).name + extension
                    if candidate_name in by_name:
                        matched_rel = by_name[candidate_name]
                        break

            if matched_rel is None:
                self.unmatched.append(image_value)
                continue

            try:
                water = self._parse_float(row["Water"], "Water")
                solids = self._parse_float(row["Solids"], "Solids")
                bitumen = self._parse_float(row["Bitumen"], "Bitumen")
                pan = self._parse_pan(row["Pan"])
            except ValueError as exc:
                # Bad cell (typo, blank, etc.) — skip this row, keep going.
                self.invalid_rows.append({"image": image_value, "reason": str(exc)})
                continue

            self.matched.append(
                {
                    "image_path": self.image_dir / matched_rel,
                    "water": water,
                    "solids": solids,
                    "bitumen": bitumen,
                    "pan": pan,
                    "campaign": parse_campaign_id(Path(matched_rel).name, row),
                }
            )

        self.total_csv_rows = len(df)

        train_portion, val_portion, test_portion = self._assign_splits(
            list(self.matched),
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
            split_mode=split_mode,
        )

        if split == "train":
            self.data = train_portion
        elif split == "val":
            self.data = val_portion
        else:
            self.data = test_portion

        self.output_stats = {}
        self.bin_edges = {}
        for key, label in (("water", "Water"), ("solids", "Solids"), ("bitumen", "Bitumen")):
            train_values = [item[key] for item in train_portion]
            mean, std = self._compute_mean_std(train_values)
            self.output_stats[label] = {"mean": mean, "std": std}
            self.bin_edges[label] = self._equal_frequency_edges(train_values, CLS_BINS)

        self.train_transforms = build_train_transforms(self.image_size)
        self.val_transforms = build_eval_transforms(self.image_size, legacy_crop=self.legacy_crop)
        self.transforms = self.train_transforms if split == "train" else self.val_transforms

    @staticmethod
    def _index_images(image_dir: Path):
        """Map CSV names (basename, relative path, or stem) to a path under ``image_dir``."""
        by_name = {}
        by_stem = {}
        for path_str in collect_images(image_dir):
            path = Path(path_str)
            try:
                rel = str(path.relative_to(image_dir))
            except ValueError:
                rel = path.name
            by_name[path.name] = rel
            by_name[rel] = rel
            by_stem[path.stem] = rel
        return by_name, by_stem

    def _assign_splits(self, matched, val_fraction, test_fraction, seed, split_mode):
        if split_mode == "experiment":
            result = self._split_by_campaign(matched, val_fraction, test_fraction, seed)
            if result is not None:
                return result
        shuffled = list(matched)
        random.seed(seed)
        random.shuffle(shuffled)
        train_portion, val_portion, test_portion = self._split_portions(
            shuffled, val_fraction=val_fraction, test_fraction=test_fraction
        )
        self.split_campaigns = {
            "train": sorted({item["campaign"] for item in train_portion}),
            "val": sorted({item["campaign"] for item in val_portion}),
            "test": sorted({item["campaign"] for item in test_portion}),
        }
        return train_portion, val_portion, test_portion

    def _split_by_campaign(self, matched, val_fraction, test_fraction, seed):
        """Hold out entire campaigns (paper Case 2). Falls back if < 2 campaigns."""
        groups = defaultdict(list)
        for item in matched:
            groups[item["campaign"]].append(item)

        campaign_ids = list(groups.keys())
        if len(campaign_ids) < 2:
            self.split_fallback_reason = (
                "Only one flotation campaign was found, so a random image split was used instead."
            )
            self.split_mode = "random"
            return None

        rng = random.Random(seed)
        rng.shuffle(campaign_ids)

        n = len(campaign_ids)
        test_n = 0
        val_n = 0
        if n >= 3 and test_fraction > 0:
            test_n = max(1, round(n * test_fraction))
        if n - test_n >= 2 and val_fraction > 0:
            val_n = max(1, round(n * val_fraction))
        while test_n + val_n >= n:
            if test_n >= val_n and test_n > 0:
                test_n -= 1
            elif val_n > 0:
                val_n -= 1
            else:
                break

        test_ids = campaign_ids[:test_n]
        val_ids = campaign_ids[test_n : test_n + val_n]
        train_ids = campaign_ids[test_n + val_n :]

        train_portion = [item for cid in train_ids for item in groups[cid]]
        val_portion = [item for cid in val_ids for item in groups[cid]]
        test_portion = [item for cid in test_ids for item in groups[cid]]

        rng.shuffle(train_portion)
        rng.shuffle(val_portion)
        rng.shuffle(test_portion)

        self.split_campaigns = {
            "train": sorted(train_ids),
            "val": sorted(val_ids),
            "test": sorted(test_ids),
        }
        return train_portion, val_portion, test_portion

    @staticmethod
    def _split_portions(shuffled, val_fraction, test_fraction):
        """Split a shuffled list into train / val / test."""
        total = len(shuffled)
        if total == 0:
            return [], [], []

        test_count = int(total * test_fraction)
        val_count = int(total * val_fraction)
        # At least one train sample if we have enough data.
        if total >= 3:
            test_count = max(1, test_count) if test_fraction > 0 else 0
            val_count = max(1, val_count) if val_fraction > 0 else 0
            while test_count + val_count >= total:
                if test_count >= val_count and test_count > 1:
                    test_count -= 1
                elif val_count > 1:
                    val_count -= 1
                else:
                    break
        elif total == 2:
            # One train + one val; skip test so train isn't empty.
            test_count = 0
            val_count = 1
        else:
            test_count = 0
            val_count = 0

        test_portion = shuffled[total - test_count :] if test_count else []
        val_end = total - test_count
        val_portion = shuffled[val_end - val_count : val_end] if val_count else []
        train_portion = shuffled[: val_end - val_count]
        return train_portion, val_portion, test_portion

    @staticmethod
    def _parse_float(raw_value, column_name):
        """Parse a numeric cell; raise ValueError with a clear message if not."""
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"{column_name}={raw_value!r} is not a valid number") from None
        if math.isnan(value):
            raise ValueError(f"{column_name} is missing/blank")
        return value

    @staticmethod
    def _parse_pan(raw_value):
        """Parse Pan as int; also accepts float-like strings like "3.0"."""
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            pass
        try:
            as_float = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"Pan={raw_value!r} is not a valid whole number") from None
        if math.isnan(as_float):
            raise ValueError("Pan is missing/blank")
        return int(as_float)

    @staticmethod
    def _compute_mean_std(values):
        count = len(values)
        if count == 0:
            return 0.0, 1.0
        mean = sum(values) / count
        variance = sum((value - mean) ** 2 for value in values) / count
        std = variance ** 0.5
        if std == 0:
            std = 1.0
        return mean, std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = prepare_image(item["image_path"], self.image_size, square=not self.legacy_crop)
        if not self.legacy_crop and image.size != (self.image_size, self.image_size):
            image = standardize_to_model_size(image, self.image_size)
        image_tensor = self.transforms(image)
        # Last-resort guard if a custom transform pipeline is swapped in later.
        if image_tensor.shape[-2:] != (self.image_size, self.image_size):
            image_tensor = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        target = [item["water"], item["solids"], item["bitumen"]]
        if self.normalise:
            for index, label in enumerate(("Water", "Solids", "Bitumen")):
                stats = self.output_stats[label]
                target[index] = (target[index] - stats["mean"]) / stats["std"]

        return image_tensor, torch.tensor(target, dtype=torch.float32)

    @staticmethod
    def _equal_frequency_edges(values, n_bins=CLS_BINS):
        """Interior edges for equal-frequency bins, computed on train labels only."""
        if n_bins < 2 or len(values) < n_bins:
            return []
        ordered = sorted(values)
        count = len(ordered)
        edges = []
        for index in range(1, n_bins):
            position = min(count - 1, max(1, round(index * count / n_bins)))
            edges.append(float(ordered[position]))
        for index in range(1, len(edges)):
            if edges[index] <= edges[index - 1]:
                edges[index] = edges[index - 1] + 1e-6
        return edges

    def get_output_stats(self):
        return self.output_stats

    def get_bin_edges(self):
        return self.bin_edges

    def get_match_summary(self):
        match_rate = len(self.matched) / self.total_csv_rows if self.total_csv_rows else 0.0
        return {
            "total_csv_rows": self.total_csv_rows,
            "matched": len(self.matched),
            "unmatched": len(self.unmatched),
            "unmatched_files": list(self.unmatched),
            "invalid": len(self.invalid_rows),
            "invalid_rows": list(self.invalid_rows),
            "match_rate": match_rate,
        }
