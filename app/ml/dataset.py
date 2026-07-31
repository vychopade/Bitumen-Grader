import os
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class RegressionDataset(Dataset):
    """Loads bitumen sample images and their (Water, Solids, Bitumen) regression targets.

    Rows are read from a CSV (columns: Image, Pan, Water, Solids, Bitumen)
    and matched against image files in ``image_dir``. The matched set is
    shuffled deterministically and split into train/val portions; whichever
    portion this instance represents (``split``) is exposed via ``__len__``/
    ``__getitem__``, while normalisation statistics are always derived from
    the training portion only.
    """

    EXPECTED_COLUMNS = ["Image", "Pan", "Water", "Solids", "Bitumen"]
    EXTENSION_CANDIDATES = (".jpg", ".jpeg", ".png", ".tif")
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, csv_path, image_dir, split="train", val_fraction=0.2, normalise=True, seed=42):
        self.csv_path = csv_path
        self.image_dir = Path(image_dir)
        self.split = split
        self.val_fraction = val_fraction
        self.normalise = normalise
        self.seed = seed

        df = pd.read_csv(csv_path)
        missing_columns = [column for column in self.EXPECTED_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV is missing expected columns: {missing_columns}")

        filenames = os.listdir(self.image_dir)
        filename_set = set(filenames)
        stem_lookup = {Path(name).stem: name for name in filenames}

        self.matched = []
        self.unmatched = []

        for _, row in df.iterrows():
            image_value = str(row["Image"])
            matched_name = None

            if image_value in filename_set:
                matched_name = image_value
            elif image_value in stem_lookup:
                matched_name = stem_lookup[image_value]
            else:
                for extension in self.EXTENSION_CANDIDATES:
                    candidate = image_value + extension
                    if candidate in filename_set:
                        matched_name = candidate
                        break

            if matched_name is None:
                self.unmatched.append(image_value)
                continue

            self.matched.append(
                {
                    "image_path": self.image_dir / matched_name,
                    "water": float(row["Water"]),
                    "solids": float(row["Solids"]),
                    "bitumen": float(row["Bitumen"]),
                    "pan": int(row["Pan"]),
                }
            )

        self.total_csv_rows = len(df)

        shuffled = list(self.matched)
        random.seed(seed)
        random.shuffle(shuffled)
        split_index = int(len(shuffled) * (1 - val_fraction))
        train_portion = shuffled[:split_index]
        val_portion = shuffled[split_index:]

        self.data = train_portion if split == "train" else val_portion

        self.output_stats = {}
        for key, label in (("water", "Water"), ("solids", "Solids"), ("bitumen", "Bitumen")):
            mean, std = self._compute_mean_std([item[key] for item in train_portion])
            self.output_stats[label] = {"mean": mean, "std": std}

        self.train_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop(224),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ]
        )
        self.transforms = self.train_transforms if split == "train" else self.val_transforms

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
        image = Image.open(item["image_path"]).convert("RGB")
        image_tensor = self.transforms(image)

        target = [item["water"], item["solids"], item["bitumen"]]
        if self.normalise:
            for index, label in enumerate(("Water", "Solids", "Bitumen")):
                stats = self.output_stats[label]
                target[index] = (target[index] - stats["mean"]) / stats["std"]

        return image_tensor, torch.tensor(target, dtype=torch.float32)

    def get_output_stats(self):
        return self.output_stats

    def get_match_summary(self):
        match_rate = len(self.matched) / self.total_csv_rows if self.total_csv_rows else 0.0
        return {
            "total_csv_rows": self.total_csv_rows,
            "matched": len(self.matched),
            "unmatched": len(self.unmatched),
            "unmatched_files": list(self.unmatched),
            "match_rate": match_rate,
        }

    def get_pan_distribution(self):
        distribution = {}
        for item in self.data:
            pan = item["pan"]
            distribution[pan] = distribution.get(pan, 0) + 1
        return distribution

    def get_output_ranges(self):
        ranges = {}
        for key, label in (("water", "Water"), ("solids", "Solids"), ("bitumen", "Bitumen")):
            values = [item[key] for item in self.data]
            if values:
                ranges[label] = {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}
            else:
                ranges[label] = {"min": 0.0, "max": 0.0, "mean": 0.0}
        return ranges
