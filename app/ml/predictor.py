"""Load a saved checkpoint and grade one or more froth photos."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import torch
from PIL import Image

from app.constants import OUTPUT_NAMES, SUM_DEVIATION_OK
from app.ml.cnn_model import BitumenRegressor, select_torch_device
from app.ml.recipe import BATCH_SIZE
from app.utils.image_utils import (
    build_eval_transforms,
    image_size_from_metadata,
    is_legacy_resnet18,
    prepare_image,
)

ProgressCallback = Callable[[int, int], None]
ImageSource = Union[str, Path, Image.Image]


class RegressionPredictor:
    """Load a trained checkpoint and predict Water / Solids / Bitumen %.

    If training used z-scored targets, we undo that with ``output_stats``
    from the metadata JSON. Otherwise raw outputs are already percentages.
    Architecture and image size come from metadata so baseline / ResNet50 /
    VGG16 / legacy ResNet-18 checkpoints all load correctly.
    """

    def __init__(self, model_path, metadata):
        device = select_torch_device()
        self.device = device
        self.metadata = metadata or {}
        self.model = BitumenRegressor.from_checkpoint(
            model_path, self.metadata, device
        )

        self.output_stats = metadata["output_stats"]
        # Older checkpoints z-scored labels; new runs store raw %.
        self.normalise_targets = bool(metadata.get("normalise_targets", True))
        self.output_names = list(OUTPUT_NAMES)
        self.model.eval()

        image_size = image_size_from_metadata(self.metadata)
        self.image_size = image_size
        self._legacy_crop = is_legacy_resnet18(self.metadata)
        self.transform = build_eval_transforms(
            image_size, legacy_crop=self._legacy_crop
        )
        self._mean = torch.tensor(
            [
                float(self.output_stats[name]["mean"])
                for name in self.output_names
            ],
            dtype=torch.float32,
        )
        self._std = torch.tensor(
            [
                float(self.output_stats[name]["std"])
                for name in self.output_names
            ],
            dtype=torch.float32,
        )
        self._batch_size = _infer_batch_size(self.metadata, device)
        self._decode_workers = min(8, os.cpu_count() or 1)

    def _prepare_source(self, source: ImageSource) -> Image.Image:
        return prepare_image(
            source, self.image_size, square=not self._legacy_crop
        )

    def _dicts_from_raw(self, raw_batch: torch.Tensor) -> List[dict]:
        values = raw_batch.detach().float().cpu()
        if self.normalise_targets:
            values = values * self._std + self._mean
        values = values.clamp(0.0, 100.0)
        results = []
        for water, solids, bitumen in values.tolist():
            total_sum = water + solids + bitumen
            sum_deviation = abs(total_sum - 100.0)
            results.append(
                {
                    "Water": {"value": water, "unit": "%"},
                    "Solids": {"value": solids, "unit": "%"},
                    "Bitumen": {"value": bitumen, "unit": "%"},
                    "sum": total_sum,
                    "sum_deviation": sum_deviation,
                    "sum_ok": sum_deviation < SUM_DEVIATION_OK,
                }
            )
        return results

    def _decode_chunk(
        self, sources: Sequence[ImageSource]
    ) -> List[Optional[torch.Tensor]]:
        prepared: List[Optional[Image.Image]]
        if len(sources) <= 1 or self._decode_workers <= 1:
            prepared = []
            for source in sources:
                try:
                    prepared.append(self._prepare_source(source))
                except Exception:  # noqa: BLE001
                    # keep grading remaining images
                    prepared.append(None)
        else:
            workers = min(self._decode_workers, len(sources))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(self._prepare_source, source)
                    for source in sources
                ]
                prepared = []
                for future in futures:
                    try:
                        prepared.append(future.result())
                    except Exception:  # noqa: BLE001
                        # keep grading remaining images
                        prepared.append(None)

        tensors: List[Optional[torch.Tensor]] = []
        for image in prepared:
            if image is None:
                tensors.append(None)
                continue
            try:
                tensors.append(self.transform(image))
            except Exception:  # noqa: BLE001 - keep grading remaining images
                tensors.append(None)
        return tensors

    def predict(self, pil_image) -> dict:
        results = self.predict_many([pil_image])
        if not results or results[0] is None:
            raise RuntimeError("Couldn't grade this image.")
        return results[0]

    def predict_many(
        self,
        sources: Sequence[ImageSource],
        *,
        on_progress: Optional[ProgressCallback] = None,
        batch_size: Optional[int] = None,
    ) -> List[Optional[dict]]:
        """Grade many photos with batched inference. Failed items are
        ``None``."""
        total = len(sources)
        results: List[Optional[dict]] = [None] * total
        if total == 0:
            return results

        chunk_size = max(1, int(batch_size or self._batch_size))
        done = 0
        if on_progress is not None:
            on_progress(0, total)

        for start in range(0, total, chunk_size):
            chunk = list(sources[start : start + chunk_size])
            decoded = self._decode_chunk(chunk)
            ready_index = [
                index
                for index, tensor in enumerate(decoded)
                if tensor is not None
            ]
            if ready_index:
                batch = torch.stack([decoded[index] for index in ready_index])
                if self.device.type == "cuda":
                    batch = batch.to(self.device, non_blocking=True)
                else:
                    batch = batch.to(self.device)
                with torch.inference_mode():
                    raw_outputs = self.model(batch)
                for local_index, result in zip(
                    ready_index, self._dicts_from_raw(raw_outputs)
                ):
                    results[start + local_index] = result
            done = min(start + len(chunk), total)
            if on_progress is not None:
                on_progress(done, total)

        return results


def _infer_batch_size(metadata: dict, device: torch.device) -> int:
    architecture = str((metadata or {}).get("architecture", "baseline"))
    if architecture in ("vgg16", "resnet50"):
        return 8 if device.type == "cpu" else 16
    if device.type == "cpu":
        return 16
    return int(BATCH_SIZE)
