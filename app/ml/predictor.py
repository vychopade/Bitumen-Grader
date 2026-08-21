import torch

from app.constants import OUTPUT_NAMES, SUM_DEVIATION_OK
from app.ml.cnn_model import BitumenRegressor
from app.utils.image_utils import (
    build_eval_transforms,
    image_size_from_metadata,
    is_legacy_resnet18,
    prepare_image,
)


class RegressionPredictor:
    """Load a trained checkpoint and predict Water / Solids / Bitumen %.

    If training used z-scored targets, we undo that with ``output_stats``
    from the metadata JSON. Otherwise raw outputs are already percentages.
    Architecture and image size come from metadata so baseline / ResNet50 /
    VGG16 / legacy ResNet-18 checkpoints all load correctly.
    """

    OUTPUT_NAMES = list(OUTPUT_NAMES)

    def __init__(self, model_path, metadata):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.metadata = metadata or {}
        self.model = BitumenRegressor.from_checkpoint(model_path, self.metadata, device)

        self.output_stats = metadata["output_stats"]
        self.normalise_targets = bool(metadata.get("normalise_targets", True))
        self.output_names = list(self.OUTPUT_NAMES)
        self.model.eval()

        image_size = image_size_from_metadata(self.metadata)
        self.image_size = image_size
        self.transform = build_eval_transforms(
            image_size, legacy_crop=is_legacy_resnet18(self.metadata)
        )

    def _to_percentages(self, raw_outputs) -> list:
        """Map 3 raw outputs to [0, 100] percentages (undo z-score if needed)."""
        values = []
        for index, name in enumerate(self.output_names):
            value = raw_outputs[index].item()
            if self.normalise_targets:
                stats = self.output_stats[name]
                value = value * stats["std"] + stats["mean"]
            value = max(0.0, min(100.0, value))
            values.append(float(value))
        return values

    def predict(self, pil_image) -> dict:
        image = prepare_image(pil_image, self.image_size)
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_outputs = self.model(tensor)[0]

        water, solids, bitumen = self._to_percentages(raw_outputs)
        total_sum = water + solids + bitumen
        sum_deviation = abs(total_sum - 100.0)

        return {
            "Water": {"value": water, "unit": "%"},
            "Solids": {"value": solids, "unit": "%"},
            "Bitumen": {"value": bitumen, "unit": "%"},
            "sum": total_sum,
            "sum_deviation": sum_deviation,
            "sum_ok": sum_deviation < SUM_DEVIATION_OK,
        }
