import logging

import torch
from PIL import Image
from torchvision import transforms

from app.ml.cnn_model import BitumenRegressor

logger = logging.getLogger(__name__)


class RegressionPredictor:
    """Loads a trained ``BitumenRegressor`` checkpoint and runs inference on images.

    Outputs are denormalised back into original percentage units (Water,
    Solids, Bitumen) using the ``output_stats`` recorded in the model's
    metadata at training time.
    """

    OUTPUT_NAMES = ["Water", "Solids", "Bitumen"]

    def __init__(self, model_path, metadata):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = BitumenRegressor.from_pretrained(model_path, device)

        self.output_stats = metadata["output_stats"]
        self.output_names = list(self.OUTPUT_NAMES)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _denormalise(self, raw_outputs) -> list:
        """Map (3,) raw model outputs back to clamped [0, 100] original-unit floats."""
        values = []
        for index, name in enumerate(self.output_names):
            stats = self.output_stats[name]
            value = raw_outputs[index].item() * stats["std"] + stats["mean"]
            value = max(0.0, min(100.0, value))
            values.append(float(value))
        return values

    def predict(self, pil_image) -> dict:
        image = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_outputs = self.model(tensor)[0]

        water, solids, bitumen = self._denormalise(raw_outputs)
        total_sum = water + solids + bitumen
        sum_deviation = abs(total_sum - 100.0)

        return {
            "Water": {"value": water, "unit": "%"},
            "Solids": {"value": solids, "unit": "%"},
            "Bitumen": {"value": bitumen, "unit": "%"},
            "sum": total_sum,
            "sum_deviation": sum_deviation,
            "sum_ok": sum_deviation < 5.0,
        }

    def predict_batch(self, image_paths) -> list:
        results = []
        for path in image_paths:
            try:
                with Image.open(path) as opened:
                    opened.load()
                    image = opened.convert("RGB")
            except (OSError, ValueError) as exc:
                logger.warning("Could not load image %r for prediction: %s", path, exc)
                results.append(None)
                continue
            results.append(self.predict(image))
        return results
