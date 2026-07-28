"""
Inference helper.

Provides a thin wrapper around a loaded CNN model that handles preprocessing
an input image and running a forward pass to produce a predicted grade/class
and associated confidence scores.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

from app.ml.cnn_model import DEFAULT_GRADE_LABELS, BitumenCNN
from app.utils.image_utils import preprocess_for_inference
from app.utils.model_io import load_model_metadata


class ModelPredictor:
    """Loads a trained ``BitumenCNN`` checkpoint and runs inference on images.

    On construction, looks for a sidecar ``.json`` metadata file (as written
    by ``app.utils.model_io.save_model``) next to the ``.pt`` checkpoint to
    determine the number of classes and grade labels, falling back to
    ``DEFAULT_GRADE_LABELS`` if none is found or explicit labels are not
    supplied.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        grade_labels: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_path = Path(model_path)

        metadata = self._load_sidecar_metadata(self.model_path)
        self.grade_labels: List[str] = grade_labels or metadata.get(
            "grade_labels", list(DEFAULT_GRADE_LABELS)
        )
        num_classes = metadata.get("num_classes", len(self.grade_labels))

        self.model = BitumenCNN.from_pretrained(
            self.model_path, num_classes=num_classes, device=self.device
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_sidecar_metadata(model_path: Path) -> Dict[str, Any]:
        """Load the ``.json`` metadata sitting alongside ``model_path``, if any."""
        json_path = model_path.with_suffix(".json")
        if json_path.exists():
            try:
                return load_model_metadata(json_path)
            except (ValueError, OSError):
                return {}
        return {}

    def predict(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Run inference on a single image and return the predicted grade.

        Args:
            pil_image: Input image to grade.

        Returns:
            A dict with keys:
                - ``grade`` (str): the predicted bitumen grade label.
                - ``confidence`` (float): predicted probability of ``grade``.
                - ``all_probabilities`` (dict[str, float]): probability for
                  every grade label the model can predict.
        """
        tensor = preprocess_for_inference(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        all_probabilities = {
            label: float(probabilities[i].item()) for i, label in enumerate(self.grade_labels)
        }

        best_idx = int(torch.argmax(probabilities).item())
        grade = self.grade_labels[best_idx]
        confidence = float(probabilities[best_idx].item())

        return {
            "grade": grade,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
        }
