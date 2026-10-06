"""
src/models/resnet_adapter.py — ResNet-18 Oxford-IIIT Pet implementation of BaseVisionModel.

Wraps the PyTorch ResNet-18 Oxford-IIIT Pet checkpoint with:
- .predict(): Runs forward pass, applies softmax, returns (class_idx, label, conf, probs).
- .explain(): Hooks the final layer4 residual block to generate GradCAM attention heatmaps.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.base import BaseVisionModel
from src.utils import (
    OXFORD_PET_CLASSES,
    OXFORD_PET_NUM_CLASSES,
    RESNET18_MODEL_PATH,
    get_device,
    load_resnet18_model,
    pil_to_tensor,
)
from src.gradcam import ResNetGradCAM


class ResNet18OxfordPetAdapter(BaseVisionModel):
    """
    Adapter bridging ResNet-18 (trained on Oxford-IIIT Pet 37-breed classification)
    to the model-agnostic BaseVisionModel interface.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.device = device or get_device()
        self.class_names = class_names or OXFORD_PET_CLASSES
        ckpt = Path(checkpoint_path) if checkpoint_path else RESNET18_MODEL_PATH
        self.model = load_resnet18_model(
            ckpt,
            self.device,
            num_classes=len(self.class_names),
        )
        self.model.eval()
        self._gradcam: Optional[ResNetGradCAM] = None

    def predict(self, image: Image.Image) -> Tuple[int, str, float, List[float]]:
        tensor = pil_to_tensor(image).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        pred_idx = int(probs.argmax().item())
        confidence = float(probs[pred_idx].item())
        label = (
            self.class_names[pred_idx]
            if pred_idx < len(self.class_names)
            else str(pred_idx)
        )
        all_probs = probs.cpu().tolist()

        return pred_idx, label, confidence, all_probs

    def explain(
        self,
        image: Image.Image,
        target_class: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        if self._gradcam is None:
            self._gradcam = ResNetGradCAM(self.model, self.device)

        cam, _, _ = self._gradcam.generate(image, target_class=target_class)
        return cam

    def close(self) -> None:
        if self._gradcam is not None:
            self._gradcam.remove_hooks()
            self._gradcam = None


# Alias for general use
ResNetAdapter = ResNet18OxfordPetAdapter
