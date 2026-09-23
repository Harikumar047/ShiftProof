"""Concrete adapter for SqueezeNet 1.1 ONNX classifier.

NOTE: This is a stand-in pair (MobileNetV2 vs SqueezeNet) for demonstrating the
ShiftProof comparison pipeline, not the proposal's full trained-bias methodology.
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional
from PIL import Image
import onnxruntime as ort

from .base import ModelAdapter


class SqueezeNetAdapter(ModelAdapter):
    """Adapter for SqueezeNet 1.1 ONNX classification model.

    Reuses standard ImageNet normalization and executes on available runtime providers.
    """

    def __init__(self) -> None:
        super().__init__()
        self._fallback = False
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None

        # Standard ImageNet preprocessing configuration
        self._preprocess_cfg = {
            "size": (224, 224),
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "rgb": True,
        }

    def load(
        self,
        model_path: str,
        labels_path: str,
        preprocess_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Load SqueezeNet ONNX model and ImageNet labels."""
        if preprocess_config:
            self._preprocess_cfg.update(preprocess_config)

        if not os.path.isfile(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        with open(labels_path, "r", encoding="utf-8") as f:
            self._labels = [line.strip() for line in f if line.strip()]

        sess_opts = ort.SessionOptions()
        # Attempt QNN provider first if available, otherwise fallback to CPU
        providers = []
        try:
            qnn_options = {"backend_path": "QnnHtp.dll"}
            providers.append("QNNExecutionProvider")
            self._session = ort.InferenceSession(
                model_path,
                sess_options=sess_opts,
                providers=providers,
                provider_options=[qnn_options],
            )
            self._active_provider = self._session.get_providers()[0]
        except Exception:
            self._fallback = True
            providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(
                model_path, sess_options=sess_opts, providers=providers
            )
            self._active_provider = self._session.get_providers()[0]

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize, normalize, and format image into [1, 3, 224, 224] float32 tensor."""
        cfg = self._preprocess_cfg
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize(cfg["size"], Image.BILINEAR)

        np_img = np.asarray(pil_img).astype(np.float32) / 255.0
        mean = np.array(cfg["mean"], dtype=np.float32)
        std = np.array(cfg["std"], dtype=np.float32)
        np_img = (np_img - mean) / std
        np_img = np.transpose(np_img, (2, 0, 1))
        np_img = np.expand_dims(np_img, axis=0)
        return np_img.astype(np.float32)

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Run inference on a single RGB numpy image array.

        Returns:
            Dict containing label, confidence, raw_scores, fallback, and active_provider.
        """
        if self._session is None:
            raise RuntimeError("Model not loaded. Call `load` first.")

        input_tensor = self._preprocess(image)
        raw_output = self._session.run(
            [self._output_name], {self._input_name: input_tensor}
        )[0]
        scores = raw_output.squeeze().astype(np.float32)

        # Softmax for probabilities
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        top_idx = int(np.argmax(probs))
        label = self._labels[top_idx] if self._labels else str(top_idx)
        confidence = float(probs[top_idx])

        return {
            "label": label,
            "confidence": confidence,
            "raw_scores": probs.tolist(),
            "fallback": self._fallback,
            "active_provider": self.active_provider,
        }
