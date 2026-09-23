# adapter/mobilenet_v2_adapter.py
"""Concrete adapter for MobileNetV2 (quantized ONNX) compatible with Qualcomm QNN.

The adapter loads the model, attempts to use the QNNExecutionProvider (NPU). If the provider is unavailable or the model cannot be loaded with QNN, it falls back to CPUExecutionProvider and records that fallback.

The public API matches :class:`adapter.base.ModelAdapter`.
"""

import os
import numpy as np
from typing import Dict, Any, List

import onnxruntime as ort
from PIL import Image

from .base import ModelAdapter


class MobileNetV2Adapter(ModelAdapter):
    """Adapter for a quantized MobileNetV2 ONNX model.

    Expected files:
    - ``model_path`` – ``.onnx`` file, preferably INT8 quantized for QNN.
    - ``labels_path`` – Text file with one label per line (e.g., ImageNet class names).
    """

    def __init__(self) -> None:
        super().__init__()
        self._fallback = False  # True if we had to fall back to CPU
        self._input_name = None
        self._output_name = None
        # Default preprocessing for ImageNet‑trained MobileNetV2
        self._preprocess_cfg = {
            "size": (224, 224),
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "rgb": True,
        }

    def load(self, model_path: str, labels_path: str, preprocess_config: Dict[str, Any] = None) -> None:
        """Load the ONNX model and label file.

        Parameters
        ----------
        model_path: str
            Path to the quantized MobileNetV2 ONNX file.
        labels_path: str
            Path to a ``labels.txt`` file.
        preprocess_config: dict, optional
            Override default preprocessing configuration.
        """
        if preprocess_config:
            self._preprocess_cfg.update(preprocess_config)

        # Load label list
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        with open(labels_path, "r", encoding="utf-8") as f:
            self._labels = [line.strip() for line in f if line.strip()]

        # Prepare session options – attempt QNN first
        sess_opts = ort.SessionOptions()
        providers = []
        provider_options = []
        # QNNExecutionProvider requires the onnxruntime‑qnn package; we attempt it and catch errors.
        try:
            # The backend_path depends on platform – on Windows it is typically QnnHtp.dll for the NPU.
            # Using a generic name; users can adjust via env var if needed.
            qnn_options = {"backend_path": "QnnHtp.dll"}
            providers.append("QNNExecutionProvider")
            provider_options.append(qnn_options)
            self._session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers, provider_options=provider_options)
        except Exception as e:
            # Fallback to CPU and record the event
            self._fallback = True
            providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)

        # Cache input/output names for fast inference
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize, normalize and convert image to the model's expected format.

        The function expects a ``numpy.ndarray`` in **RGB** order (0‑255 uint8).
        Returns a ``float32`` tensor shaped ``[1, C, H, W]``.
        """
        cfg = self._preprocess_cfg
        # Resize
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize(cfg["size"], Image.BILINEAR)
        # Convert to float32 and scale to [0, 1]
        np_img = np.asarray(pil_img).astype(np.float32) / 255.0
        # Normalize
        mean = np.array(cfg["mean"], dtype=np.float32)
        std = np.array(cfg["std"], dtype=np.float32)
        np_img = (np_img - mean) / std
        # Change to CHW layout
        np_img = np.transpose(np_img, (2, 0, 1))
        # Add batch dimension
        np_img = np.expand_dims(np_img, axis=0)
        return np_img.astype(np.float32)

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Run inference on a single image.

        Parameters
        ----------
        image: np.ndarray
            The image as a ``numpy`` array in **RGB** order (height, width, 3).

        Returns
        -------
        dict
            ``{"label": str, "confidence": float, "raw_scores": List[float]}``
        """
        if self._session is None:
            raise RuntimeError("Model not loaded. Call `load` first.")

        # Preprocess image to tensor
        input_tensor = self._preprocess(image)
        # Run inference
        raw_output = self._session.run([self._output_name], {self._input_name: input_tensor})[0]
        # The output is typically a 2‑D array [[batch, num_classes]]
        scores = raw_output.squeeze().astype(np.float32)
        # Apply softmax to obtain probabilities
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
