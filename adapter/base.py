# adapter/base.py
"""Abstract base class for model adapters.

All concrete adapters must implement:
- `load(model_path: str, labels_path: str, preprocess_config: dict) -> None`
- `predict(image: "numpy.ndarray") -> dict` returning a dict with keys `label`, `confidence`, `raw_scores`
- `active_provider` property exposing the ONNX Runtime execution provider that was actually used.
"""

import abc
from typing import Dict, Any


class ModelAdapter(abc.ABC):
    """Base interface for image‑classifier adapters.

    The adapter hides model‑specific loading, preprocessing and inference details.
    It also reports which ONNX Runtime execution provider is active after the session is created.
    """

    def __init__(self) -> None:
        self._session = None  # type: ignore
        self._labels = []
        self._preprocess_cfg = {}

    @abc.abstractmethod
    def load(self, model_path: str, labels_path: str, preprocess_config: Dict[str, Any] = None) -> None:
        """Load an ONNX model and associated label mapping.

        Parameters
        ----------
        model_path: str
            Path to the ``.onnx`` file.
        labels_path: str
            Path to a plain‑text file with one label per line.
        preprocess_config: dict, optional
            Dictionary describing preprocessing steps (e.g. target size, means, stds).
        """

    @abc.abstractmethod
    def predict(self, image) -> Dict[str, Any]:
        """Run inference on a single image.

        Returns a dictionary with at least ``label``, ``confidence`` and ``raw_scores`` keys.
        """

    @property
    def active_provider(self) -> str:
        """Return the execution provider that the session is currently using.

        If the session has not been created yet, returns ``"unloaded"``.
        """
        if self._session is None:
            return "unloaded"
        try:
            providers = self._session.get_providers()
            return providers[0] if providers else "unknown"
        except Exception:
            return "unknown"
