# run_static_demo.py
"""Static demonstration of MobileNetV2 ONNX inference.

- Downloads a pre‑trained MobileNetV2 ONNX model (if not present).
- Downloads ImageNet class labels.
- Downloads a sample image.
- Runs inference via :class:`adapter.mobilenet_v2_adapter.MobileNetV2Adapter`.
- Prints the predicted label, confidence, and the actual ONNX Runtime execution provider used.
"""

import os
import urllib.request
from pathlib import Path

from PIL import Image
import numpy as np

# Import the adapter from the project package
from adapter.mobilenet_v2_adapter import MobileNetV2Adapter

def download_file(url: str, dest: Path) -> None:
    """Download *url* to *dest* if the file does not already exist."""
    if dest.is_file():
        print(f"[OK] Already present: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DOWN] Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, str(dest))
    print(f"[OK] Download complete: {dest}")

# ---------------------------------------------------------------------------
# Paths & URLs
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLE_DIR = PROJECT_ROOT / "sample"

MODEL_URL = "https://huggingface.co/onnxmodelzoo/mobilenetv2-12/resolve/main/mobilenetv2-12.onnx"
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
IMAGE_URL = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"

MODEL_PATH = MODELS_DIR / "mobilenetv2-12.onnx"
LABELS_PATH = MODELS_DIR / "imagenet_classes.txt"
IMAGE_PATH = SAMPLE_DIR / "sample.jpg"

# ---------------------------------------------------------------------------
# Download assets (model, labels, image)
# ---------------------------------------------------------------------------
download_file(MODEL_URL, MODEL_PATH)
download_file(LABELS_URL, LABELS_PATH)
download_file(IMAGE_URL, IMAGE_PATH)

# ---------------------------------------------------------------------------
# Load model via adapter
# ---------------------------------------------------------------------------
adapter = MobileNetV2Adapter()
adapter.load(str(MODEL_PATH), str(LABELS_PATH))

# ---------------------------------------------------------------------------
# Prepare image
# ---------------------------------------------------------------------------
pil_img = Image.open(IMAGE_PATH).convert("RGB")
np_img = np.array(pil_img)

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
result = adapter.predict(np_img)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
print("--- Inference Result ---")
print(f"Predicted label   : {result['label']}")
print(f"Confidence (prob): {result['confidence']:.4f}")
print(f"Execution provider: {result.get('active_provider', 'unknown')}")
print(f"Fallback to CPU   : {result.get('fallback')}")
