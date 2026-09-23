# ShiftProof

**ShiftProof** is an edge AI shift-testing pipeline for evaluating how ImageNet classifiers behave under real-world condition changes — lighting, background clutter, distance, and more. It combines guided webcam capture, ONNX model inference, comparative evaluation, latency benchmarking, and self-contained HTML reporting into a single end-to-end workflow.

---

## Project Status

MVP complete. All core modules implemented and tested:

| Module | Status |
| :--- | :--- |
| `adapter/` — ONNX model adapters (MobileNetV2, SqueezeNet) | ✅ Done |
| `storage/` — SQLite persistence (models, sessions, captures) | ✅ Done |
| `capture/` — Camera session + mandatory label-confirmed capture flow | ✅ Done |
| `evaluation/` — Single-model scoring + cross-model comparison | ✅ Done |
| `benchmark/` — Inference latency + end-to-end response time | ✅ Done |
| `reporting/` — Self-contained HTML report with disclosures | ✅ Done |
| `ui/` — PySide6 guided capture widget with live inference preview | ✅ Done |
| `run_pipeline.py` — CLI evaluation & report pipeline | ✅ Done |
| `main.py` — App entry point | ✅ Done |

---

## Quickstart

### Prerequisites

```bash
pip install onnxruntime opencv-python Pillow numpy PySide6
```

Download models and place in `models/`:
- `mobilenetv2-12.onnx` from the ONNX Model Zoo
- `squeezenet1.1-7.onnx` from the ONNX Model Zoo
- `imagenet_classes.txt` — 1000-line ImageNet label file

### Run the Guided Capture GUI

```bash
python main.py
```

Opens the PySide6 capture widget. For each capture:
1. Position your object inside the cyan alignment guide
2. The live prediction updates at ~7 fps
3. Type or select the **exact ground-truth label** from the autocomplete list
4. Choose condition type (`baseline` / `background` / `lighting` / `distance`)
5. Click **Confirm & Save Capture**

### Run the Evaluation & Report Pipeline

```bash
# Evaluate the most recent session automatically
python run_pipeline.py

# Evaluate a specific session by ID
python run_pipeline.py --session-id 2

# Use a custom database path
python run_pipeline.py --db-path path/to/other.db

# Force generate synthetic demo data
python run_pipeline.py --force-synthetic
```

Output: `reports/session_<id>_report.html` — self-contained, no external assets needed.

### Run the Static Inference Demo

```bash
python run_static_demo.py
```

Runs a single inference on a bundled sample image and prints the predicted label, confidence, and the actual ONNX execution provider used.

---

## Project Structure

```
ShiftProof/
├── adapter/                  # ONNX model adapters
│   ├── base.py               # Abstract ModelAdapter base class
│   ├── mobilenet_v2_adapter.py
│   └── squeezenet_adapter.py
├── benchmark/
│   └── latency.py            # Inference + end-to-end latency profiling
├── capture/
│   ├── camera.py             # CameraSession with CameraNotFoundError
│   └── capture_flow.py       # CaptureFlow with mandatory label confirmation
├── evaluation/
│   ├── scoring.py            # Single-model scoring
│   └── compare.py            # Cross-model comparison
├── reporting/
│   └── report_generator.py   # Self-contained HTML report generator
├── storage/
│   ├── db.py                 # SQLite init and schema
│   ├── repository.py         # Typed insert/query helpers
│   └── export.py             # JSON export and session deletion
├── ui/
│   └── capture_widget.py     # PySide6 guided capture UI widget
├── models/                   # ONNX weights + ImageNet labels (not committed)
├── sample/                   # Reference test image
├── captures/                 # Runtime capture images (not committed)
├── reports/                  # Generated HTML reports (not committed)
├── main.py                   # GUI application entry point
├── run_pipeline.py           # CLI evaluation + report pipeline
├── run_static_demo.py        # Single-image static inference demo
├── test_storage.py           # Storage module tests
├── test_capture_flow.py      # Headless capture flow tests
├── test_evaluation.py        # Evaluation engine tests
└── test_ui_widget.py         # Headless PySide6 UI logic tests
```

---

## Key Design Decisions

- **BGR/RGB boundary**: Camera frames stay in OpenCV BGR format throughout. Conversion to RGB only occurs at the model input boundary (`predict()`) and the display boundary (`CameraPreviewWidget`).
- **Mandatory label confirmation**: `CaptureFlow.confirm_and_save()` raises `LabelConfirmationError` if `true_label` is empty. The UI disables the confirm button unless a label is typed — two independent enforcement layers.
- **Provider transparency**: Both adapters attempt QNN first and fall back to CPU. The active provider is always recorded in the database and printed in every report. The benchmark explicitly notes "NPU/QNN not tested — target hardware not yet available".
- **No calibrated probabilities**: Confidence scores are raw softmax outputs and are never described as calibrated probabilities in any report output or UI label.

---

## Disclosures

- **Stand-in model pair**: MobileNetV2 vs SqueezeNet 1.1 is a demonstration pair for the comparison pipeline, not the final trained-bias methodology described in the project proposal.
- **Capture data in demo report**: The reference demo report (`reports/shiftproof_evaluation_report.html`) uses programmatically generated condition images (transforms applied to a single reference photo) to validate the end-to-end pipeline. Live webcam capture is fully operational via `main.py`.
- **Hardware**: All benchmarks run on `CPUExecutionProvider`. QNN/NPU target hardware testing is deferred until the target device is available.

---

## Testing

All tests are headless and do not require a webcam:

```bash
python test_storage.py       # Storage CRUD, export, and cascade delete
python test_capture_flow.py  # Camera error handling, capture flow, disk persistence, DB verify
python test_evaluation.py    # Scoring, comparison engine, condition accuracy matrix
python test_ui_widget.py     # PySide6 widget logic (offscreen, no display needed)
```
