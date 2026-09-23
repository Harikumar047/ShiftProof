# ShiftProof — Demo Script

This script walks through the ShiftProof MVP end-to-end, suitable for a live demo or self-guided walkthrough.

---

## Setup (one-time)

```bash
# Install dependencies
pip install onnxruntime opencv-python Pillow numpy PySide6

# Download models into models/
# mobilenetv2-12.onnx  — from ONNX Model Zoo
# squeezenet1.1-7.onnx — from ONNX Model Zoo
# imagenet_classes.txt — 1000-line ImageNet label file
```

---

## Part 1: Static Inference Demo

**Goal**: Confirm that ONNX inference is wired correctly and the execution provider fallback is recorded.

```bash
python run_static_demo.py
```

**Expected output**:
```
Predicted Label : Samoyed
Confidence      : 0.8595
Active Provider : CPUExecutionProvider
Fallback        : True  (QNN requested but unavailable on this host)
```

**Talking point**: The adapter always records whether the requested provider (QNN/NPU) was actually used or fell back to CPU. This is written to the database and appears in every report — no silent provider substitutions.

---

## Part 2: Storage Module Demo

```bash
python test_storage.py
```

**What it demonstrates**:
- Schema creation (models, sessions, captures tables)
- Insert and query typed records
- Session summary counts by condition type
- JSON export
- Cascade delete

---

## Part 3: Guided Capture UI

```bash
python main.py
```

**Walk through the UI**:

1. **Live camera preview** appears on the left with a cyan alignment guide overlay.
2. **Live inference** updates every ~150ms — predicted label and confidence bar visible on the right panel.
3. **Active Provider badge** shows which ONNX execution provider is actually running.

**Demonstrate mandatory label confirmation**:
- Leave the true label blank → Confirm button stays greyed out.
- Type `water bottle` → Confirm button activates immediately.

**Make a capture**:
- Condition: `baseline`, Note: `neutral indoor lighting`
- Click **Confirm & Save Capture**
- Green toast shows: `[OK] Saved Capture #<id> [baseline] → Correct/Mismatch`
- Summary counter increments live.

**Camera not found scenario** (demo on any machine without a webcam):
- UI displays an in-widget error banner with a Reconnect button — app does not crash.

---

## Part 4: Capture Flow Headless Test

```bash
python test_capture_flow.py
```

**What it demonstrates**:
- `CameraNotFoundError` raised (not a silent black frame) when camera index is invalid.
- Full capture flow: predict → label confirm → write image to disk → insert DB row.
- File existence and DB record verified programmatically.

---

## Part 5: Evaluation & Comparison Engine

```bash
python test_evaluation.py
```

**What it demonstrates**:
- `score_model()` — per-capture correctness, per-condition accuracy, overall accuracy.
- `compare_models()` — runs BOTH models against the SAME saved images, never re-captures.
- Prediction change rate and per-condition accuracy delta matrix.

---

## Part 6: End-to-End Pipeline & HTML Report

```bash
# Evaluate most recent real session
python run_pipeline.py

# Or target a specific session
python run_pipeline.py --session-id 1
```

**What it demonstrates**:
- Database path printed on startup (verify GUI and pipeline use the same file)
- Real captures loaded — falls back to synthetic ONLY if no sessions with captures exist
- Latency benchmark: median + p95 after warm-up
- Self-contained HTML report generated with embedded images

**Open the report**:
```
reports/shiftproof_evaluation_report.html
```

**Point out in the report**:
1. Overview stats: per-model accuracy, prediction shift rate, agreement rate
2. Condition-level shift matrix table
3. Paired capture gallery with before/after model comparison
4. Benchmark deployment config table (provider_requested vs provider_active)
5. **Disclosures section** — four explicit disclosures:
   - CPU-only execution / NPU not tested
   - Stand-in model pair (not final trained-bias methodology)
   - Softmax scores ≠ calibrated probabilities
   - Capture data provenance (programmatic transforms vs live webcam)

---

## Key Numbers to Quote

| Metric | Value |
| :--- | :--- |
| MobileNetV2 inference latency (median) | ~2–8 ms (CPU) |
| SqueezeNet 1.1 inference latency (median) | ~2–5 ms (CPU) |
| End-to-end response (disk load + infer) | ~10–40 ms |
| Prediction shift rate (distance condition) | 25% — both models fail on heavily padded far-field crop |
| Distance condition accuracy | 0% both models — genuine domain shift finding |

---

## What Is NOT in This MVP

- QNN/NPU execution (target hardware not yet available)
- The final trained-bias dataset methodology from the proposal
- Multi-class or multi-object scene evaluation
- Export to PDF or structured JSON report (HTML only)
- Cloud sync or networked session sharing
