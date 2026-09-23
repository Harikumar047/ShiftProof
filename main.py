"""ShiftProof Main Application Entry Point - Guided Capture GUI."""

import sys
import os
from pathlib import Path

from PySide6.QtWidgets import QApplication

from storage.db import init_db
from storage.repository import insert_model, insert_session
from adapter.mobilenet_v2_adapter import MobileNetV2Adapter
from capture.camera import CameraSession
from capture.capture_flow import CaptureFlow
from ui.capture_widget import CaptureWidget

DB_PATH = "shiftproof.db"
MODEL_PATH = "models/mobilenetv2-12.onnx"
LABELS_PATH = "models/imagenet_classes.txt"


def main():
    print("=" * 60)
    print("Starting ShiftProof - Edge AI Shift Testing & Guided Capture")
    print("=" * 60)

    # 1. Initialize SQLite Database
    abs_db_path = str(Path(DB_PATH).resolve())
    print(f"[*] Initializing SQLite database at: '{abs_db_path}'")
    conn = init_db(DB_PATH)

    # 2. Load Model Adapter
    print(f"[*] Loading MobileNetV2 adapter (model: {MODEL_PATH})...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print(f"[!] Error: Model or labels file missing in 'models/' directory.")
        sys.exit(1)

    adapter = MobileNetV2Adapter()
    adapter.load(model_path=MODEL_PATH, labels_path=LABELS_PATH)
    active_prov = adapter.active_provider
    fallback_note = "Genuine fallback on host / QNN unavailable" if adapter._fallback else "Direct NPU execution"
    print(f"[*] Active Execution Provider: {active_prov} ({fallback_note})")

    # 3. Register Model & New Session in SQLite
    model_id = insert_model(
        conn=conn,
        name="MobileNetV2",
        version="v2.0-quant",
        path=MODEL_PATH,
        provider_requested="QNN",
        provider_active=active_prov,
        provider_note=fallback_note,
    )

    session_id = insert_session(
        conn=conn,
        model_id=model_id,
        label="Guided Capture Live Session",
    )
    print(f"[*] Registered Session ID: #{session_id} (Model ID: #{model_id})")

    # 4. Initialize CaptureFlow
    flow = CaptureFlow(
        adapter=adapter,
        conn=conn,
        session_id=session_id,
        base_dir="captures",
    )

    # 5. Launch PySide6 GUI
    print("[*] Launching ShiftProof Guided Capture Window...")
    app = QApplication(sys.argv)
    
    # Camera session (defaults to device 0)
    camera_session = CameraSession(device_index=0)

    window = CaptureWidget(
        flow=flow,
        camera_session=camera_session,
        model_name="MobileNetV2",
    )
    window.show()

    exit_code = app.exec()
    conn.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
