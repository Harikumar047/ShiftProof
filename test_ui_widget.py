"""Headless UI test verifying CaptureWidget logic and signal flow with PySide6."""

import os
import sys
import shutil
import cv2
import numpy as np

# Set offscreen platform before creating QApplication
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from storage.db import init_db
from storage.repository import insert_model, insert_session, get_captures_by_session
from adapter.mobilenet_v2_adapter import MobileNetV2Adapter
from capture.camera import CameraSession
from capture.capture_flow import CaptureFlow
from ui.capture_widget import CaptureWidget

TEST_DB_PATH = "test_ui.db"
TEST_CAPTURES_DIR = "test_ui_captures"
TEST_IMAGE_PATH = "sample/sample.jpg"
MODEL_PATH = "models/mobilenetv2-12.onnx"
LABELS_PATH = "models/imagenet_classes.txt"


def cleanup():
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass
    if os.path.exists(TEST_CAPTURES_DIR):
        try:
            shutil.rmtree(TEST_CAPTURES_DIR)
        except Exception:
            pass


def main():
    print("=" * 60)
    print("Running Headless PySide6 UI Logic Test")
    print("=" * 60)

    cleanup()

    try:
        app = QApplication.instance() or QApplication(sys.argv)

        # 1. Init DB & Session
        conn = init_db(TEST_DB_PATH)
        model_id = insert_model(
            conn, "MobileNetV2", "v2.0", MODEL_PATH, "QNN", "CPUExecutionProvider", "Fallback"
        )
        session_id = insert_session(conn, model_id, "UI Test Session")

        # 2. Init Adapter & Flow
        adapter = MobileNetV2Adapter()
        adapter.load(MODEL_PATH, LABELS_PATH)
        flow = CaptureFlow(adapter, conn, session_id, base_dir=TEST_CAPTURES_DIR)

        # 3. Create CaptureWidget with camera pointing to invalid index (testing graceful error handling)
        print("\n1. Testing CaptureWidget initialization with CameraNotFoundError handling...")
        bad_cam = CameraSession(device_index=999)
        widget = CaptureWidget(flow=flow, camera_session=bad_cam, model_name="MobileNetV2")

        # Verify in-UI error state
        assert "Camera Not Found" in widget.cam_status_lbl.text()
        assert widget.preview_widget._error_message is not None
        print("   [OK] Camera error handled gracefully in UI without crashing.")

        # 4. Inject a test frame directly into widget
        print("\n2. Injecting test camera frame & testing live prediction preview...")
        test_frame = cv2.imread(TEST_IMAGE_PATH)
        widget._latest_frame_bgr = test_frame
        widget.preview_widget.update_frame(test_frame)
        widget._run_inference_preview()

        print(f"   Prediction label: '{widget.pred_label_lbl.text()}'")
        print(f"   Confidence bar: {widget.conf_bar.value()}%")
        assert widget.conf_bar.value() > 50, "Expected confidence > 50%"
        print("   [OK] Prediction preview updated successfully.")

        # 5. Verify Confirm Button Gating
        print("\n3. Testing Confirm & Save button gating...")
        assert not widget.confirm_btn.isEnabled(), "Confirm button must be disabled when true_label is empty!"
        print("   [OK] Confirm button is disabled when ground-truth label is empty.")

        # Click 'Use Live Prediction'
        widget._copy_prediction_to_true_label()
        print(f"   Copied label into true_label input: '{widget.true_label_input.text()}'")
        assert widget.confirm_btn.isEnabled(), "Confirm button must now be enabled!"
        print("   [OK] Confirm button is enabled when ground-truth label is populated.")

        # 6. Execute Confirm & Save via UI button trigger
        print("\n4. Triggering Confirm & Save action...")
        widget.condition_combo.setCurrentText("background")
        widget.condition_note_input.setText("Simulated cluttered background")
        widget._on_confirm_and_save()

        # 7. Verify UI Summary Counter & DB State
        print("\n5. Verifying summary counters & DB records...")
        print(f"   UI Status message: {widget.status_msg_lbl.text()}")
        assert "Saved Capture" in widget.status_msg_lbl.text()

        # Check DB
        rows = get_captures_by_session(conn, session_id)
        assert len(rows) == 1, f"Expected 1 capture in DB, got {len(rows)}"
        print(f"   DB Record: #{rows[0]['id']}, Condition={rows[0]['condition_type']}, Correct={rows[0]['correct']}")
        print(f"   Summary Label: '{widget.summary_background_lbl.text()}'")
        assert "Background: <b>1</b>" in widget.summary_background_lbl.text()
        print("   [OK] Session summary counter dynamically refreshed.")

        # Clean widget
        widget.close()
        print("\n" + "=" * 60)
        print("ALL UI WIDGET TESTS PASSED")
        print("=" * 60)

    finally:
        conn.close()
        cleanup()


if __name__ == "__main__":
    main()
