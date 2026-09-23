"""Headless validation test for Camera wrapper and CaptureFlow pipeline."""

import os
import shutil
import sqlite3
import cv2
import numpy as np
from pathlib import Path

from adapter.mobilenet_v2_adapter import MobileNetV2Adapter
from storage.db import init_db
from storage.repository import insert_model, insert_session, get_captures_by_session, get_session_summary
from capture.camera import CameraSession, CameraNotFoundError
from capture.capture_flow import CaptureFlow, LabelConfirmationError

TEST_DB_PATH = "test_capture_flow.db"
TEST_CAPTURES_DIR = "test_captures"
TEST_IMAGE_PATH = "sample/sample.jpg"
MODEL_PATH = "models/mobilenetv2-12.onnx"
LABELS_PATH = "models/imagenet_classes.txt"


def cleanup():
    """Remove test artifacts."""
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
    print("=" * 65)
    print("ShiftProof Capture Module - Headless Flow Test")
    print("=" * 65)

    cleanup()

    try:
        # 1. Test CameraSession error handling when no camera is found
        print("\n1. Testing CameraSession graceful error handling (index 999)...")
        cam = CameraSession(device_index=999)
        try:
            cam.open()
            print("   [WARN] Unexpected: Device 999 opened.")
            cam.close()
        except CameraNotFoundError as e:
            print("   [OK] CameraNotFoundError caught successfully as expected:")
            print(f"        Message: {e}")

        # 2. Initialize Database and create Model & Session
        print("\n2. Initializing SQLite Database & Test Session...")
        conn = init_db(TEST_DB_PATH)
        model_id = insert_model(
            conn=conn,
            name="MobileNetV2",
            version="v2.0",
            path=MODEL_PATH,
            provider_requested="QNN",
            provider_active="CPUExecutionProvider",
            provider_note="Test execution provider",
        )
        session_id = insert_session(
            conn=conn,
            model_id=model_id,
            label="Headless Capture Flow Test Run",
        )
        print(f"   [OK] Created Model ID: {model_id}, Session ID: {session_id}")

        # 3. Load Model Adapter
        print("\n3. Loading MobileNetV2 Adapter...")
        adapter = MobileNetV2Adapter()
        adapter.load(model_path=MODEL_PATH, labels_path=LABELS_PATH)
        print(f"   [OK] Adapter loaded successfully (active provider: {adapter.active_provider})")

        # 4. Load static test image (OpenCV loads BGR numpy array)
        print(f"\n4. Loading static test frame from '{TEST_IMAGE_PATH}'...")
        frame_bgr = cv2.imread(TEST_IMAGE_PATH)
        if frame_bgr is None:
            raise FileNotFoundError(f"Test image not found at {TEST_IMAGE_PATH}")
        print(f"   [OK] Frame loaded: shape={frame_bgr.shape}, dtype={frame_bgr.dtype} (BGR format)")

        # 5. Initialize CaptureFlow
        flow = CaptureFlow(
            adapter=adapter,
            conn=conn,
            session_id=session_id,
            base_dir=TEST_CAPTURES_DIR,
        )

        # 6. Live Prediction Preview
        print("\n5. Running live prediction preview on BGR frame...")
        preview = flow.preview_prediction(frame_bgr)
        print(f"   Preview result: Predicted='{preview['predicted_label']}', Confidence={preview['confidence']:.4f}")
        assert preview["predicted_label"] == "Samoyed", f"Expected Samoyed, got {preview['predicted_label']}"
        print("   [OK] Live prediction preview returned expected label.")

        # 7. Test Label Confirmation Enforcement
        print("\n6. Testing mandatory label confirmation constraint...")
        try:
            flow.confirm_and_save(
                frame_bgr=frame_bgr,
                true_label="",  # Missing confirmation!
                condition_type="baseline",
            )
            raise AssertionError("Should have raised LabelConfirmationError for empty true_label!")
        except LabelConfirmationError as e:
            print(f"   [OK] Blocked unconfirmed save with exception: {e}")

        # 8. Confirm & Save Valid Capture 1 (baseline condition)
        print("\n7. Executing confirm_and_save with confirmed true_label='Samoyed', condition='baseline'...")
        cap1 = flow.confirm_and_save(
            frame_bgr=frame_bgr,
            true_label="Samoyed",
            condition_type="baseline",
            condition_note="Clean background, neutral lighting",
            predicted_label=preview["predicted_label"],
            confidence=preview["confidence"],
        )
        print(f"   [OK] Capture #1 saved with ID: {cap1['id']}")
        print(f"        Image path: {cap1['image_path']}, Correct: {cap1['correct']}")

        # 9. Confirm & Save Valid Capture 2 (lighting condition with deliberate mismatch test)
        print("\n8. Executing confirm_and_save with condition='lighting'...")
        cap2 = flow.confirm_and_save(
            frame_bgr=frame_bgr,
            true_label="Golden Retriever",  # Intentionally different ground truth
            condition_type="lighting",
            condition_note="Low light simulation test",
            predicted_label=preview["predicted_label"],
            confidence=preview["confidence"],
        )
        print(f"   [OK] Capture #2 saved with ID: {cap2['id']}")
        print(f"        Image path: {cap2['image_path']}, Correct: {cap2['correct']}")

        # 10. Verify File System Artifacts
        print("\n9. Verifying image files on disk...")
        file1_path = Path(TEST_CAPTURES_DIR) / str(session_id) / f"{cap1['id']}.jpg"
        file2_path = Path(TEST_CAPTURES_DIR) / str(session_id) / f"{cap2['id']}.jpg"

        print(f"   Checking: {file1_path} -> Exists? {file1_path.exists()} ({file1_path.stat().st_size} bytes)")
        print(f"   Checking: {file2_path} -> Exists? {file2_path.exists()} ({file2_path.stat().st_size} bytes)")
        assert file1_path.exists() and file1_path.stat().st_size > 0, "Capture 1 file missing!"
        assert file2_path.exists() and file2_path.stat().st_size > 0, "Capture 2 file missing!"
        print("   [OK] All capture image files successfully saved to disk.")

        # 11. Verify Database Records
        print("\n10. Verifying Database Rows in SQLite...")
        rows = get_captures_by_session(conn, session_id)
        assert len(rows) == 2, f"Expected 2 rows in DB, got {len(rows)}"
        for row in rows:
            print(f"    - DB Record #{row['id']}: Session={row['session_id']}, Condition={row['condition_type']}, "
                  f"True='{row['true_label']}', Pred='{row['predicted_label']}', Correct={row['correct']}, Path='{row['image_path']}'")

        summary = get_session_summary(conn, session_id)
        print("\n11. Session Summary:")
        print(f"    {summary}")
        assert summary.get("baseline") == 1
        assert summary.get("lighting") == 1

        print("\n" + "=" * 65)
        print("ALL CAPTURE FLOW TESTS PASSED SUCCESSFULLY")
        print("=" * 65)

    finally:
        conn.close()
        cleanup()


if __name__ == "__main__":
    main()
