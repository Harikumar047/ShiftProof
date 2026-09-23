"""Standalone test script for ShiftProof storage module."""

import os
import json
from pathlib import Path
from storage.db import init_db
from storage.repository import (
    insert_model,
    insert_session,
    insert_capture,
    get_model,
    get_session,
    get_captures_by_session,
    get_captures_by_condition,
    get_session_summary,
)
from storage.export import export_session_to_json, delete_session

TEST_DB_PATH = "test_storage.db"
EXPORT_JSON_PATH = "test_export_session.json"


def cleanup():
    for f in [TEST_DB_PATH, EXPORT_JSON_PATH]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass


def main():
    print("=" * 60)
    print("ShiftProof Storage Module Test")
    print("=" * 60)

    cleanup()

    try:
        # 1. Initialize fresh test DB
        print("\n1. Initializing fresh test DB at:", TEST_DB_PATH)
        conn = init_db(TEST_DB_PATH)
        print("   [OK] Database initialized with schema tables: models, sessions, captures.")

        # 2. Insert fake model
        print("\n2. Inserting fake model...")
        model_id = insert_model(
            conn=conn,
            name="MobileNetV2",
            version="v2.0-quant",
            path="models/mobilenetv2-7.onnx",
            provider_requested="QNN",
            provider_active="CPUExecutionProvider",
            provider_note="QNN unavailable on this host; fallback to CPU",
        )
        print(f"   [OK] Inserted model ID: {model_id}")
        model = get_model(conn, model_id)
        print(f"   Model Details: name={model['name']}, version={model['version']}, provider_active={model['provider_active']}")

        # 3. Insert fake session
        print("\n3. Inserting fake session...")
        session_id = insert_session(
            conn=conn,
            model_id=model_id,
            label="baseline demo run",
        )
        print(f"   [OK] Inserted session ID: {session_id}")
        session = get_session(conn, session_id)
        print(f"   Session Details: id={session['id']}, label='{session['label']}', model_id={session['model_id']}")

        # 4. Insert three fake captures (one per condition type)
        print("\n4. Inserting three fake captures (baseline, background, lighting)...")
        c1 = insert_capture(
            conn=conn,
            session_id=session_id,
            image_path="sample/samoyed_baseline.jpg",
            true_label="Samoyed",
            condition_type="baseline",
            condition_note="Standard indoor lighting, centered subject",
            predicted_label="Samoyed",
            confidence=0.8595,
            correct=None,  # Derived automatically (1)
        )
        c2 = insert_capture(
            conn=conn,
            session_id=session_id,
            image_path="sample/samoyed_busy_bg.jpg",
            true_label="Samoyed",
            condition_type="background",
            condition_note="Outdoor park with cluttered foliage background",
            predicted_label="Samoyed",
            confidence=0.7420,
            correct=None,  # Derived automatically (1)
        )
        c3 = insert_capture(
            conn=conn,
            session_id=session_id,
            image_path="sample/samoyed_low_light.jpg",
            true_label="Samoyed",
            condition_type="lighting",
            condition_note="Low-light / glare shadow test",
            predicted_label="Pomeranian",
            confidence=0.4810,
            correct=None,  # Derived automatically (0)
        )
        print(f"   [OK] Inserted capture IDs: {c1}, {c2}, {c3}")

        # 5. Read back captures and print summary
        print("\n5. Reading back captures & checking queries...")
        all_captures = get_captures_by_session(conn, session_id)
        print(f"   Total captures in session {session_id}: {len(all_captures)}")
        for cap in all_captures:
            print(f"   - Capture #{cap['id']} [{cap['condition_type']}]: True='{cap['true_label']}', "
                  f"Pred='{cap['predicted_label']}', Conf={cap['confidence']:.4f}, Correct={cap['correct']}")

        bg_captures = get_captures_by_condition(conn, "background", session_id=session_id)
        print(f"   Captures under condition 'background': {len(bg_captures)}")

        # Print session summary
        summary = get_session_summary(conn, session_id)
        print("\n6. Session Summary (counts per condition_type):")
        print("   " + json.dumps(summary, indent=4))

        # 7. Test Export to JSON
        print("\n7. Testing JSON Export...")
        export_data = export_session_to_json(conn, session_id, out_path=EXPORT_JSON_PATH)
        print(f"   [OK] Exported session {session_id} to '{EXPORT_JSON_PATH}' ({len(export_data['captures'])} captures)")

        # 8. Delete session and verify it is gone
        print("\n8. Deleting session...")
        deleted = delete_session(conn, session_id)
        print(f"   Delete success: {deleted}")

        # Verification checks
        remaining_session = get_session(conn, session_id)
        remaining_captures = get_captures_by_session(conn, session_id)
        print(f"   Session exists in DB? {remaining_session is not None}")
        print(f"   Associated captures in DB? {len(remaining_captures)} remaining")

        assert remaining_session is None, "Session should be deleted!"
        assert len(remaining_captures) == 0, "Captures should be deleted!"
        print("   [OK] Confirmed session and associated captures are completely removed.")

        print("\n" + "=" * 60)
        print("ALL STORAGE TESTS PASSED SUCCESSFULLY")
        print("=" * 60)

    finally:
        conn.close()
        cleanup()


if __name__ == "__main__":
    main()
