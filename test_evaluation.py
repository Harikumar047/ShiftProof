"""Test script for ShiftProof evaluation & comparison engine."""

import os
import shutil
import cv2
import numpy as np

from storage.db import init_db
from storage.repository import insert_model, insert_session, insert_capture, get_captures_by_session
from adapter.mobilenet_v2_adapter import MobileNetV2Adapter
from adapter.squeezenet_adapter import SqueezeNetAdapter
from evaluation.scoring import score_model
from evaluation.compare import compare_models

TEST_DB_PATH = "test_eval.db"
TEST_DIR = "test_eval_captures"
SAMPLE_IMG = "sample/sample.jpg"
MOBILENET_PATH = "models/mobilenetv2-12.onnx"
SQUEEZENET_PATH = "models/squeezenet1.1-7.onnx"
LABELS_PATH = "models/imagenet_classes.txt"


def cleanup():
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass
    if os.path.exists(TEST_DIR):
        try:
            shutil.rmtree(TEST_DIR)
        except Exception:
            pass


def main():
    print("=" * 65)
    print("ShiftProof Evaluation & Comparison Engine Test")
    print("=" * 65)

    cleanup()
    os.makedirs(TEST_DIR, exist_ok=True)

    try:
        # 1. Setup DB and Captures
        conn = init_db(TEST_DB_PATH)
        model_id = insert_model(conn, "MobileNetV2", "v2", MOBILENET_PATH, "QNN", "CPUExecutionProvider")
        session_id = insert_session(conn, model_id, "Evaluation Test Session")

        # Create 3 test capture image files
        img = cv2.imread(SAMPLE_IMG)
        c1_path = os.path.join(TEST_DIR, "cap_baseline.jpg")
        c2_path = os.path.join(TEST_DIR, "cap_lighting.jpg")
        c3_path = os.path.join(TEST_DIR, "cap_background.jpg")

        cv2.imwrite(c1_path, img)
        cv2.imwrite(c2_path, (img * 0.5).astype(np.uint8))  # simulated darker lighting
        cv2.imwrite(c3_path, cv2.flip(img, 1))  # flipped background

        insert_capture(conn, session_id, c1_path, "Samoyed", "baseline", "Studio baseline")
        insert_capture(conn, session_id, c2_path, "Samoyed", "lighting", "Dimmed lighting")
        insert_capture(conn, session_id, c3_path, "Samoyed", "background", "Modified background")

        captures = get_captures_by_session(conn, session_id)
        print(f"\n1. Loaded {len(captures)} test captures from database.")

        # 2. Load Model Adapters
        print("\n2. Loading Model Adapters (MobileNetV2 vs SqueezeNet)...")
        mobilenet = MobileNetV2Adapter()
        mobilenet.load(MOBILENET_PATH, LABELS_PATH)

        squeezenet = SqueezeNetAdapter()
        squeezenet.load(SQUEEZENET_PATH, LABELS_PATH)
        print("   [OK] Both adapters loaded successfully.")

        # 3. Test Single Model Scoring
        print("\n3. Testing Single Model Scoring (MobileNetV2)...")
        score_a = score_model(captures, mobilenet, model_name="MobileNetV2")
        print(f"   Overall Accuracy: {score_a['overall_accuracy']:.2%} ({score_a['total_correct']}/{score_a['total_captures']})")
        print(f"   Condition Accuracies: {score_a['condition_accuracy']}")

        # 4. Test Model Comparison
        print("\n4. Testing Comparison Engine (MobileNetV2 vs SqueezeNet)...")
        comparison = compare_models(
            captures=captures,
            model_a=mobilenet,
            model_b=squeezenet,
            model_a_name="MobileNetV2",
            model_b_name="SqueezeNet 1.1",
        )

        print(f"   Model A (MobileNetV2) Accuracy: {comparison['model_a']['overall_accuracy']:.2%}")
        print(f"   Model B (SqueezeNet)  Accuracy: {comparison['model_b']['overall_accuracy']:.2%}")
        print(f"   Prediction Changes: {comparison['prediction_changes']}/{comparison['total_captures']} ({comparison['prediction_change_rate']:.2%})")
        print(f"   Agreement Rate: {comparison['agreement_rate']:.2%}")

        print("\n5. Paired Captures Analysis:")
        for p in comparison["paired_captures"]:
            print(f"   - Capture #{p['capture_id']} [{p['condition_type']}]: True='{p['true_label']}' | "
                  f"A: '{p['model_a']['predicted_label']}' ({p['model_a']['confidence']:.2f}, corr={p['model_a']['correct']}) | "
                  f"B: '{p['model_b']['predicted_label']}' ({p['model_b']['confidence']:.2f}, corr={p['model_b']['correct']}) | "
                  f"Outcome: {p['outcome']}")

        print("\n6. Condition-Level Comparison Matrix:")
        for cond, stats in comparison["condition_comparison"].items():
            print(f"   - {cond.capitalize():<12}: Model A={stats['model_a_accuracy']:.1%}, "
                  f"Model B={stats['model_b_accuracy']:.1%}, Delta={stats['accuracy_delta']:+.1%}")

        print("\n" + "=" * 65)
        print("ALL EVALUATION TESTS PASSED")
        print("=" * 65)

    finally:
        conn.close()
        cleanup()


if __name__ == "__main__":
    main()
