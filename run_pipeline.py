"""End-to-end pipeline runner for ShiftProof comparison, benchmarking, and HTML reporting."""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

from storage.db import init_db
from storage.repository import (
    insert_model,
    insert_session,
    insert_capture,
    get_session,
    get_captures_by_session,
    get_latest_session_with_captures,
)
from adapter.mobilenet_v2_adapter import MobileNetV2Adapter
from adapter.squeezenet_adapter import SqueezeNetAdapter
from evaluation.compare import compare_models
from benchmark.latency import benchmark_model_inference, benchmark_end_to_end
from reporting.report_generator import generate_html_report

MOBILENET_PATH = "models/mobilenetv2-12.onnx"
SQUEEZENET_PATH = "models/squeezenet1.1-7.onnx"
LABELS_PATH = "models/imagenet_classes.txt"
SAMPLE_IMG_PATH = "sample/sample.jpg"


def ensure_sample_captures(conn, session_id: int):
    """Seed synthetic condition captures if and only if no real captures exist (demo fallback)."""
    captures = get_captures_by_session(conn, session_id)
    if captures:
        return captures

    os.makedirs(f"captures/{session_id}", exist_ok=True)
    base_img = cv2.imread(SAMPLE_IMG_PATH)
    if base_img is None:
        raise FileNotFoundError(f"Sample image not found at {SAMPLE_IMG_PATH}")

    # 1. Baseline
    p1 = f"captures/{session_id}/1_baseline.jpg"
    cv2.imwrite(p1, base_img)
    insert_capture(conn, session_id, p1, "Samoyed", "baseline", "Neutral indoor lighting, centered")

    # 2. Background shift
    p2 = f"captures/{session_id}/2_background.jpg"
    bg_img = cv2.GaussianBlur(base_img, (15, 15), 0)
    cv2.imwrite(p2, bg_img)
    insert_capture(conn, session_id, p2, "Samoyed", "background", "Simulated cluttered background")

    # 3. Lighting shift
    p3 = f"captures/{session_id}/3_lighting.jpg"
    dark_img = np.clip(base_img.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
    cv2.imwrite(p3, dark_img)
    insert_capture(conn, session_id, p3, "Samoyed", "lighting", "Low-light / underexposed shadow")

    # 4. Distance shift
    p4 = f"captures/{session_id}/4_distance.jpg"
    h, w, _ = base_img.shape
    dist_img = cv2.resize(base_img, (w // 3, h // 3))
    canvas = np.zeros_like(base_img)
    y_off = (h - dist_img.shape[0]) // 2
    x_off = (w - dist_img.shape[1]) // 2
    canvas[y_off:y_off + dist_img.shape[0], x_off:x_off + dist_img.shape[1]] = dist_img
    cv2.imwrite(p4, canvas)
    insert_capture(conn, session_id, p4, "Samoyed", "distance", "Distant subject with heavy canvas padding")

    return get_captures_by_session(conn, session_id)


def main():
    parser = argparse.ArgumentParser(
        description="ShiftProof Comparative Evaluation & HTML Shift Report Pipeline"
    )
    parser.add_argument(
        "--session-id",
        type=int,
        default=None,
        help="Target session ID to evaluate. If omitted, defaults to the most recent session with captures.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="shiftproof.db",
        help="Path to the SQLite database file (default: shiftproof.db).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path for the generated HTML report file.",
    )
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Force generation of synthetic sample captures even if real sessions exist.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ShiftProof End-to-End Evaluation, Benchmark & Report Pipeline")
    print("=" * 70)

    # 1. Connect to database and display absolute path
    abs_db_path = str(Path(args.db_path).resolve())
    print(f"\n1. Connecting to SQLite Database at: '{abs_db_path}'")
    conn = init_db(args.db_path)

    # 2. Resolve Session and Captures
    session_id = args.session_id
    captures = []
    session_info = None

    if session_id is not None:
        # User specified an explicit session ID
        session_info = get_session(conn, session_id)
        if not session_info:
            print(f"[!] Error: Session ID #{session_id} not found in '{abs_db_path}'.")
            sys.exit(1)
        captures = get_captures_by_session(conn, session_id)
        print(f"   [OK] Selected explicit Session #{session_id} ('{session_info.get('label')}')")
        if not captures:
            if args.force_synthetic:
                print("   [!] No captures in session. Generating synthetic demo captures due to --force-synthetic flag...")
                captures = ensure_sample_captures(conn, session_id)
            else:
                print(f"[!] Error: Session #{session_id} exists but has 0 captures.")
                print("    Capture images via the GUI first, or run without --session-id to auto-select active sessions.")
                sys.exit(1)
        else:
            print(f"   [OK] Loaded {len(captures)} real capture(s) from Session #{session_id}.")
    else:
        # Auto-select the most recent session that contains captures
        latest_active = get_latest_session_with_captures(conn)
        if latest_active and not args.force_synthetic:
            session_id = latest_active["id"]
            session_info = latest_active
            captures = get_captures_by_session(conn, session_id)
            print(f"   [OK] Auto-detected latest active Session #{session_id} ('{latest_active.get('label')}') "
                  f"with {len(captures)} existing capture(s).")
        else:
            # Fallback: create a demo session and seed synthetic condition captures
            print("   [*] No existing sessions with captures found (or --force-synthetic requested).")
            model_id = insert_model(
                conn=conn,
                name="MobileNetV2",
                version="v2.0-quant",
                path=MOBILENET_PATH,
                provider_requested="QNN",
                provider_active="CPUExecutionProvider",
                provider_note="Demo fallback execution",
            )
            session_id = insert_session(
                conn, model_id=model_id, label="ShiftProof Full Evaluation Demo Session"
            )
            session_info = get_session(conn, session_id)
            captures = ensure_sample_captures(conn, session_id)
            print(f"   [OK] Initialized Demo Session #{session_id} with {len(captures)} synthetic condition captures.")

    print(f"\n2. Active captures to evaluate in Session #{session_id}:")
    for c in captures:
        print(f"   - Capture #{c['id']} [{c['condition_type']}]: True='{c['true_label']}', "
              f"Path='{c.get('image_path')}', Note='{c.get('condition_note')}'")

    # 3. Load Model Adapters
    print("\n3. Loading Model Adapters (MobileNetV2 vs SqueezeNet 1.1)...")
    if not os.path.exists(MOBILENET_PATH) or not os.path.exists(SQUEEZENET_PATH) or not os.path.exists(LABELS_PATH):
        print(f"[!] Error: Model or labels missing in 'models/' directory.")
        sys.exit(1)

    mobilenet = MobileNetV2Adapter()
    mobilenet.load(MOBILENET_PATH, LABELS_PATH)

    squeezenet = SqueezeNetAdapter()
    squeezenet.load(SQUEEZENET_PATH, LABELS_PATH)
    print("   [OK] MobileNetV2 (Base) loaded.")
    print("   [OK] SqueezeNet 1.1 (Alternative) loaded.")

    # 4. Comparative Evaluation on the SAME captures
    print("\n4. Running Comparative Evaluation on identical capture images...")
    comparison = compare_models(
        captures=captures,
        model_a=mobilenet,
        model_b=squeezenet,
        model_a_name="MobileNetV2 (Base)",
        model_b_name="SqueezeNet 1.1 (Alternative)",
    )

    print(f"   Model A (MobileNetV2) Accuracy: {comparison['model_a']['overall_accuracy']:.1%}")
    print(f"   Model B (SqueezeNet)  Accuracy: {comparison['model_b']['overall_accuracy']:.1%}")
    print(f"   Prediction Shift Rate: {comparison['prediction_change_rate']:.1%} ({comparison['prediction_changes']} changes)")
    print(f"   Agreement Rate: {comparison['agreement_rate']:.1%}")

    print("\n   Condition-Level Accuracy & Shift Matrix:")
    for cond, data in comparison["condition_comparison"].items():
        delta = data["accuracy_delta"]
        sign = "+" if delta > 0 else ""
        print(f"   - {cond.capitalize():<12} (n={data['total_count']}): "
              f"Model A={data['model_a_accuracy']:.1%} | Model B={data['model_b_accuracy']:.1%} | Delta={sign}{delta:.1%}")

    # 5. Benchmarking Latency & E2E
    print("\n5. Running Latency & End-to-End Response Benchmarks...")
    bench_a = benchmark_model_inference(mobilenet, model_name="MobileNetV2", runs=30, warmup=10)
    bench_b = benchmark_model_inference(squeezenet, model_name="SqueezeNet 1.1", runs=30, warmup=10)

    img_paths = [c["image_path"] for c in captures if os.path.exists(c.get("image_path", ""))]
    e2e_a = benchmark_end_to_end(mobilenet, img_paths, model_name="MobileNetV2", runs_per_image=5) if img_paths else {}
    e2e_b = benchmark_end_to_end(squeezenet, img_paths, model_name="SqueezeNet 1.1", runs_per_image=5) if img_paths else {}

    print(f"   MobileNetV2  -> Latency Median: {bench_a['latency_median_ms']} ms (p95: {bench_a['latency_p95_ms']} ms) | E2E: {e2e_a.get('e2e_median_ms', 'N/A')} ms")
    print(f"   SqueezeNet   -> Latency Median: {bench_b['latency_median_ms']} ms (p95: {bench_b['latency_p95_ms']} ms) | E2E: {e2e_b.get('e2e_median_ms', 'N/A')} ms")
    print(f"   Provider Disclosures: {bench_a['provider_note']}")

    benchmarks = {
        "model_a": bench_a,
        "model_b": bench_b,
        "e2e_a": e2e_a,
        "e2e_b": e2e_b,
    }

    # 6. Generate Self-Contained HTML Report
    report_output_path = (
        args.output or f"reports/session_{session_id}_report.html"
    )
    print(f"\n6. Generating self-contained HTML report at '{report_output_path}'...")
    report_file = generate_html_report(
        session=session_info or {"id": session_id, "label": f"Session #{session_id}"},
        comparison_results=comparison,
        benchmark_results=benchmarks,
        output_path=report_output_path,
    )

    print(f"   [OK] HTML Report generated successfully: {report_file} ({os.path.getsize(report_file)} bytes)")
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 70)

    conn.close()
    return report_file


if __name__ == "__main__":
    main()
