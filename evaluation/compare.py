"""Model comparison engine for comparing two models across identical saved captures."""

from typing import List, Dict, Any, Optional
from adapter.base import ModelAdapter
from evaluation.scoring import score_model


def compare_models(
    captures: List[Dict[str, Any]],
    model_a: ModelAdapter,
    model_b: ModelAdapter,
    model_a_name: str = "MobileNetV2 (Base)",
    model_b_name: str = "SqueezeNet (Alternative)",
) -> Dict[str, Any]:
    """Run both models against the identical set of saved capture images (never re-capturing).

    Args:
        captures: List of saved capture records.
        model_a: First model adapter.
        model_b: Second model adapter.
        model_a_name: Display name for Model A.
        model_b_name: Display name for Model B.

    Returns:
        Dict containing per-model evaluation scores, condition-level comparisons,
        prediction change rate, and paired capture comparisons.
    """
    if not captures:
        return {
            "model_a": score_model([], model_a, model_a_name),
            "model_b": score_model([], model_b, model_b_name),
            "total_captures": 0,
            "prediction_changes": 0,
            "prediction_change_rate": 0.0,
            "agreement_rate": 1.0,
            "condition_comparison": {},
            "paired_captures": [],
        }

    score_a = score_model(captures, model_a, model_a_name)
    score_b = score_model(captures, model_b, model_b_name)

    results_a = {r["capture_id"]: r for r in score_a["per_capture_results"]}
    results_b = {r["capture_id"]: r for r in score_b["per_capture_results"]}

    prediction_changes = 0
    paired_captures = []

    # Map conditions
    all_conditions = sorted(
        list(
            set(score_a["condition_accuracy"].keys()).union(
                score_b["condition_accuracy"].keys()
            )
        )
    )

    for cap in captures:
        cid = cap.get("id")
        ra = results_a.get(cid, {})
        rb = results_b.get(cid, {})

        pred_a = ra.get("predicted_label", "")
        pred_b = rb.get("predicted_label", "")
        changed = (pred_a.lower() != pred_b.lower())
        if changed:
            prediction_changes += 1

        correct_a = ra.get("correct", 0)
        correct_b = rb.get("correct", 0)

        if correct_a and correct_b:
            outcome = "both_correct"
        elif not correct_a and not correct_b:
            outcome = "both_incorrect"
        elif correct_a and not correct_b:
            outcome = "a_only_correct"
        else:
            outcome = "b_only_correct"

        paired_captures.append({
            "capture_id": cid,
            "image_path": cap.get("image_path"),
            "true_label": cap.get("true_label"),
            "condition_type": cap.get("condition_type"),
            "condition_note": cap.get("condition_note"),
            "model_a": {
                "name": model_a_name,
                "predicted_label": pred_a,
                "confidence": ra.get("confidence", 0.0),
                "correct": correct_a,
            },
            "model_b": {
                "name": model_b_name,
                "predicted_label": pred_b,
                "confidence": rb.get("confidence", 0.0),
                "correct": correct_b,
            },
            "prediction_changed": changed,
            "outcome": outcome,
        })

    change_rate = prediction_changes / len(captures) if captures else 0.0
    agreement_rate = 1.0 - change_rate

    # Condition-level comparison matrix
    condition_comparison = {}
    for cond in all_conditions:
        acc_a = score_a["condition_accuracy"].get(cond, 0.0)
        acc_b = score_b["condition_accuracy"].get(cond, 0.0)
        cnt = score_a["condition_counts"].get(cond, {}).get("total", 0)
        condition_comparison[cond] = {
            "total_count": cnt,
            "model_a_accuracy": acc_a,
            "model_b_accuracy": acc_b,
            "accuracy_delta": acc_b - acc_a,
        }

    return {
        "model_a": score_a,
        "model_b": score_b,
        "total_captures": len(captures),
        "prediction_changes": prediction_changes,
        "prediction_change_rate": float(change_rate),
        "agreement_rate": float(agreement_rate),
        "condition_comparison": condition_comparison,
        "paired_captures": paired_captures,
    }
