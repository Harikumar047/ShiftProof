"""Evaluation scoring engine for evaluating model adapters against saved captures."""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from adapter.base import ModelAdapter


def score_model(
    captures: List[Dict[str, Any]],
    adapter: ModelAdapter,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """Evaluate a single model adapter across a list of saved capture records.

    Args:
        captures: List of capture dictionaries (from storage.repository).
        adapter: Model adapter implementing :class:`ModelAdapter`.
        model_name: Descriptive name for the model.

    Returns:
        Dict containing:
            - model_name: str
            - overall_accuracy: float
            - total_captures: int
            - total_correct: int
            - condition_accuracy: Dict[str, float]
            - condition_counts: Dict[str, Dict[str, int]]
            - per_capture_results: List[Dict[str, Any]]
    """
    if not captures:
        return {
            "model_name": model_name,
            "overall_accuracy": 0.0,
            "total_captures": 0,
            "total_correct": 0,
            "condition_accuracy": {},
            "condition_counts": {},
            "per_capture_results": [],
        }

    per_capture_results = []
    condition_stats: Dict[str, Dict[str, int]] = {}
    total_correct = 0

    for cap in captures:
        img_path = cap.get("image_path", "")
        true_label = str(cap.get("true_label", "")).strip()
        condition_type = cap.get("condition_type", "unknown")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Capture image not found on disk: {img_path}")

        # Load image as BGR and convert to RGB for model inference
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise ValueError(f"Failed to decode image at: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        pred = adapter.predict(rgb)
        predicted_label = str(pred.get("label", "")).strip()
        confidence = float(pred.get("confidence", 0.0))

        correct = 1 if (true_label.lower() == predicted_label.lower()) else 0
        if correct:
            total_correct += 1

        if condition_type not in condition_stats:
            condition_stats[condition_type] = {"total": 0, "correct": 0}
        condition_stats[condition_type]["total"] += 1
        if correct:
            condition_stats[condition_type]["correct"] += 1

        per_capture_results.append({
            "capture_id": cap.get("id"),
            "image_path": img_path,
            "true_label": true_label,
            "condition_type": condition_type,
            "condition_note": cap.get("condition_note"),
            "predicted_label": predicted_label,
            "confidence": confidence,
            "correct": correct,
        })

    # Calculate per-condition accuracies
    condition_accuracy: Dict[str, float] = {}
    for cond, stats in condition_stats.items():
        condition_accuracy[cond] = (
            stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        )

    overall_acc = total_correct / len(captures) if captures else 0.0

    return {
        "model_name": model_name,
        "overall_accuracy": float(overall_acc),
        "total_captures": len(captures),
        "total_correct": total_correct,
        "condition_accuracy": condition_accuracy,
        "condition_counts": condition_stats,
        "per_capture_results": per_capture_results,
    }
