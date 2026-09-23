"""Benchmarking utilities for model inference latency and end-to-end pipeline response time."""

import os
import time
import platform
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from adapter.base import ModelAdapter


def get_hardware_info() -> Dict[str, str]:
    """Collect host hardware and platform environment details."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }


def benchmark_model_inference(
    adapter: ModelAdapter,
    model_name: str = "Model",
    model_version: str = "v1.0",
    runs: int = 50,
    warmup: int = 10,
    input_shape: tuple = (1, 3, 224, 224),
) -> Dict[str, Any]:
    """Measure inference latency (prepared input tensor to raw output) after warm-up.

    Args:
        adapter: Loaded model adapter.
        model_name: Descriptive model name.
        model_version: Version identifier.
        runs: Number of timed runs.
        warmup: Number of warm-up iterations.
        input_shape: Input tensor shape.

    Returns:
        Dict with median, p95, mean, min, max latencies (in milliseconds) and execution provider metadata.
    """
    if adapter._session is None:
        raise RuntimeError("Model session is not loaded.")

    dummy_tensor = np.random.randn(*input_shape).astype(np.float32)
    input_name = adapter._input_name
    output_name = adapter._output_name

    # 1. Warm-up iterations
    for _ in range(warmup):
        adapter._session.run([output_name], {input_name: dummy_tensor})

    # 2. Timed benchmark runs
    latencies_ms: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        adapter._session.run([output_name], {input_name: dummy_tensor})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms.sort()
    median_ms = float(np.median(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    mean_ms = float(np.mean(latencies_ms))
    min_ms = float(np.min(latencies_ms))
    max_ms = float(np.max(latencies_ms))

    # Hardware & provider disclosure
    active_prov = getattr(adapter, "active_provider", "CPUExecutionProvider")
    provider_note = (
        "NPU/QNN not tested — target hardware not yet available; benchmark executed on host CPU."
        if "CPU" in active_prov
        else "Executed on requested hardware provider."
    )

    return {
        "model_name": model_name,
        "model_version": model_version,
        "provider_requested": "QNN",
        "provider_active": active_prov,
        "provider_note": provider_note,
        "runs": runs,
        "warmup": warmup,
        "latency_median_ms": round(median_ms, 2),
        "latency_p95_ms": round(p95_ms, 2),
        "latency_mean_ms": round(mean_ms, 2),
        "latency_min_ms": round(min_ms, 2),
        "latency_max_ms": round(max_ms, 2),
        "hardware": get_hardware_info(),
    }


def benchmark_end_to_end(
    adapter: ModelAdapter,
    image_paths: List[str],
    model_name: str = "Model",
    runs_per_image: int = 10,
) -> Dict[str, Any]:
    """Measure end-to-end response time (disk load + preprocess + inference + postprocess)

    across a set of saved capture image paths.
    """
    if not image_paths:
        return {"e2e_median_ms": 0.0, "e2e_p95_ms": 0.0, "total_samples": 0}

    valid_paths = [p for p in image_paths if os.path.exists(p)]
    if not valid_paths:
        raise FileNotFoundError("None of the specified image_paths exist on disk.")

    timings_ms: List[float] = []

    # Warmup
    sample_bgr = cv2.imread(valid_paths[0])
    sample_rgb = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB)
    for _ in range(3):
        adapter.predict(sample_rgb)

    for path in valid_paths:
        for _ in range(runs_per_image):
            t0 = time.perf_counter()
            bgr = cv2.imread(path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            adapter.predict(rgb)
            t1 = time.perf_counter()
            timings_ms.append((t1 - t0) * 1000.0)

    timings_ms.sort()
    return {
        "model_name": model_name,
        "total_samples": len(valid_paths),
        "total_timed_runs": len(timings_ms),
        "e2e_median_ms": round(float(np.median(timings_ms)), 2),
        "e2e_p95_ms": round(float(np.percentile(timings_ms, 95)), 2),
        "e2e_mean_ms": round(float(np.mean(timings_ms)), 2),
    }
