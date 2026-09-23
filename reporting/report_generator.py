"""Self-contained HTML Report Generator for ShiftProof Shift Testing and Model Comparison."""

import os
import base64
import json
from pathlib import Path
from typing import Dict, Any, Optional


def _image_to_base64(image_path: str) -> str:
    """Read image file and encode as base64 data URI for standalone HTML embedding."""
    if not image_path or not os.path.exists(image_path):
        return ""
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower().replace(".", "")
        mime = "jpeg" if ext in ["jpg", "jpeg"] else ext
        return f"data:image/{mime};base64,{encoded}"
    except Exception:
        return ""


def generate_html_report(
    session: Dict[str, Any],
    comparison_results: Dict[str, Any],
    benchmark_results: Dict[str, Any],
    output_path: str = "reports/shiftproof_report.html",
) -> str:
    """Generate a self-contained HTML evaluation report.

    Args:
        session: Session metadata dict.
        comparison_results: Evaluation output from :func:`evaluation.compare.compare_models`.
        benchmark_results: Latency & benchmark dict from :mod:`benchmark.latency`.
        output_path: Destination filepath for the generated HTML.

    Returns:
        str: Absolute path of the generated HTML file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model_a_data = comparison_results.get("model_a", {})
    model_b_data = comparison_results.get("model_b", {})
    model_a_name = model_a_data.get("model_name", "Model A")
    model_b_name = model_b_data.get("model_name", "Model B")

    total_caps = comparison_results.get("total_captures", 0)
    change_rate = comparison_results.get("prediction_change_rate", 0.0)
    agreement_rate = comparison_results.get("agreement_rate", 1.0)
    changes_count = comparison_results.get("prediction_changes", 0)

    acc_a = model_a_data.get("overall_accuracy", 0.0)
    acc_b = model_b_data.get("overall_accuracy", 0.0)

    # Condition Comparison Table rows
    cond_rows_html = ""
    for cond, data in comparison_results.get("condition_comparison", {}).items():
        delta = data["accuracy_delta"]
        delta_color = "#38a169" if delta > 0 else ("#e53e3e" if delta < 0 else "#a0aec0")
        delta_sign = "+" if delta > 0 else ""
        cond_rows_html += f"""
        <tr>
            <td style="font-weight: 600; text-transform: capitalize;">{cond}</td>
            <td style="text-align: center;">{data['total_count']}</td>
            <td style="text-align: center;">{data['model_a_accuracy']:.1%}</td>
            <td style="text-align: center;">{data['model_b_accuracy']:.1%}</td>
            <td style="text-align: center; color: {delta_color}; font-weight: bold;">{delta_sign}{delta:.1%}</td>
        </tr>
        """

    # Paired Captures Gallery
    paired_cards_html = ""
    for p in comparison_results.get("paired_captures", []):
        img_uri = _image_to_base64(p.get("image_path", ""))
        a_info = p.get("model_a", {})
        b_info = p.get("model_b", {})

        a_corr = a_info.get("correct", 0) == 1
        b_corr = b_info.get("correct", 0) == 1
        changed = p.get("prediction_changed", False)

        change_badge = (
            '<span class="badge badge-warning">Prediction Shifted</span>'
            if changed
            else '<span class="badge badge-success">Consistent</span>'
        )

        a_badge = (
            f'<span class="badge badge-success">Correct</span>'
            if a_corr
            else '<span class="badge badge-danger">Mismatch</span>'
        )
        b_badge = (
            f'<span class="badge badge-success">Correct</span>'
            if b_corr
            else '<span class="badge badge-danger">Mismatch</span>'
        )

        img_tag = (
            f'<img src="{img_uri}" alt="Capture #{p.get("capture_id")}" class="capture-thumb" />'
            if img_uri
            else '<div class="no-img">No Image File</div>'
        )

        paired_cards_html += f"""
        <div class="card capture-card">
            <div class="capture-flex">
                <div class="capture-img-col">
                    {img_tag}
                    <div class="capture-meta">
                        <b>ID #{p.get('capture_id')}</b> &bull; <span class="tag">{p.get('condition_type')}</span>
                    </div>
                </div>
                <div class="capture-details-col">
                    <div class="true-label-box">
                        <span class="label-heading">Ground-Truth Label:</span>
                        <span class="true-label-text">{p.get('true_label')}</span>
                        {change_badge}
                    </div>
                    {f'<div class="condition-note">Note: {p.get("condition_note")}</div>' if p.get("condition_note") else ''}
                    
                    <div class="model-comparison-grid">
                        <div class="model-box {'box-correct' if a_corr else 'box-incorrect'}">
                            <div class="model-box-title">{model_a_name} {a_badge}</div>
                            <div class="pred-text">Pred: <b>{a_info.get('predicted_label')}</b></div>
                            <div class="conf-text">Confidence: {a_info.get('confidence', 0.0):.2%}</div>
                        </div>
                        <div class="model-box {'box-correct' if b_corr else 'box-incorrect'}">
                            <div class="model-box-title">{model_b_name} {b_badge}</div>
                            <div class="pred-text">Pred: <b>{b_info.get('predicted_label')}</b></div>
                            <div class="conf-text">Confidence: {b_info.get('confidence', 0.0):.2%}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """

    # Benchmark metadata rows
    bench_a = benchmark_results.get("model_a", {})
    bench_b = benchmark_results.get("model_b", {})
    hw_info = bench_a.get("hardware", benchmark_results.get("hardware", {}))

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ShiftProof Evaluation Report - Session #{session.get('id', 'N/A')}</title>
    <style>
        :root {{
            --bg: #0d1117;
            --card-bg: #161b22;
            --border: #30363d;
            --text-main: #f0f6fc;
            --text-muted: #8b949e;
            --accent: #58a6ff;
            --accent-green: #238636;
            --accent-green-light: #3fb950;
            --accent-red: #da3633;
            --accent-yellow: #d29922;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            padding: 32px 20px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        header {{
            margin-bottom: 28px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
        }}
        .title {{
            font-size: 26px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 6px;
        }}
        .subtitle {{
            color: var(--text-muted);
            font-size: 14px;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 24px;
        }}
        .grid-4 {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
        }}
        .stat-card {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: 700;
            color: #ffffff;
            margin: 4px 0;
        }}
        .stat-label {{
            color: var(--text-muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        h2 {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 14px;
            color: #ffffff;
            border-left: 4px solid var(--accent);
            padding-left: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            padding: 10px 14px;
            border: 1px solid var(--border);
            text-align: left;
        }}
        th {{
            background: #21262d;
            color: #ffffff;
            font-weight: 600;
        }}
        tr:nth-child(even) {{ background: #12161c; }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-success {{ background: #238636; color: #ffffff; }}
        .badge-danger {{ background: #da3633; color: #ffffff; }}
        .badge-warning {{ background: #9e6a03; color: #ffffff; }}
        .tag {{
            background: #21262d;
            border: 1px solid var(--border);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
            color: var(--accent);
            font-family: monospace;
        }}
        .capture-card {{
            margin-bottom: 16px;
        }}
        .capture-flex {{
            display: flex;
            gap: 20px;
        }}
        .capture-img-col {{
            width: 160px;
            flex-shrink: 0;
            text-align: center;
        }}
        .capture-thumb {{
            width: 160px;
            height: 120px;
            object-fit: cover;
            border-radius: 6px;
            border: 1px solid var(--border);
        }}
        .capture-meta {{
            margin-top: 6px;
            font-size: 12px;
            color: var(--text-muted);
        }}
        .capture-details-col {{
            flex-grow: 1;
        }}
        .true-label-box {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }}
        .label-heading {{ font-size: 13px; color: var(--text-muted); }}
        .true-label-text {{ font-size: 16px; font-weight: 700; color: #ffffff; }}
        .condition-note {{ font-size: 12px; color: #a0aec0; margin-bottom: 10px; }}
        .model-comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }}
        .model-box {{
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 10px 12px;
        }}
        .box-correct {{ border-left: 4px solid var(--accent-green-light); }}
        .box-incorrect {{ border-left: 4px solid var(--accent-red); }}
        .model-box-title {{
            font-size: 12px;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }}
        .pred-text {{ font-size: 14px; margin-bottom: 2px; }}
        .conf-text {{ font-size: 12px; color: var(--text-muted); }}
        .disclaimer-card {{
            background: #1c1917;
            border: 1px solid #78350f;
            border-radius: 8px;
            padding: 18px;
            margin-top: 32px;
        }}
        .disclaimer-card h3 {{
            color: #fbbf24;
            font-size: 15px;
            margin-bottom: 8px;
        }}
        .disclaimer-card ul {{
            padding-left: 20px;
            color: #d6d3d1;
            font-size: 13px;
        }}
        .disclaimer-card li {{
            margin-bottom: 6px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="title">ShiftProof Comparative Evaluation & Shift Report</div>
            <div class="subtitle">
                Session #{session.get('id', 'N/A')} &bull; Label: <i>"{session.get('label', 'Default Session')}"</i> &bull; Total Captures: <b>{total_caps}</b>
            </div>
        </header>

        <!-- Top Overview Stats -->
        <div class="grid-4">
            <div class="card stat-card">
                <div class="stat-label">{model_a_name} Accuracy</div>
                <div class="stat-value" style="color: #63b3ed;">{acc_a:.1%}</div>
                <div style="font-size: 11px; color: var(--text-muted);">{model_a_data.get('total_correct', 0)} / {total_caps} correct</div>
            </div>
            <div class="card stat-card">
                <div class="stat-label">{model_b_name} Accuracy</div>
                <div class="stat-value" style="color: #9f7aea;">{acc_b:.1%}</div>
                <div style="font-size: 11px; color: var(--text-muted);">{model_b_data.get('total_correct', 0)} / {total_caps} correct</div>
            </div>
            <div class="card stat-card">
                <div class="stat-label">Prediction Shift Rate</div>
                <div class="stat-value" style="color: #f6ad55;">{change_rate:.1%}</div>
                <div style="font-size: 11px; color: var(--text-muted);">{changes_count} of {total_caps} predictions changed</div>
            </div>
            <div class="card stat-card">
                <div class="stat-label">Agreement Rate</div>
                <div class="stat-value" style="color: #68d391;">{agreement_rate:.1%}</div>
                <div style="font-size: 11px; color: var(--text-muted);">Models agreed on {total_caps - changes_count} captures</div>
            </div>
        </div>

        <!-- Condition Breakdown Table -->
        <div class="card" style="margin-bottom: 24px;">
            <h2>Condition-Level Accuracy & Shift Matrix</h2>
            <table>
                <thead>
                    <tr>
                        <th>Condition Type</th>
                        <th style="text-align: center;">Captures</th>
                        <th style="text-align: center;">{model_a_name} Acc</th>
                        <th style="text-align: center;">{model_b_name} Acc</th>
                        <th style="text-align: center;">Delta (&Delta;)</th>
                    </tr>
                </thead>
                <tbody>
                    {cond_rows_html if cond_rows_html else '<tr><td colspan="5" style="text-align:center;">No condition data recorded.</td></tr>'}
                </tbody>
            </table>
        </div>

        <!-- Latency & Hardware Benchmark Config Table -->
        <div class="card" style="margin-bottom: 24px;">
            <h2>Deployment & Latency Benchmarks</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric / Config Parameter</th>
                        <th>{model_a_name}</th>
                        <th>{model_b_name}</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Requested Execution Provider</td>
                        <td><code>{bench_a.get('provider_requested', 'QNN')}</code></td>
                        <td><code>{bench_b.get('provider_requested', 'QNN')}</code></td>
                    </tr>
                    <tr>
                        <td>Active Execution Provider</td>
                        <td><b>{bench_a.get('provider_active', 'CPUExecutionProvider')}</b></td>
                        <td><b>{bench_b.get('provider_active', 'CPUExecutionProvider')}</b></td>
                    </tr>
                    <tr>
                        <td>Inference Latency (Median)</td>
                        <td><b>{bench_a.get('latency_median_ms', 'N/A')} ms</b></td>
                        <td><b>{bench_b.get('latency_median_ms', 'N/A')} ms</b></td>
                    </tr>
                    <tr>
                        <td>Inference Latency (p95)</td>
                        <td>{bench_a.get('latency_p95_ms', 'N/A')} ms</td>
                        <td>{bench_b.get('latency_p95_ms', 'N/A')} ms</td>
                    </tr>
                    <tr>
                        <td>End-to-End Response (Median)</td>
                        <td>{benchmark_results.get('e2e_a', {}).get('e2e_median_ms', 'N/A')} ms</td>
                        <td>{benchmark_results.get('e2e_b', {}).get('e2e_median_ms', 'N/A')} ms</td>
                    </tr>
                    <tr>
                        <td>Hardware & Platform</td>
                        <td colspan="2"><code>{hw_info.get('processor', 'CPU')} &bull; {hw_info.get('platform', 'Host Platform')}</code></td>
                    </tr>
                    <tr>
                        <td>Benchmark Provider Note</td>
                        <td colspan="2" style="color: #fbbf24; font-size: 12px;">{bench_a.get('provider_note', 'Executed on CPU.')}</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Paired Before/After Captures Gallery -->
        <div class="card" style="margin-bottom: 24px;">
            <h2>Paired Capture Comparison Gallery (Identical Saved Frames)</h2>
            {paired_cards_html if paired_cards_html else '<p style="color: var(--text-muted);">No captures available in this session.</p>'}
        </div>

        <!-- Mandatory Limitations and Methodological Disclosures -->
        <div class="disclaimer-card">
            <h3>Methodology Limitations & Disclosures</h3>
            <ul>
                <li>
                    <b>Execution Provider Disclosure:</b> NPU/QNN execution was not tested — target hardware is not yet available in this host environment. Benchmarks and predictions were executed via <code>CPUExecutionProvider</code> with full fallback tracking enabled.
                </li>
                <li>
                    <b>Stand-in Model Pair Disclosure:</b> The compared models ({model_a_name} and {model_b_name}) represent a stand-in baseline pair for demonstrating comparative pipeline mechanics and shift diagnostics, not the proposal's full trained-bias methodology.
                </li>
                <li>
                    <b>Confidence Interpretation:</b> Confidence metrics represent top-class softmax activation scores from the respective model heads and must not be interpreted as calibrated Bayesian probabilities.
                </li>
                <li>
                    <b>Capture Data Disclosure:</b> The condition images in this report were generated programmatically (image transforms applied to a single reference photo) to validate the evaluation, comparison, and reporting pipeline end-to-end. Live webcam capture with mandatory label confirmation is fully implemented and demonstrated separately via <code>main.py</code> / <code>ui/capture_widget.py</code>, but is not the data source for the report shown here.
                </li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    return str(out_file.resolve())
