"""PySide6 Guided Capture Widget with live preview, alignment overlay, and ground-truth confirmation."""

import os
import sqlite3
import numpy as np
from typing import Optional, Dict, Any, List

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QPushButton,
    QGroupBox,
    QProgressBar,
    QCompleter,
    QFrame,
    QMessageBox,
    QSizePolicy,
)

from capture.camera import CameraSession, CameraNotFoundError, CameraError
from capture.capture_flow import CaptureFlow, LabelConfirmationError, VALID_CONDITION_TYPES
from storage.repository import get_session_summary


class CameraPreviewWidget(QLabel):
    """Custom preview label that displays the RGB video frame with a central alignment guide overlay."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #121417; border-radius: 8px; border: 1px solid #2d3748;")
        self._current_pixmap: Optional[QPixmap] = None
        self._error_message: Optional[str] = None

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        """Convert BGR frame to RGB and set as display pixmap."""
        self._error_message = None
        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        
        # Explicitly convert BGR to RGB at display boundary
        # Using numpy slicing [..., ::-1] or cv2 for RGB conversion
        rgb_data = frame_bgr[..., ::-1].copy()
        
        q_img = QImage(rgb_data.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(q_img)
        self.update()

    def set_error(self, message: str) -> None:
        """Display an in-widget camera error state."""
        self._current_pixmap = None
        self._error_message = message
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw frame or placeholder
        if self._current_pixmap and not self._current_pixmap.isNull():
            scaled = self._current_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

            # Draw Alignment Guide Overlay over the scaled image area
            guide_w = int(scaled.width() * 0.60)
            guide_h = int(scaled.height() * 0.60)
            gx = x + (scaled.width() - guide_w) // 2
            gy = y + (scaled.height() - guide_h) // 2

            # Semi-transparent bounding box with glowing cyan border
            pen = QPen(QColor(0, 229, 255, 200), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 229, 255, 15))
            painter.drawRoundedRect(gx, gy, guide_w, guide_h, 8, 8)

            # Corner brackets for tactile framing
            bracket_len = 20
            corner_pen = QPen(QColor(0, 229, 255, 255), 3, Qt.SolidLine)
            painter.setPen(corner_pen)
            # Top-left
            painter.drawLine(gx, gy, gx + bracket_len, gy)
            painter.drawLine(gx, gy, gx, gy + bracket_len)
            # Top-right
            painter.drawLine(gx + guide_w, gy, gx + guide_w - bracket_len, gy)
            painter.drawLine(gx + guide_w, gy, gx + guide_w, gy + bracket_len)
            # Bottom-left
            painter.drawLine(gx, gy + guide_h, gx + bracket_len, gy + guide_h)
            painter.drawLine(gx, gy + guide_h, gx, gy + guide_h - bracket_len)
            # Bottom-right
            painter.drawLine(gx + guide_w, gy + guide_h, gx + guide_w - bracket_len, gy + guide_h)
            painter.drawLine(gx + guide_w, gy + guide_h, gx + guide_w, gy + guide_h - bracket_len)

            # Guide Text
            painter.setPen(QColor(0, 229, 255, 220))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.drawText(
                gx, gy + guide_h - 12, guide_w, 20, Qt.AlignCenter, "ALIGN OBJECT HERE"
            )

        else:
            # Placeholder / Camera Not Found
            painter.fillRect(self.rect(), QColor("#1a1d24"))
            painter.setPen(QColor("#e53e3e" if self._error_message else "#718096"))
            painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
            msg = self._error_message or "Connecting to Camera..."
            painter.drawText(self.rect(), Qt.AlignCenter, msg)


class CaptureWidget(QWidget):
    """Main Guided Capture UI Widget integrating camera stream, live inference preview,

    ground-truth confirmation, condition tagging, and session summary.
    """

    capture_saved = Signal(dict)

    def __init__(
        self,
        flow: CaptureFlow,
        camera_session: Optional[CameraSession] = None,
        model_name: str = "MobileNetV2",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.flow = flow
        self.camera = camera_session or CameraSession(device_index=0)
        self.model_name = model_name

        self._latest_frame_bgr: Optional[np.ndarray] = None
        self._latest_prediction: Optional[Dict[str, Any]] = None

        # Timers
        self._frame_timer = QTimer(self)
        self._frame_timer.timeout.connect(self._poll_frame)

        self._inference_timer = QTimer(self)
        self._inference_timer.timeout.connect(self._run_inference_preview)

        self._init_ui()
        self._start_camera()
        self._refresh_session_summary()

    def _init_ui(self) -> None:
        self.setWindowTitle(f"ShiftProof - Guided Capture (Session #{self.flow.session_id})")
        self.resize(1020, 680)
        self.setStyleSheet("""
            QWidget {
                background-color: #0f1117;
                color: #e2e8f0;
                font-family: 'Segoe UI', Inter, sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #2d3748;
                border-radius: 8px;
                margin-top: 14px;
                padding-top: 14px;
                font-weight: bold;
                color: #90cdf4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px 0 6px;
            }
            QLineEdit, QComboBox {
                background-color: #1a202c;
                border: 1px solid #4a5568;
                border-radius: 6px;
                padding: 6px 10px;
                color: #f7fafc;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #63b3ed;
            }
            QPushButton {
                background-color: #2b6cb0;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 8px 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3182ce;
            }
            QPushButton:pressed {
                background-color: #2c5282;
            }
            QPushButton:disabled {
                background-color: #2d3748;
                color: #718096;
            }
            QProgressBar {
                background-color: #1a202c;
                border: 1px solid #4a5568;
                border-radius: 4px;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #319795;
                border-radius: 3px;
            }
        """)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # Left Column: Camera Viewport + Status
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)

        # Header Info Banner
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #1a202c; border-radius: 6px; padding: 6px;")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(10, 4, 10, 4)

        self.session_title_lbl = QLabel(f"<b>Session #{self.flow.session_id}</b>")
        self.session_title_lbl.setStyleSheet("color: #63b3ed; font-size: 14px;")
        
        self.provider_badge = QLabel("Provider: Initializing...")
        self.provider_badge.setStyleSheet("color: #a0aec0; font-size: 12px;")

        header_layout.addWidget(self.session_title_lbl)
        header_layout.addStretch()
        header_layout.addWidget(self.provider_badge)
        left_layout.addWidget(header_frame)

        # Camera Preview
        self.preview_widget = CameraPreviewWidget()
        left_layout.addWidget(self.preview_widget, stretch=1)

        # Camera Controls / Retry
        cam_ctrl_layout = QHBoxLayout()
        self.cam_status_lbl = QLabel("Camera: Starting...")
        self.cam_status_lbl.setStyleSheet("color: #a0aec0; font-size: 12px;")
        self.retry_cam_btn = QPushButton("Reconnect Camera")
        self.retry_cam_btn.setStyleSheet("background-color: #4a5568; font-size: 12px; padding: 4px 10px;")
        self.retry_cam_btn.clicked.connect(self._start_camera)

        cam_ctrl_layout.addWidget(self.cam_status_lbl)
        cam_ctrl_layout.addStretch()
        cam_ctrl_layout.addWidget(self.retry_cam_btn)
        left_layout.addLayout(cam_ctrl_layout)

        main_layout.addLayout(left_layout, stretch=3)

        # Right Column: Live Prediction, Tagging Form, Session Summary
        right_layout = QVBoxLayout()
        right_layout.setSpacing(12)

        # 1. Live Prediction Card
        pred_group = QGroupBox("Live Inference Preview (MobileNetV2)")
        pred_layout = QVBoxLayout(pred_group)
        pred_layout.setSpacing(8)

        self.pred_label_lbl = QLabel("Waiting for model...")
        self.pred_label_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #48bb78;")

        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setValue(0)
        self.conf_bar.setFormat("%v% Confidence")
        self.conf_bar.setFixedHeight(20)

        pred_layout.addWidget(self.pred_label_lbl)
        pred_layout.addWidget(self.conf_bar)
        right_layout.addWidget(pred_group)

        # 2. Confirmation & Tagging Form
        tag_group = QGroupBox("Ground-Truth & Condition Confirmation")
        tag_layout = QVBoxLayout(tag_group)
        tag_layout.setSpacing(10)

        # True Label Field
        true_label_hdr = QHBoxLayout()
        true_label_hdr.addWidget(QLabel("<b>Ground-Truth Label:</b> <span style='color:#e53e3e;'>*</span>"))
        self.copy_pred_btn = QPushButton("Use Live Prediction")
        self.copy_pred_btn.setStyleSheet("background-color: #2c5282; font-size: 11px; padding: 2px 8px;")
        self.copy_pred_btn.clicked.connect(self._copy_prediction_to_true_label)
        true_label_hdr.addStretch()
        true_label_hdr.addWidget(self.copy_pred_btn)
        tag_layout.addLayout(true_label_hdr)

        self.true_label_input = QLineEdit()
        self.true_label_input.setPlaceholderText("Enter or select true object class...")
        self.true_label_input.textChanged.connect(self._validate_form)
        
        # Populate autocomplete from adapter labels if available
        if hasattr(self.flow.adapter, "_labels") and self.flow.adapter._labels:
            completer = QCompleter(self.flow.adapter._labels, self)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            completer.setFilterMode(Qt.MatchContains)
            self.true_label_input.setCompleter(completer)

        tag_layout.addWidget(self.true_label_input)

        # Condition Type Dropdown
        tag_layout.addWidget(QLabel("<b>Condition Type:</b>"))
        self.condition_combo = QComboBox()
        for cond in ["baseline", "background", "lighting", "distance"]:
            self.condition_combo.addItem(cond)
        tag_layout.addWidget(self.condition_combo)

        # Condition Note
        tag_layout.addWidget(QLabel("<b>Condition Note (Optional):</b>"))
        self.condition_note_input = QLineEdit()
        self.condition_note_input.setPlaceholderText("e.g. low-light shadow, cluttered background...")
        tag_layout.addWidget(self.condition_note_input)

        # Confirm & Save Button
        self.confirm_btn = QPushButton("Confirm & Save Capture")
        self.confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #38a169;
                font-size: 14px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2f855a;
            }
            QPushButton:disabled {
                background-color: #2d3748;
                color: #718096;
            }
        """)
        self.confirm_btn.setEnabled(False)  # Disabled until true_label is non-empty
        self.confirm_btn.clicked.connect(self._on_confirm_and_save)
        tag_layout.addWidget(self.confirm_btn)

        # Toast / Status Banner
        self.status_msg_lbl = QLabel("")
        self.status_msg_lbl.setStyleSheet("color: #cbd5e0; font-size: 12px;")
        self.status_msg_lbl.setWordWrap(True)
        tag_layout.addWidget(self.status_msg_lbl)

        right_layout.addWidget(tag_group)

        # 3. Session Summary Counter & Report Export
        summary_group = QGroupBox("Session Captures Summary & Reporting")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.setSpacing(8)

        self.summary_baseline_lbl = QLabel("  * Baseline: 0")
        self.summary_background_lbl = QLabel("  * Background: 0")
        self.summary_lighting_lbl = QLabel("  * Lighting: 0")
        self.summary_distance_lbl = QLabel("  * Distance: 0")
        self.summary_total_lbl = QLabel("<b>Total Captures: 0</b>")
        self.summary_total_lbl.setStyleSheet("color: #63b3ed;")

        summary_layout.addWidget(self.summary_baseline_lbl)
        summary_layout.addWidget(self.summary_background_lbl)
        summary_layout.addWidget(self.summary_lighting_lbl)
        summary_layout.addWidget(self.summary_distance_lbl)
        summary_layout.addWidget(self.summary_total_lbl)

        # Export / Compare Button
        self.export_report_btn = QPushButton("Run Comparison & Export Report")
        self.export_report_btn.setStyleSheet("""
            QPushButton {
                background-color: #805ad5;
                color: #ffffff;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #6b46c1;
            }
        """)
        self.export_report_btn.clicked.connect(self._on_run_comparison_and_export)
        summary_layout.addWidget(self.export_report_btn)

        right_layout.addWidget(summary_group)
        right_layout.addStretch()

        main_layout.addLayout(right_layout, stretch=2)

    def _start_camera(self) -> None:
        """Attempt to open camera and start timers, or show error banner."""
        try:
            self.camera.open()
            self.cam_status_lbl.setText("<span style='color:#48bb78;'>Camera Active (Index 0)</span>")
            self.retry_cam_btn.setVisible(False)
            self._frame_timer.start(33)  # ~30 FPS frame polling
            self._inference_timer.start(150)  # ~6.6 FPS inference preview
        except CameraNotFoundError as e:
            self.cam_status_lbl.setText("<span style='color:#e53e3e;'>Camera Not Found</span>")
            self.preview_widget.set_error(
                "Camera device not found.\nCheck webcam connection and click 'Reconnect Camera'."
            )
            self.retry_cam_btn.setVisible(True)
        except Exception as e:
            self.cam_status_lbl.setText(f"<span style='color:#e53e3e;'>Camera Error</span>")
            self.preview_widget.set_error(f"Error opening camera: {e}")
            self.retry_cam_btn.setVisible(True)

    def _poll_frame(self) -> None:
        """Capture frame from camera session and update preview."""
        try:
            if self.camera.is_opened():
                frame_bgr = self.camera.read_frame()
                self._latest_frame_bgr = frame_bgr
                self.preview_widget.update_frame(frame_bgr)
        except Exception as e:
            self.preview_widget.set_error(f"Frame read error: {e}")

    def _run_inference_preview(self) -> None:
        """Run inference on the latest frame and update prediction panel."""
        if self._latest_frame_bgr is None or self._latest_frame_bgr.size == 0:
            return

        try:
            res = self.flow.preview_prediction(self._latest_frame_bgr)
            self._latest_prediction = res

            label = res.get("predicted_label", "Unknown")
            conf = res.get("confidence", 0.0)
            active_provider = res.get("active_provider", "CPU")

            self.pred_label_lbl.setText(f"Predicted: {label}")
            conf_pct = int(conf * 100)
            self.conf_bar.setValue(conf_pct)
            self.conf_bar.setFormat(f"{conf_pct}% ({label})")

            # Update provider badge
            fallback_text = " (Fallback)" if res.get("fallback") else ""
            self.provider_badge.setText(f"Active Provider: <b>{active_provider}{fallback_text}</b>")
        except Exception as e:
            self.pred_label_lbl.setText("Prediction error")

    def _validate_form(self) -> None:
        """Gate the Confirm button on non-empty true_label."""
        text = self.true_label_input.text().strip()
        self.confirm_btn.setEnabled(len(text) > 0)

    def _copy_prediction_to_true_label(self) -> None:
        """Convenience action to copy current predicted label into ground-truth input."""
        if self._latest_prediction and self._latest_prediction.get("predicted_label"):
            self.true_label_input.setText(self._latest_prediction["predicted_label"])

    def _on_confirm_and_save(self) -> None:
        """Execute confirmation and save flow."""
        true_label = self.true_label_input.text().strip()
        condition_type = self.condition_combo.currentText()
        condition_note = self.condition_note_input.text().strip() or None

        if not true_label:
            QMessageBox.warning(
                self, "Label Required", "Please enter a ground-truth label before saving."
            )
            return

        if self._latest_frame_bgr is None:
            QMessageBox.warning(self, "No Frame", "No active camera frame available to capture.")
            return

        try:
            pred_lbl = self._latest_prediction.get("predicted_label") if self._latest_prediction else None
            conf = self._latest_prediction.get("confidence") if self._latest_prediction else None

            record = self.flow.confirm_and_save(
                frame_bgr=self._latest_frame_bgr,
                true_label=true_label,
                condition_type=condition_type,
                condition_note=condition_note,
                predicted_label=pred_lbl,
                confidence=conf,
            )

            # Success feedback
            match_str = "Match (Correct)" if record["correct"] == 1 else "Mismatch (Incorrect)"
            self.status_msg_lbl.setStyleSheet("color: #48bb78; font-weight: bold;")
            self.status_msg_lbl.setText(
                f"[OK] Saved Capture #{record['id']} [{condition_type}] -> {match_str} (Path: {record['image_path']})"
            )

            # Refresh summary counter
            self._refresh_session_summary()

            # Reset note field
            self.condition_note_input.clear()

            self.capture_saved.emit(record)

        except LabelConfirmationError as e:
            QMessageBox.critical(self, "Confirmation Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save capture: {e}")

    def _refresh_session_summary(self) -> None:
        """Query repository for session summary counts and update UI."""
        summary = get_session_summary(self.flow.conn, self.flow.session_id)
        
        b = summary.get("baseline", 0)
        bg = summary.get("background", 0)
        l = summary.get("lighting", 0)
        d = summary.get("distance", 0)
        total = b + bg + l + d

        self.summary_baseline_lbl.setText(f"  * Baseline: <b>{b}</b>")
        self.summary_background_lbl.setText(f"  * Background: <b>{bg}</b>")
        self.summary_lighting_lbl.setText(f"  * Lighting: <b>{l}</b>")
        self.summary_distance_lbl.setText(f"  * Distance: <b>{d}</b>")
        self.summary_total_lbl.setText(f"<b>Total Captures in Session: {total}</b>")

    def _on_run_comparison_and_export(self) -> None:
        """Run comparison against alternative model (SqueezeNet) on all session captures and generate HTML report."""
        from storage.repository import get_captures_by_session, get_session
        from adapter.squeezenet_adapter import SqueezeNetAdapter
        from evaluation.compare import compare_models
        from benchmark.latency import benchmark_model_inference, benchmark_end_to_end
        from reporting.report_generator import generate_html_report

        captures = get_captures_by_session(self.flow.conn, self.flow.session_id)
        if not captures:
            QMessageBox.information(
                self,
                "No Captures Yet",
                "Please save at least one capture before running comparative evaluation.",
            )
            return

        try:
            # 1. Load second model adapter
            squeezenet = SqueezeNetAdapter()
            squeezenet.load(
                model_path="models/squeezenet1.1-7.onnx",
                labels_path="models/imagenet_classes.txt",
            )

            # 2. Run Comparison
            comparison = compare_models(
                captures=captures,
                model_a=self.flow.adapter,
                model_b=squeezenet,
                model_a_name=f"{self.model_name} (Base)",
                model_b_name="SqueezeNet 1.1 (Alternative)",
            )

            # 3. Benchmark
            bench_a = benchmark_model_inference(self.flow.adapter, model_name=self.model_name)
            bench_b = benchmark_model_inference(squeezenet, model_name="SqueezeNet 1.1")
            
            img_paths = [c["image_path"] for c in captures if os.path.exists(c.get("image_path", ""))]
            e2e_a = benchmark_end_to_end(self.flow.adapter, img_paths, model_name=self.model_name) if img_paths else {}
            e2e_b = benchmark_end_to_end(squeezenet, img_paths, model_name="SqueezeNet 1.1") if img_paths else {}

            benchmarks = {
                "model_a": bench_a,
                "model_b": bench_b,
                "e2e_a": e2e_a,
                "e2e_b": e2e_b,
            }

            # 4. Generate HTML Report
            session = get_session(self.flow.conn, self.flow.session_id) or {"id": self.flow.session_id}
            report_path = f"reports/session_{self.flow.session_id}_report.html"
            final_path = generate_html_report(session, comparison, benchmarks, output_path=report_path)

            QMessageBox.information(
                self,
                "Report Generated",
                f"Comparative evaluation report generated successfully!\n\nSaved to: {final_path}\n"
                f"Shift Rate: {comparison['prediction_change_rate']:.1%}\n"
                f"Base Acc: {comparison['model_a']['overall_accuracy']:.1%} | Alt Acc: {comparison['model_b']['overall_accuracy']:.1%}",
            )

        except Exception as e:
            QMessageBox.critical(self, "Evaluation Error", f"Failed to generate report: {e}")

    def closeEvent(self, event) -> None:
        """Ensure camera and timers are cleaned up on window close."""
        self._frame_timer.stop()
        self._inference_timer.stop()
        self.camera.close()
        super().closeEvent(event)

