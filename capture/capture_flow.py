"""Capture flow orchestrator with model preview, label confirmation, and storage persistence."""

import os
import sqlite3
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union

from adapter.base import ModelAdapter
from storage.repository import insert_capture


class LabelConfirmationError(ValueError):
    """Raised when an attempt is made to save a capture without explicit ground truth label confirmation."""
    pass


VALID_CONDITION_TYPES = {"baseline", "background", "lighting", "distance"}


class CaptureFlow:
    """Coordinates camera frame processing, real-time model inference preview,

    explicit user label confirmation, image persistence to disk, and database logging.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        conn: sqlite3.Connection,
        session_id: int,
        base_dir: Union[str, Path] = "captures",
    ) -> None:
        """Initialize capture flow.

        Args:
            adapter: Loaded model adapter implementing :class:`ModelAdapter`.
            conn: Active SQLite connection to ShiftProof database.
            session_id: ID of the active test session.
            base_dir: Base directory where capture image files will be stored.
        """
        self.adapter = adapter
        self.conn = conn
        self.session_id = session_id
        self.base_dir = Path(base_dir)

    def preview_prediction(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Run inference on the incoming BGR camera frame for preview.

        Converts the BGR OpenCV frame to RGB before passing to the adapter.

        Args:
            frame_bgr: Raw video frame in BGR format.

        Returns:
            Dict containing 'predicted_label', 'confidence', 'raw_scores',
            'fallback', and 'active_provider'.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Invalid frame supplied for prediction preview.")

        # Convert OpenCV BGR to RGB for model input
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.adapter.predict(frame_rgb)
        return {
            "predicted_label": result.get("label"),
            "confidence": float(result.get("confidence", 0.0)),
            "raw_scores": result.get("raw_scores"),
            "fallback": result.get("fallback", False),
            "active_provider": result.get("active_provider"),
        }

    def confirm_and_save(
        self,
        frame_bgr: np.ndarray,
        true_label: Optional[str],
        condition_type: str,
        condition_note: Optional[str] = None,
        predicted_label: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Explicitly confirm ground-truth label and condition tag, save image to disk,

        and insert record into the storage repository.

        Args:
            frame_bgr: Frame in native OpenCV BGR format to be saved to disk.
            true_label: The confirmed ground-truth label (MANDATORY).
            condition_type: Condition type tag ('baseline', 'background', 'lighting', 'distance').
            condition_note: Optional textual description of the environment condition.
            predicted_label: Optional predicted label. If omitted, will be derived by inference.
            confidence: Optional confidence score. If omitted, will be derived by inference.

        Returns:
            Dict[str, Any]: Saved capture record.

        Raises:
            LabelConfirmationError: If true_label is missing or blank.
            ValueError: If condition_type is invalid or frame is corrupt.
        """
        # Proposal Requirement: MUST NOT allow saving without explicit confirmation of true_label
        if not true_label or not str(true_label).strip():
            raise LabelConfirmationError(
                "Cannot save capture: Ground-truth label confirmation is mandatory. "
                "Please provide a confirmed 'true_label'."
            )

        true_label = str(true_label).strip()

        cond_norm = condition_type.strip().lower() if condition_type else ""
        if cond_norm not in VALID_CONDITION_TYPES:
            raise ValueError(
                f"Invalid condition_type '{condition_type}'. "
                f"Must be one of: {sorted(list(VALID_CONDITION_TYPES))}"
            )

        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Cannot save an empty or corrupt frame.")

        # Run preview prediction if not already provided
        if predicted_label is None or confidence is None:
            preview = self.preview_prediction(frame_bgr)
            predicted_label = preview["predicted_label"]
            confidence = preview["confidence"]

        # Derive correctness: 1 if true_label matches predicted_label else 0
        correct = 1 if (true_label.lower() == str(predicted_label).strip().lower()) else 0

        # Create session capture directory: captures/{session_id}
        session_folder = self.base_dir / str(self.session_id)
        session_folder.mkdir(parents=True, exist_ok=True)

        # Temporary placeholder path for initial DB insertion
        temp_path = str(session_folder / "pending.jpg").replace("\\", "/")

        capture_id = insert_capture(
            conn=self.conn,
            session_id=self.session_id,
            image_path=temp_path,
            true_label=true_label,
            condition_type=cond_norm,
            condition_note=condition_note,
            predicted_label=predicted_label,
            confidence=confidence,
            correct=correct,
        )

        # Final image path: captures/{session_id}/{capture_id}.jpg
        rel_image_path = f"captures/{self.session_id}/{capture_id}.jpg"
        abs_image_path = self.base_dir / str(self.session_id) / f"{capture_id}.jpg"

        # Save frame to disk in native BGR format
        save_success = cv2.imwrite(str(abs_image_path), frame_bgr)
        if not save_success:
            raise IOError(f"Failed to write image frame to disk at: {abs_image_path}")

        # Update database with exact file path
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE captures SET image_path = ? WHERE id = ?",
            (rel_image_path, capture_id),
        )
        self.conn.commit()

        return {
            "id": capture_id,
            "session_id": self.session_id,
            "image_path": rel_image_path,
            "true_label": true_label,
            "condition_type": cond_norm,
            "condition_note": condition_note,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "correct": correct,
        }
