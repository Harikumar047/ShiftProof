"""Camera wrapper with explicit error handling and BGR/RGB management."""

import cv2
import numpy as np
from typing import Optional


class CameraError(Exception):
    """Base exception for camera capture errors."""
    pass


class CameraNotFoundError(CameraError):
    """Raised when the specified camera index cannot be opened or device is unavailable."""
    pass


class CameraFrameReadError(CameraError):
    """Raised when reading a frame from an open camera fails."""
    pass


class CameraSession:
    """Wraps OpenCV VideoCapture for robust frame capture.

    Frames captured by :meth:`read_frame` are in native OpenCV **BGR** format
    (numpy uint8 array with shape `(H, W, 3)`).
    Conversion to **RGB** must only occur at the boundary of display (e.g., UI widgets)
    or model inference (e.g., `MobileNetV2Adapter.predict`).
    """

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open the video capture device.

        Raises:
            CameraNotFoundError: If the camera cannot be opened or device index is invalid.
        """
        if self._cap is not None and self._cap.isOpened():
            return

        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise CameraNotFoundError(
                f"Camera device at index {self.device_index} could not be opened. "
                f"Verify that a webcam is connected and accessible."
            )

    def is_opened(self) -> bool:
        """Check if camera device is currently open."""
        return self._cap is not None and self._cap.isOpened()

    def read_frame(self) -> np.ndarray:
        """Read a single frame from the camera.

        Returns:
            np.ndarray: Captured frame in **BGR** uint8 format.

        Raises:
            CameraError: If camera is not opened.
            CameraFrameReadError: If frame capture fails or returns empty.
        """
        if not self.is_opened():
            raise CameraError("Camera is not opened. Call open() first.")

        ret, frame = self._cap.read()
        if not ret or frame is None or frame.size == 0:
            raise CameraFrameReadError("Failed to read frame from camera (empty or corrupt frame).")

        return frame

    def close(self) -> None:
        """Release the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraSession":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
