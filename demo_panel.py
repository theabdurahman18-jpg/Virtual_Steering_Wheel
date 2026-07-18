"""
On-screen demo panel drawn on the webcam feed.

Shows a car/dot moving left/right so you can see steering work instantly
without Notepad or any other app.
"""

from typing import Tuple

import cv2
import numpy as np

from steering import SteeringDirection


class SteeringDemoPanel:
    """Draw a live steering test bar at the bottom of the camera frame."""

    def __init__(self, margin: int = 20, bar_height: int = 50) -> None:
        self.margin = margin
        self.bar_height = bar_height
        self._car_x = 0.5  # 0.0 = left, 1.0 = right

    def draw(
        self,
        frame: np.ndarray,
        direction: SteeringDirection,
        hands_ready: bool,
    ) -> None:
        """
        Update and render the demo car position from steering direction.

        The car slides left/right while you turn, even if no keyboard app is open.
        """
        if direction == SteeringDirection.LEFT:
            self._car_x = max(0.05, self._car_x - 0.03)
        elif direction == SteeringDirection.RIGHT:
            self._car_x = min(0.95, self._car_x + 0.03)
        elif direction == SteeringDirection.STRAIGHT and hands_ready:
            # Gently return to center when driving straight.
            self._car_x += (0.5 - self._car_x) * 0.08

        height, width = frame.shape[:2]
        bar_y1 = height - self.margin - self.bar_height
        bar_y2 = height - self.margin
        bar_x1 = self.margin
        bar_x2 = width - self.margin

        overlay = frame.copy()
        cv2.rectangle(overlay, (bar_x1, bar_y1), (bar_x2, bar_y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (200, 200, 200), 2)
        center_x = (bar_x1 + bar_x2) // 2
        cv2.line(frame, (center_x, bar_y1), (center_x, bar_y2), (80, 80, 80), 1)

        car_x = int(bar_x1 + (bar_x2 - bar_x1) * self._car_x)
        car_y = (bar_y1 + bar_y2) // 2
        color = (0, 220, 0) if hands_ready else (120, 120, 120)
        if direction in (SteeringDirection.LEFT, SteeringDirection.RIGHT):
            color = (0, 200, 255)

        cv2.circle(frame, (car_x, car_y), 16, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            frame,
            "LIVE DEMO - car moves when you steer (no Notepad needed)",
            (bar_x1 + 8, bar_y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
