"""
Shared utilities for the Virtual Steering Wheel application.
"""

from collections import deque
from typing import Deque, Iterable, Optional, Tuple

import cv2
import numpy as np


def moving_average(values: Deque[float]) -> float:
    """
    Compute the arithmetic mean of values in a deque.

    Returns 0.0 when the buffer is empty so callers can safely use the result
    without extra checks.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Restrict a numeric value to the inclusive range [minimum, maximum]."""
    return max(minimum, min(maximum, value))


def angle_between_points(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
) -> float:
    """
    Calculate the angle (in degrees) of the vector from point_a to point_b
    relative to the positive x-axis.

    Used to measure how the line between both hands is rotated, which maps
    directly to steering wheel orientation.
    """
    delta_x = point_b[0] - point_a[0]
    delta_y = point_b[1] - point_a[1]
    radians = np.arctan2(delta_y, delta_x)
    return float(np.degrees(radians))


def average_landmark_points(points: Iterable[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """
    Average multiple (x, y) landmark positions into a single point.

    Returns None when no points are supplied.
    """
    point_list = list(points)
    if not point_list:
        return None
    x_sum = sum(point[0] for point in point_list)
    y_sum = sum(point[1] for point in point_list)
    count = len(point_list)
    return (x_sum / count, y_sum / count)


def draw_text_block(
    frame: np.ndarray,
    lines: Iterable[str],
    origin: Tuple[int, int] = (10, 30),
    line_height: int = 28,
    font_scale: float = 0.7,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> None:
    """Draw multiple lines of HUD text onto an OpenCV frame."""
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX

    for index, line in enumerate(lines):
        y_offset = y + index * line_height
        cv2.putText(
            frame,
            line,
            (x, y_offset),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x, y_offset),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


class FpsCounter:
    """Track and report frames-per-second using a sliding time window."""

    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: Deque[float] = deque(maxlen=window_size)

    def tick(self, timestamp: float) -> float:
        """
        Record a frame timestamp and return the current smoothed FPS estimate.

        Args:
            timestamp: Monotonic time in seconds (from time.perf_counter()).
        """
        self._timestamps.append(timestamp)
        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed
