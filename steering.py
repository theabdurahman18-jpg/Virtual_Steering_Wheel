"""
Steering angle calculation, smoothing, and on-screen visualization.
"""

from collections import deque
from enum import Enum
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from hand_tracker import HandData
from utils import angle_between_points, clamp, moving_average


class SteeringDirection(str, Enum):
    """Discrete steering states derived from the smoothed angle."""

    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STRAIGHT = "STRAIGHT"
    UNKNOWN = "UNKNOWN"


class SteeringController:
    """
    Convert two-hand landmark data into a smoothed steering angle and direction.

    The angle is measured between the left and right hand center points relative
    to the horizontal axis. A moving average filter reduces jitter.
    """

    STRAIGHT_THRESHOLD_DEG = 15.0
    MAX_VISUAL_ANGLE_DEG = 90.0

    def __init__(
        self,
        smoothing_window: int = 8,
        straight_threshold_deg: float = 15.0,
    ) -> None:
        self._angle_buffer: Deque[float] = deque(maxlen=smoothing_window)
        self._last_valid_angle: float = 0.0
        self.STRAIGHT_THRESHOLD_DEG = straight_threshold_deg

    def update(
        self,
        left_hand: Optional[HandData],
        right_hand: Optional[HandData],
    ) -> Tuple[float, SteeringDirection, bool]:
        """
        Update steering state from the latest hand detections.

        Uses two-hand wheel angle when both hands are visible. Falls back to
        single-hand tilt when only one hand is detected.

        Returns:
            smoothed_angle: Filtered angle in degrees.
            direction: LEFT, RIGHT, STRAIGHT, or UNKNOWN.
            hands_ready: True when steering input is active this frame.
        """
        if left_hand is not None and right_hand is not None:
            raw_angle = angle_between_points(left_hand.center, right_hand.center)
            self._angle_buffer.append(raw_angle)
            self._last_valid_angle = moving_average(self._angle_buffer)
            hands_ready = True
        elif left_hand is not None or right_hand is not None:
            single_hand = left_hand if left_hand is not None else right_hand
            raw_angle = self._single_hand_angle(single_hand)
            self._angle_buffer.append(raw_angle)
            self._last_valid_angle = moving_average(self._angle_buffer)
            hands_ready = True
        elif self._angle_buffer:
            self._last_valid_angle = moving_average(self._angle_buffer)
            hands_ready = False
        else:
            hands_ready = False

        smoothed_angle = self._last_valid_angle
        direction = self._direction_from_angle(smoothed_angle, hands_ready)
        return smoothed_angle, direction, hands_ready

    @staticmethod
    def _single_hand_angle(hand: HandData) -> float:
        """
        Estimate steering tilt from one hand using wrist-to-index orientation.

        Raise/lower the hand to steer when the second hand is not visible.
        """
        return angle_between_points(hand.wrist, hand.index_tip) - 90.0

    def _direction_from_angle(
        self,
        angle: float,
        hands_ready: bool,
    ) -> SteeringDirection:
        """
        Map a smoothed angle to a steering direction using ±15° dead zone.

        Negative angles turn left; positive angles turn right.
        """
        if not hands_ready:
            return SteeringDirection.UNKNOWN

        if angle < -self.STRAIGHT_THRESHOLD_DEG:
            return SteeringDirection.LEFT
        if angle > self.STRAIGHT_THRESHOLD_DEG:
            return SteeringDirection.RIGHT
        return SteeringDirection.STRAIGHT

    def reset(self) -> None:
        """Clear smoothing history and return to neutral steering."""
        self._angle_buffer.clear()
        self._last_valid_angle = 0.0

    @staticmethod
    def direction_label(direction: SteeringDirection) -> str:
        """Human-readable label for HUD output."""
        if direction == SteeringDirection.LEFT:
            return "Turn Left"
        if direction == SteeringDirection.RIGHT:
            return "Turn Right"
        if direction == SteeringDirection.STRAIGHT:
            return "Straight"
        return "Show your hand(s)"


class SteeringWheelRenderer:
    """Draw a virtual steering wheel overlay that rotates with the current angle."""

    def __init__(
        self,
        center: Tuple[int, int] = (640, 120),
        radius: int = 90,
        max_angle_deg: float = SteeringController.MAX_VISUAL_ANGLE_DEG,
        straight_threshold_deg: float = SteeringController.STRAIGHT_THRESHOLD_DEG,
    ) -> None:
        self.center = center
        self.radius = radius
        self.max_angle_deg = max_angle_deg
        self.straight_threshold_deg = straight_threshold_deg

    def draw(
        self,
        frame: np.ndarray,
        angle_deg: float,
        direction: SteeringDirection,
        hands_ready: bool,
    ) -> None:
        """
        Render the steering wheel circle, spoke, and angle readout on the frame.

        The spoke rotates according to the smoothed steering angle. Color reflects
        the active direction (green = straight, yellow = turning).
        """
        cx, cy = self.center
        color = self._color_for_direction(direction, hands_ready)

        cv2.circle(frame, (cx, cy), self.radius, color, 3, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 6, color, -1, lineType=cv2.LINE_AA)

        display_angle = clamp(angle_deg, -self.max_angle_deg, self.max_angle_deg)
        spoke_radians = np.radians(display_angle)
        end_x = int(cx + self.radius * np.cos(spoke_radians))
        end_y = int(cy + self.radius * np.sin(spoke_radians))
        cv2.line(
            frame,
            (cx, cy),
            (end_x, end_y),
            color,
            4,
            lineType=cv2.LINE_AA,
        )

        # Small markers at ±15° to visualize the straight dead zone.
        for threshold in (-self.straight_threshold_deg, self.straight_threshold_deg):
            marker_rad = np.radians(threshold)
            mx = int(cx + (self.radius + 12) * np.cos(marker_rad))
            my = int(cy + (self.radius + 12) * np.sin(marker_rad))
            cv2.circle(frame, (mx, my), 4, (180, 180, 180), -1, lineType=cv2.LINE_AA)

        status = "ACTIVE" if hands_ready else "INACTIVE"
        cv2.putText(
            frame,
            f"Wheel: {display_angle:+.1f} deg ({status})",
            (cx - 130, cy + self.radius + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _color_for_direction(
        direction: SteeringDirection,
        hands_ready: bool,
    ) -> Tuple[int, int, int]:
        """Choose BGR color based on steering state."""
        if not hands_ready:
            return (120, 120, 120)
        if direction == SteeringDirection.STRAIGHT:
            return (0, 220, 0)
        if direction in (SteeringDirection.LEFT, SteeringDirection.RIGHT):
            return (0, 200, 255)
        return (120, 120, 120)
