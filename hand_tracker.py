"""
Hand detection and landmark extraction using MediaPipe Hands.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


# MediaPipe landmark indices used for steering input.
WRIST = 0
THUMB_TIP = 4
INDEX_FINGER_TIP = 8


@dataclass
class HandData:
    """Normalized and pixel-space landmarks for one detected hand."""

    label: str
    wrist: Tuple[float, float]
    thumb_tip: Tuple[float, float]
    index_tip: Tuple[float, float]
    center: Tuple[float, float]
    pixel_wrist: Tuple[int, int]
    pixel_thumb: Tuple[int, int]
    pixel_index: Tuple[int, int]
    pixel_center: Tuple[int, int]


class HandTracker:
    """
    Wrap MediaPipe Hands to detect up to two hands and expose steering landmarks.

    Tracks wrist, thumb tip, and index finger tip for each hand. The center
    point is the average of those three landmarks and is used for angle math.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        model_complexity: int = 0,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, List[HandData]]:
        """
        Detect hands in a BGR webcam frame and return annotated output plus data.

        Args:
            frame_bgr: Input frame from OpenCV (BGR).

        Returns:
            A tuple of (annotated_frame, list_of_hand_data).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)

        frame_rgb.flags.writeable = True
        annotated = frame_bgr.copy()
        height, width = annotated.shape[:2]
        hand_data: List[HandData] = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness,
            ):
                label = handedness.classification[0].label.lower()
                self._mp_drawing.draw_landmarks(
                    annotated,
                    landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_styles.get_default_hand_landmarks_style(),
                    self._mp_styles.get_default_hand_connections_style(),
                )

                parsed = self._extract_hand_data(landmarks, label, width, height)
                if parsed is not None:
                    hand_data.append(parsed)
                    self._draw_tracking_points(annotated, parsed)

        hand_data = self._sort_hands_left_to_right(hand_data)
        self._draw_hand_connection(annotated, hand_data)
        return annotated, hand_data

    @staticmethod
    def _draw_hand_connection(frame: np.ndarray, hands: List[HandData]) -> None:
        """Draw a line between both hands so tilt angle is visible on screen."""
        if len(hands) < 2:
            return
        cv2.line(
            frame,
            hands[0].pixel_center,
            hands[1].pixel_center,
            (255, 128, 0),
            3,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(frame, hands[0].pixel_center, 10, (255, 128, 0), 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, hands[1].pixel_center, 10, (255, 128, 0), 2, lineType=cv2.LINE_AA)

    def _extract_hand_data(
        self,
        landmarks,
        label: str,
        width: int,
        height: int,
    ) -> Optional[HandData]:
        """Convert MediaPipe landmarks into normalized and pixel coordinates."""
        wrist = self._landmark_to_point(landmarks.landmark[WRIST], width, height)
        thumb = self._landmark_to_point(landmarks.landmark[THUMB_TIP], width, height)
        index_tip = self._landmark_to_point(
            landmarks.landmark[INDEX_FINGER_TIP],
            width,
            height,
        )
        center = (
            (wrist[0] + thumb[0] + index_tip[0]) / 3.0,
            (wrist[1] + thumb[1] + index_tip[1]) / 3.0,
        )

        return HandData(
            label=label,
            wrist=wrist,
            thumb_tip=thumb,
            index_tip=index_tip,
            center=center,
            pixel_wrist=(int(wrist[0]), int(wrist[1])),
            pixel_thumb=(int(thumb[0]), int(thumb[1])),
            pixel_index=(int(index_tip[0]), int(index_tip[1])),
            pixel_center=(int(center[0]), int(center[1])),
        )

    @staticmethod
    def _landmark_to_point(landmark, width: int, height: int) -> Tuple[float, float]:
        """Map a normalized MediaPipe landmark to pixel coordinates."""
        return (landmark.x * width, landmark.y * height)

    @staticmethod
    def _sort_hands_left_to_right(hands: List[HandData]) -> List[HandData]:
        """Order detected hands by horizontal wrist position (left to right)."""
        return sorted(hands, key=lambda hand: hand.center[0])

    @staticmethod
    def _draw_tracking_points(frame: np.ndarray, hand: HandData) -> None:
        """Highlight wrist, thumb, and index landmarks used for steering."""
        colors = {
            "wrist": (0, 255, 255),
            "thumb": (255, 0, 255),
            "index": (0, 255, 0),
        }
        points = [
            (hand.pixel_wrist, colors["wrist"]),
            (hand.pixel_thumb, colors["thumb"]),
            (hand.pixel_index, colors["index"]),
        ]
        for point, color in points:
            cv2.circle(frame, point, 6, color, -1, lineType=cv2.LINE_AA)

        cv2.line(
            frame,
            hand.pixel_wrist,
            hand.pixel_index,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()

    @staticmethod
    def summarize_detection(hands: List[HandData]) -> Dict[str, Optional[HandData]]:
        """
        Split the hand list into left and right entries for steering logic.

        When only one hand is visible, the missing side is returned as None.
        """
        if not hands:
            return {"left": None, "right": None}

        if len(hands) == 1:
            single = hands[0]
            if single.label == "left":
                return {"left": single, "right": None}
            return {"left": None, "right": single}

        return {"left": hands[0], "right": hands[1]}
