"""
hand_detector.py
----------------
Thin wrapper around MediaPipe Hands that turns a decoded image into a list of
21 hand landmarks.

A separate HandDetector instance should be created per WebSocket connection:
MediaPipe's ``Hands`` object is stateful (it tracks hands across frames) and is
not safe to share across concurrent connections.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - surfaced clearly at startup
    raise ImportError(
        "mediapipe is required. Install backend dependencies with "
        "`pip install -r requirements.txt`."
    ) from exc


# One landmark = (x, y, z), all normalised to [0, 1] by MediaPipe.
Landmark = Tuple[float, float, float]


class HandDetector:
    """Detects hands in a BGR image and returns their landmarks."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        try:
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,  # keep tracking state across frames
                max_num_hands=max_num_hands,
                model_complexity=0,       # fastest model -> lower latency
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except Exception:  # pragma: no cover - model/asset load failure
            logger.exception("Failed to initialise MediaPipe Hands model")
            raise
        # Connection list used by the frontend to draw the skeleton overlay.
        self.connections = [list(c) for c in self._mp_hands.HAND_CONNECTIONS]

    def detect(
        self, image_bgr: np.ndarray
    ) -> Tuple[List[List[Landmark]], List[str]]:
        """
        Run hand detection on a single BGR frame.

        Args:
            image_bgr: HxWx3 uint8 image in BGR order (as decoded by OpenCV).

        Returns:
            (hands_landmarks, handedness) where hands_landmarks is a list with
            one entry per detected hand, each being a list of 21 (x, y, z)
            landmarks, and handedness is a parallel list of "Left"/"Right".
            Both lists are empty when no hand is found.
        """
        # MediaPipe expects a C-contiguous RGB image. cv2.cvtColor both
        # reorders the channels and returns a contiguous array (a plain
        # ``[:, :, ::-1]`` view is not contiguous and MediaPipe rejects it).
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(image_rgb)

        hands_landmarks: List[List[Landmark]] = []
        handedness: List[str] = []

        if not results.multi_hand_landmarks:
            return hands_landmarks, handedness

        labels = results.multi_handedness or []
        for idx, hand in enumerate(results.multi_hand_landmarks):
            landmarks: List[Landmark] = [
                (lm.x, lm.y, lm.z) for lm in hand.landmark
            ]
            hands_landmarks.append(landmarks)
            try:
                handedness.append(labels[idx].classification[0].label)
            except (IndexError, AttributeError):
                handedness.append("Unknown")

        return hands_landmarks, handedness

    @staticmethod
    def largest_hand_index(hands_landmarks: List[List[Landmark]]) -> Optional[int]:
        """
        Return the index of the largest hand by landmark bounding-box area.

        Used to pick a single hand to play with when several are in frame,
        preferring the one closest to the camera (largest on screen).
        """
        if not hands_landmarks:
            return None

        best_idx = 0
        best_area = -1.0
        for i, landmarks in enumerate(hands_landmarks):
            xs = [p[0] for p in landmarks]
            ys = [p[1] for p in landmarks]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            if area > best_area:
                best_area = area
                best_idx = i
        return best_idx

    def close(self) -> None:
        """Release the MediaPipe graph. Call when a connection ends."""
        try:
            self._hands.close()
        except Exception:  # pragma: no cover
            logger.debug("HandDetector already closed", exc_info=True)
