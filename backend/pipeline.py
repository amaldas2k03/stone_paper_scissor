"""
pipeline.py
-----------
The single shared "detect a hand sign in one frame" routine.

Both entry points use this so they cannot drift:

    detect_sign.py  -> standalone webcam detector
    main.py         -> FastAPI WebSocket server (the browser UI)

Given a decoded BGR frame and a live :class:`HandDetector`, it runs detection,
picks the primary (largest / closest) hand, classifies the gesture with the
orientation-invariant classifier, and applies the confidence floor. Callers add
their own presentation layer on top (an OpenCV window, or a JSON envelope).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from gesture_classifier import classify
from hand_detector import HandDetector

# Below this confidence a recognised gesture is reported as "unknown" instead of
# a shaky guess. Callers may override per call.
DEFAULT_MIN_CONFIDENCE = 0.35

Landmark = Tuple[float, float, float]


@dataclass
class Detection:
    """Result of analysing one frame.

    gesture: 'rock' | 'paper' | 'scissors' | 'unknown' | 'none'
        'none' means no hand was found; 'unknown' means a hand was found but the
        pose was ambiguous or below the confidence floor.
    confidence: 0..1 (0.0 for 'none'/'unknown').
    landmarks: the 21 (x, y, z) landmarks of the chosen hand, or None.
    finger_states: per-finger extended (True) / curled (False), empty if no hand.
    hand_count: how many hands were detected in the frame.
    handedness: 'Left' / 'Right' / 'Unknown' for the chosen hand, or None.
    """

    gesture: str
    confidence: float
    landmarks: Optional[List[Landmark]] = None
    finger_states: Dict[str, bool] = field(default_factory=dict)
    hand_count: int = 0
    handedness: Optional[str] = None


def analyze(
    detector: HandDetector,
    image_bgr: np.ndarray,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> Detection:
    """Detect + classify the primary hand in a single BGR frame."""
    hands_landmarks, handedness = detector.detect(image_bgr)
    hand_count = len(hands_landmarks)
    if hand_count == 0:
        return Detection(gesture="none", confidence=0.0, hand_count=0)

    # With multiple hands, use the largest (closest to the camera).
    idx = detector.largest_hand_index(hands_landmarks) or 0
    landmarks = hands_landmarks[idx]
    gesture, confidence, states = classify(landmarks)

    # Runtime safeguard: reject low-confidence guesses.
    if gesture != "unknown" and confidence < min_confidence:
        gesture, confidence = "unknown", 0.0

    hand_label = handedness[idx] if idx < len(handedness) else None
    return Detection(
        gesture=gesture,
        confidence=confidence,
        landmarks=landmarks,
        finger_states=states,
        hand_count=hand_count,
        handedness=hand_label,
    )
