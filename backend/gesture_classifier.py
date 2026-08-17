"""
gesture_classifier.py
----------------------
Rule-based Rock-Paper-Scissors gesture classifier operating on the 21 hand
landmarks produced by MediaPipe Hands.

MediaPipe landmark indices (see https://google.github.io/mediapipe/solutions/hands):

        8   12  16  20        <- finger TIPs
        |   |   |   |
        7   11  15  19        <- DIP joints
        |   |   |   |
        6   10  14  18        <- PIP joints
        |   |   |   |
    4   5   9   13  17        <- MCP joints (4 = thumb TIP)
     \\  |   |   |   /
      3 |   |   |  /
       \\|   |   | /
        2   |   |/
         \\  |  /
          1 | /
           \\|/
            0                 <- wrist

The classification strategy is intentionally simple and orientation-tolerant:

* For each of the four fingers (index, middle, ring, pinky) we decide whether it
  is EXTENDED or CURLED by comparing how far the fingertip sits from the wrist
  versus how far the finger's PIP joint sits from the wrist. When a finger
  curls, the tip folds back toward the palm, so tip-to-wrist distance shrinks
  below pip-to-wrist distance. This ratio test works regardless of whether the
  hand points up, sideways, or is tilted, which a raw "is the tip above the
  joint?" y-coordinate test would not.

* The thumb is handled separately because it moves sideways rather than
  curling toward the wrist. We measure the thumb tip's distance to the index
  finger's MCP joint, normalised by overall hand size.

From the per-finger states the gesture is decided:

    rock     -> all four fingers curled (a closed fist)
    paper    -> all four fingers extended (an open hand)
    scissors -> only index and middle extended (a "V")
    unknown  -> anything that doesn't cleanly match the above

A confidence score in [0, 1] is derived from how decisively each finger sat on
its side of the extended/curled boundary, so ambiguous half-curled hands report
low confidence and can be rejected by the caller.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

# Landmark indices grouped for readability.
WRIST = 0
THUMB_TIP = 4
THUMB_IP = 3
INDEX_MCP = 5
MIDDLE_MCP = 9

# (tip, pip) landmark index pairs for the four curling fingers.
FINGER_JOINTS: Dict[str, Tuple[int, int]] = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}

# A finger is treated as clearly EXTENDED when tip/pip distance-to-wrist ratio is
# above EXTEND_THRESHOLD and clearly CURLED below CURL_THRESHOLD. The gap between
# them is a dead-zone that produces lower confidence.
EXTEND_THRESHOLD = 1.05
CURL_THRESHOLD = 0.85

Point = Sequence[float]


def _dist(a: Point, b: Point) -> float:
    """Euclidean distance between two (x, y, z) landmarks."""
    return math.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    )


def _hand_size(landmarks: Sequence[Point]) -> float:
    """A scale-invariant measure of hand size (wrist to middle-finger MCP)."""
    size = _dist(landmarks[WRIST], landmarks[MIDDLE_MCP])
    return size if size > 1e-6 else 1e-6


def _finger_extension_ratios(landmarks: Sequence[Point]) -> Dict[str, float]:
    """
    For each curling finger, return tip-to-wrist / pip-to-wrist distance.

    Ratio > 1 means the tip is farther from the wrist than the PIP joint, i.e.
    the finger is stretched out. Ratio < 1 means the tip has folded back toward
    the palm, i.e. the finger is curled.
    """
    wrist = landmarks[WRIST]
    ratios: Dict[str, float] = {}
    for name, (tip_idx, pip_idx) in FINGER_JOINTS.items():
        pip_dist = _dist(landmarks[pip_idx], wrist)
        tip_dist = _dist(landmarks[tip_idx], wrist)
        ratios[name] = tip_dist / pip_dist if pip_dist > 1e-6 else 0.0
    return ratios


def _thumb_extended(landmarks: Sequence[Point]) -> bool:
    """
    Decide whether the thumb is splayed out.

    Because the thumb abducts sideways rather than folding toward the wrist, we
    look at how far the thumb tip sits from the index-finger MCP joint relative
    to overall hand size. A tucked-in (fist) thumb sits close to that knuckle; a
    splayed (open-hand) thumb sits noticeably farther away.
    """
    spread = _dist(landmarks[THUMB_TIP], landmarks[INDEX_MCP])
    return spread / _hand_size(landmarks) > 0.9


def _finger_confidence(ratio: float, extended: bool) -> float:
    """
    Map how far a finger's ratio sits from the decision boundary into [0, 1].

    Fingers that are unambiguously extended or curled score near 1.0; fingers
    sitting inside the EXTEND/CURL dead-zone score lower.
    """
    if extended:
        # Distance above the extend threshold, saturating around +0.25.
        margin = (ratio - EXTEND_THRESHOLD) / 0.25
    else:
        # Distance below the curl threshold, saturating around -0.25.
        margin = (CURL_THRESHOLD - ratio) / 0.25
    return max(0.0, min(1.0, 0.5 + 0.5 * margin))


def classify(landmarks: Sequence[Point]) -> Tuple[str, float, Dict[str, bool]]:
    """
    Classify a hand pose into 'rock', 'paper', 'scissors', or 'unknown'.

    Args:
        landmarks: sequence of 21 (x, y, z) landmark coordinates. The absolute
            scale is irrelevant because every measurement is a ratio or is
            normalised by hand size, so MediaPipe's normalised [0, 1]
            coordinates work directly.

    Returns:
        (gesture, confidence, finger_states) where finger_states maps each
        finger name to a bool (True = extended). Confidence is in [0, 1].
    """
    if landmarks is None or len(landmarks) < 21:
        return "unknown", 0.0, {}

    ratios = _finger_extension_ratios(landmarks)

    # Per-finger extended/curled decision plus a confidence for that decision.
    states: Dict[str, bool] = {}
    confidences: List[float] = []
    for name, ratio in ratios.items():
        extended = ratio >= EXTEND_THRESHOLD
        # Treat the dead-zone as "curled" for the boolean, but reflect the
        # ambiguity through a reduced confidence.
        states[name] = extended
        confidences.append(_finger_confidence(ratio, extended))

    states["thumb"] = _thumb_extended(landmarks)

    index_up = states["index"]
    middle_up = states["middle"]
    ring_up = states["ring"]
    pinky_up = states["pinky"]

    # --- Match the four-finger pattern against each gesture -----------------
    if not any((index_up, middle_up, ring_up, pinky_up)):
        gesture = "rock"          # closed fist: every finger curled
    elif all((index_up, middle_up, ring_up, pinky_up)):
        gesture = "paper"         # open hand: every finger extended
    elif index_up and middle_up and not ring_up and not pinky_up:
        gesture = "scissors"      # V shape: index + middle only
    else:
        gesture = "unknown"       # ambiguous / transitional pose

    # Overall confidence is the mean of the four per-finger confidences,
    # dropped to zero for an unknown pose so callers can reject it cleanly.
    base_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    confidence = 0.0 if gesture == "unknown" else base_confidence

    return gesture, round(confidence, 3), states
