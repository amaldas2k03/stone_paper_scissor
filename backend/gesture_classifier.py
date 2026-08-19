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

Orientation robustness
======================
Earlier versions decided "extended vs curled" from the *distance* between the
fingertip and the wrist versus the PIP joint and the wrist. That is invariant to
in-plane rotation and scale, but it breaks when the hand tilts toward or away
from the camera: the projected fingertip-to-wrist distance foreshortens, so an
extended finger reads as curled.

This version instead measures each finger's curl by the **interior angle at its
PIP joint** (MCP->PIP->TIP), computed in ``features.py``:

* An angle between two vectors is inherently invariant to translation, scale and
  rotation — a rotated or moved hand produces the same angle.
* It is measured in the 2-D image plane (x, y). MediaPipe's z depth is a coarse,
  noisy estimate that, folded into the angle, makes even straight fingers read
  as slightly bent. The image-plane angle is still robust to perspective: a
  straight finger projects to a straight line at any viewing angle, so a finger
  pointing toward the camera still reads as ~180 degrees rather than "short".

An extended finger's PIP angle is near 180 degrees; a curled finger's is small.

From the per-finger extended/curled states the gesture is decided:

    rock     -> all four fingers curled (a closed fist)
    paper    -> all four fingers extended (an open hand)
    scissors -> only index and middle extended (a "V")
    unknown  -> anything that doesn't cleanly match the above

A confidence score in [0, 1] is derived from how decisively each finger's angle
sat on its side of the extended/curled boundary, so ambiguous half-curled hands
report low confidence and can be rejected by the caller.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from features import (
    finger_curl_angles,
    hand_scale,
    thumb_curl_angle,
    THUMB_TIP,
    INDEX_MCP,
    _dist,
)

Point = Sequence[float]

# A finger is treated as clearly EXTENDED when its PIP interior angle is at or
# above EXTEND_ANGLE, and clearly CURLED at or below CURL_ANGLE. The midpoint is
# the boolean decision boundary; the gap around it is a dead-zone that yields
# lower confidence. Values are in degrees.
EXTEND_ANGLE = 160.0
CURL_ANGLE = 110.0
MID_ANGLE = (EXTEND_ANGLE + CURL_ANGLE) / 2.0   # 135 -> boolean boundary
HALF_WIDTH = (EXTEND_ANGLE - CURL_ANGLE) / 2.0  # 25  -> confidence saturates here

# Thumb: an interior IP angle above this is "extended". The thumb does not
# affect the rock/paper/scissors decision (only the four fingers do), but the
# state is reported for callers/overlays.
THUMB_EXTEND_ANGLE = 150.0


def _finger_confidence(angle: float, extended: bool) -> float:
    """Map how far a finger's angle sits from the decision boundary into [0, 1].

    Fingers unambiguously extended (>= EXTEND_ANGLE) or curled (<= CURL_ANGLE)
    score near 1.0; fingers sitting inside the dead-zone around MID_ANGLE score
    lower, bottoming out at 0.5 exactly on the boundary.
    """
    if extended:
        margin = (angle - MID_ANGLE) / HALF_WIDTH
    else:
        margin = (MID_ANGLE - angle) / HALF_WIDTH
    return max(0.0, min(1.0, 0.5 + 0.5 * margin))


def _thumb_extended(landmarks: Sequence[Point]) -> bool:
    """Decide whether the thumb is splayed/straight rather than tucked in.

    Uses the thumb's IP-joint angle (rotation-invariant) and, as a sanity check
    for the fully-tucked fist where the thumb lies flat across the palm, its
    distance from the index MCP normalised by hand size. Either a clearly bent
    thumb or one hugging the index knuckle counts as "not extended".
    """
    angle = thumb_curl_angle(landmarks)
    spread = _dist(landmarks[THUMB_TIP], landmarks[INDEX_MCP]) / hand_scale(landmarks)
    return angle >= THUMB_EXTEND_ANGLE and spread > 0.5


def classify(landmarks: Sequence[Point]) -> Tuple[str, float, Dict[str, bool]]:
    """Classify a hand pose into 'rock', 'paper', 'scissors', or 'unknown'.

    Args:
        landmarks: sequence of 21 (x, y, z) landmark coordinates. Absolute
            position, scale and rotation are irrelevant because every signal is
            a joint angle; MediaPipe's normalised [0, 1] coordinates work
            directly.

    Returns:
        (gesture, confidence, finger_states) where finger_states maps each
        finger name to a bool (True = extended). Confidence is in [0, 1].
    """
    if landmarks is None or len(landmarks) < 21:
        return "unknown", 0.0, {}

    angles = finger_curl_angles(landmarks)

    # Per-finger extended/curled decision plus a confidence for that decision.
    states: Dict[str, bool] = {}
    confidences: List[float] = []
    for name in ("index", "middle", "ring", "pinky"):
        angle = angles[name]
        extended = angle >= MID_ANGLE
        states[name] = extended
        confidences.append(_finger_confidence(angle, extended))

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
