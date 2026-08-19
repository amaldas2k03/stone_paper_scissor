"""
Unit tests for the rule-based gesture classifier.

Rather than depend on real camera data, these tests synthesise geometrically
valid 21-point landmark sets for an upright hand where each of the four fingers
is either extended (fingertip reaches up, straight from the palm) or curled
(fingertip folds back down toward the palm). Only the four curling fingers
affect the classification, so the thumb is placed in a fixed neutral spot.

Beyond the basic upright cases, the suite also asserts the property that
motivated the rewrite: the classifier must return the **same** gesture when the
hand is rotated in-plane, scaled (further/closer to the camera), mirrored
(other hand), tilted out-of-plane (angled toward the camera), and jittered.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_classifier import classify  # noqa: E402
from features import WRIST, augment_landmarks  # noqa: E402

# Base x positions for each finger in an upright right hand.
_FINGER_X = {"index": 0.42, "middle": 0.50, "ring": 0.58, "pinky": 0.66}
# Landmark slots (mcp, pip, dip, tip) for each finger in the 21-point array.
_SLOTS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def _make_hand(index=True, middle=True, ring=True, pinky=True):
    """Build 21 (x, y, z) landmarks for an upright hand with the given fingers
    extended (True) or curled (False)."""
    lm = [(0.0, 0.0, 0.0)] * 21
    lm[0] = (0.50, 0.95, 0.0)                 # wrist (bottom-centre)
    # Thumb (indices 1-4), fixed neutral position; not used for the decision.
    lm[1] = (0.40, 0.85, 0.0)
    lm[2] = (0.34, 0.78, 0.0)
    lm[3] = (0.29, 0.72, 0.0)
    lm[4] = (0.25, 0.66, 0.0)

    states = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}
    for name, extended in states.items():
        x = _FINGER_X[name]
        mcp, pip, dip, tip = _SLOTS[name]
        lm[mcp] = (x, 0.65, 0.0)
        lm[pip] = (x, 0.55, 0.0)
        if extended:
            # Fingertip reaches upward in a straight line from the palm.
            lm[dip] = (x, 0.48, 0.0)
            lm[tip] = (x, 0.40, 0.0)
        else:
            # Fingertip folds back down toward the palm (closer to the wrist).
            lm[dip] = (x, 0.61, 0.0)
            lm[tip] = (x, 0.66, 0.0)
    return lm


# Canonical gesture hands, reused across the invariance tests.
_ROCK = _make_hand(False, False, False, False)
_PAPER = _make_hand(True, True, True, True)
_SCISSORS = _make_hand(True, True, False, False)
_GESTURES = {"rock": _ROCK, "paper": _PAPER, "scissors": _SCISSORS}


def _tilt_out_of_plane(landmarks, phi_deg):
    """Rotate the hand about the x-axis through the wrist by ``phi_deg``.

    This simulates the hand angling toward/away from the camera: the projected
    y-extent foreshortens while depth (z) grows. A 2-D distance-based curl test
    misreads this as curled; a 3-D angle-based one does not.
    """
    phi = math.radians(phi_deg)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    wx, wy, wz = landmarks[WRIST]
    out = []
    for x, y, z in landmarks:
        ry, rz = y - wy, z - wz
        out.append((x, ry * cos_p - rz * sin_p + wy, ry * sin_p + rz * cos_p + wz))
    return out


# --------------------------------------------------------------------------- #
# Baseline upright cases (unchanged behaviour)
# --------------------------------------------------------------------------- #
def test_rock_is_fist():
    gesture, conf, _ = classify(_ROCK)
    assert gesture == "rock"
    assert conf > 0.5


def test_paper_is_open_hand():
    gesture, conf, _ = classify(_PAPER)
    assert gesture == "paper"
    assert conf > 0.5


def test_scissors_is_index_and_middle():
    gesture, conf, _ = classify(_SCISSORS)
    assert gesture == "scissors"
    assert conf > 0.5


def test_ambiguous_pose_is_unknown():
    # Only the ring finger up is not a valid RPS gesture.
    gesture, conf, _ = classify(_make_hand(False, False, True, False))
    assert gesture == "unknown"
    assert conf == 0.0


def test_too_few_landmarks_is_unknown():
    gesture, conf, states = classify([(0.0, 0.0, 0.0)] * 5)
    assert gesture == "unknown"
    assert conf == 0.0
    assert states == {}


# --------------------------------------------------------------------------- #
# Orientation / scale / distance invariance — the point of the rewrite
# --------------------------------------------------------------------------- #
def test_invariant_to_in_plane_rotation():
    """Every gesture must survive rotation through a full turn."""
    for expected, hand in _GESTURES.items():
        for deg in range(0, 360, 15):
            rotated = augment_landmarks(hand, rotation_deg=deg)
            gesture, conf, _ = classify(rotated)
            assert gesture == expected, f"{expected} misread as {gesture} at {deg}deg"
            assert conf > 0.5


def test_invariant_to_scale_and_distance():
    """Hand held closer/further from the camera must not change the result."""
    for expected, hand in _GESTURES.items():
        for scale in (0.4, 0.7, 1.5, 2.5):
            scaled = augment_landmarks(hand, scale=scale)
            gesture, _, _ = classify(scaled)
            assert gesture == expected, f"{expected} misread as {gesture} at x{scale}"


def test_invariant_to_translation():
    """Where the hand sits in the frame must not matter."""
    for expected, hand in _GESTURES.items():
        for dx, dy in ((0.2, 0.0), (-0.15, 0.1), (0.0, -0.25)):
            moved = augment_landmarks(hand, translate=(dx, dy))
            assert classify(moved)[0] == expected


def test_invariant_to_mirror_other_hand():
    """Left vs right hand (a horizontal mirror) must classify the same."""
    for expected, hand in _GESTURES.items():
        mirrored = augment_landmarks(hand, mirror=True)
        assert classify(mirrored)[0] == expected


def test_invariant_to_out_of_plane_tilt():
    """Hand angled toward/away from the camera must still classify correctly.

    This is the case the old distance-ratio test failed on: foreshortening
    shrank the projected fingertip-to-wrist distance and made extended fingers
    look curled.
    """
    for expected, hand in _GESTURES.items():
        for phi in (-45, -30, -15, 15, 30, 45):
            tilted = _tilt_out_of_plane(hand, phi)
            gesture, _, _ = classify(tilted)
            assert gesture == expected, f"{expected} misread as {gesture} at tilt {phi}"


def test_invariant_under_combined_transform_with_jitter():
    """Rotation + scale + translation + landmark noise, all at once."""
    rng = random.Random(1234)
    for expected, hand in _GESTURES.items():
        for _ in range(20):
            noisy = augment_landmarks(
                hand,
                rotation_deg=rng.uniform(0, 360),
                scale=rng.uniform(0.5, 2.0),
                translate=(rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)),
                jitter=0.004,
                mirror=rng.random() < 0.5,
                rng=rng,
            )
            assert classify(noisy)[0] == expected
