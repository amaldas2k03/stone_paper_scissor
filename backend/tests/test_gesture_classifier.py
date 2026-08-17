"""
Unit tests for the rule-based gesture classifier.

Rather than depend on real camera data, these tests synthesise geometrically
valid 21-point landmark sets for an upright hand where each of the four fingers
is either extended (fingertip reaches up, away from the wrist) or curled
(fingertip folds back down toward the palm). Only the four curling fingers
affect the classification, so the thumb is placed in a fixed neutral spot.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_classifier import classify  # noqa: E402

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
            # Fingertip reaches upward, farther from the wrist than the PIP.
            lm[dip] = (x, 0.48, 0.0)
            lm[tip] = (x, 0.40, 0.0)
        else:
            # Fingertip folds back down toward the palm (closer to the wrist).
            lm[dip] = (x, 0.61, 0.0)
            lm[tip] = (x, 0.66, 0.0)
    return lm


def test_rock_is_fist():
    gesture, conf, _ = classify(_make_hand(False, False, False, False))
    assert gesture == "rock"
    assert conf > 0.5


def test_paper_is_open_hand():
    gesture, conf, _ = classify(_make_hand(True, True, True, True))
    assert gesture == "paper"
    assert conf > 0.5


def test_scissors_is_index_and_middle():
    gesture, conf, _ = classify(_make_hand(True, True, False, False))
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
