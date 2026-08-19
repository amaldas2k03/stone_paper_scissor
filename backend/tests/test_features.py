"""
Unit tests for the invariant feature module.

These verify the mathematical properties the classifier relies on:

* ``joint_angle`` returns the geometric angle we expect.
* ``normalize_landmarks`` collapses translated / scaled / rotated copies of the
  same hand onto (nearly) identical canonical coordinates.
* ``finger_curl_angles`` are stable under rotation and scale.
* ``augment_landmarks`` transforms as advertised.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import (  # noqa: E402
    WRIST,
    MIDDLE_MCP,
    angle_between,
    augment_landmarks,
    finger_curl_angles,
    joint_angle,
    normalize_landmarks,
)

# Import the synthetic hand builder from the classifier tests.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_gesture_classifier import _make_hand  # noqa: E402


def _max_coord_diff(a, b):
    return max(
        max(abs(pa[i] - pb[i]) for i in range(3)) for pa, pb in zip(a, b)
    )


def test_joint_angle_straight_and_right_angle():
    straight = joint_angle((0, 1, 0), (0, 0, 0), (0, -1, 0))
    assert abs(straight - 180.0) < 1e-6
    right = joint_angle((1, 0, 0), (0, 0, 0), (0, 1, 0))
    assert abs(right - 90.0) < 1e-6


def test_angle_between_handles_zero_vector():
    assert angle_between((0, 0, 0), (1, 0, 0)) == 0.0


def test_normalize_is_translation_invariant():
    hand = _make_hand(True, True, True, True)
    moved = augment_landmarks(hand, translate=(0.3, -0.2))
    assert _max_coord_diff(normalize_landmarks(hand), normalize_landmarks(moved)) < 1e-9


def test_normalize_is_scale_invariant():
    hand = _make_hand(True, True, False, False)
    scaled = augment_landmarks(hand, scale=2.3)
    assert _max_coord_diff(normalize_landmarks(hand), normalize_landmarks(scaled)) < 1e-9


def test_normalize_is_rotation_invariant():
    hand = _make_hand(False, False, False, False)
    rotated = augment_landmarks(hand, rotation_deg=57.0)
    assert _max_coord_diff(normalize_landmarks(hand), normalize_landmarks(rotated)) < 1e-6


def test_normalize_puts_wrist_at_origin_and_reference_up():
    hand = _make_hand(True, True, True, True)
    norm = normalize_landmarks(hand)
    # Wrist at origin.
    assert all(abs(c) < 1e-9 for c in norm[WRIST])
    # Reference (wrist->middle MCP) points straight up (-y), unit length.
    mx, my, _ = norm[MIDDLE_MCP]
    assert abs(mx) < 1e-6
    assert abs(my - (-1.0)) < 1e-6


def test_finger_curl_angles_stable_under_rotation_and_scale():
    hand = _make_hand(True, False, True, False)
    base = finger_curl_angles(hand)
    transformed = finger_curl_angles(
        augment_landmarks(hand, rotation_deg=123.0, scale=0.6)
    )
    for name in base:
        assert abs(base[name] - transformed[name]) < 1e-6


def test_augment_mirror_flips_x_about_wrist():
    hand = _make_hand(True, True, True, True)
    mirrored = augment_landmarks(hand, mirror=True)
    wx = hand[WRIST][0]
    for orig, flip in zip(hand, mirrored):
        assert abs((orig[0] - wx) + (flip[0] - wx)) < 1e-9  # x offsets negate
        assert abs(orig[1] - flip[1]) < 1e-9                # y unchanged
