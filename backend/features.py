"""
features.py
-----------
Orientation-, scale- and translation-invariant features derived from the 21
MediaPipe hand landmarks.

Why this module exists
======================
Raw MediaPipe landmarks are absolute, per-frame image coordinates. Feeding them
(or naive distance comparisons on them) into a classifier makes recognition
depend on *how the hand happens to be posed in the frame*:

* **Translation** — where the hand sits in the image.
* **Scale** — how far the hand is from the camera.
* **In-plane rotation** — how tilted/rotated the hand is.
* **Perspective / out-of-plane tilt** — the hand angled toward/away from the
  camera, which foreshortens 2-D distances.

This module turns landmarks into representations that are stable under all of
the above so the classifier sees the *same* numbers for the same gesture
regardless of orientation:

* :func:`normalize_landmarks` produces a canonical landmark set — translated to
  the wrist, scaled by a stable reference length, and rotated so the hand's
  in-plane orientation is fixed. Useful for logging, debugging and as a
  drop-in feature vector for any future trained model.
* :func:`joint_angle` / :func:`finger_curl_angles` measure finger curl as the
  interior angle at a joint. Angles between vectors are naturally invariant to
  translation, scale *and* rotation, and are much more robust to perspective
  foreshortening than radial distances — this is the signal the rule-based
  classifier actually uses.
* :func:`augment_landmarks` synthetically rotates/scales/jitters/mirrors a
  landmark set. It multiplies the effective variety of any collected samples
  (data augmentation) and lets the test-suite assert that a rotated hand yields
  the same classification as an upright one.

Everything here is pure-Python ``math`` so it has no third-party dependency and
can be unit-tested without a camera, model, or NumPy.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Landmark indices (MediaPipe Hands ordering)
# --------------------------------------------------------------------------- #
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

# (mcp, pip, tip) landmark triples for the four curling fingers. The curl of a
# finger is captured by the interior angle at its PIP joint, spanning MCP..TIP.
FINGER_ANGLE_JOINTS: Dict[str, Tuple[int, int, int]] = {
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}

Point = Sequence[float]


# --------------------------------------------------------------------------- #
# Small vector helpers (operate on (x, y, z) tuples)
# --------------------------------------------------------------------------- #
def _sub(a: Point, b: Point) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Point) -> float:
    return math.sqrt(_dot(a, a))


def _dist(a: Point, b: Point) -> float:
    return _norm(_sub(a, b))


def angle_between(v1: Point, v2: Point) -> float:
    """Angle (degrees) between two 3-D vectors, in [0, 180].

    Returns 0.0 if either vector is (near) zero length, so degenerate landmark
    triples don't blow up.
    """
    n1 = _norm(v1)
    n2 = _norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    # Clamp to guard against tiny floating-point overshoot outside [-1, 1].
    cos = max(-1.0, min(1.0, _dot(v1, v2) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def joint_angle(a: Point, b: Point, c: Point, use_z: bool = True) -> float:
    """Interior angle (degrees) at vertex ``b`` formed by points a-b-c.

    For a finger this is the bend at a knuckle: ~180 when the bone is straight
    (finger extended) and small when the joint is folded (finger curled).
    Because it is an angle between two vectors it does not depend on where the
    hand is, how big it is, or how it is rotated.

    ``use_z=False`` measures the angle in the 2-D image plane (x, y only). This
    is what the curl detector uses: MediaPipe's ``z`` is a coarse, noisy
    relative-depth estimate, and folding it into the angle makes even straight
    fingers read as slightly bent. The image-plane angle avoids that while
    staying robust to the hand tilting toward/away from the camera — a straight
    finger projects to a straight line under any viewing angle (a pinhole camera
    maps straight lines to straight lines), so its projected joint angle stays
    ~180 regardless of tilt.
    """
    v1 = _sub(a, b)
    v2 = _sub(c, b)
    if not use_z:
        v1 = (v1[0], v1[1], 0.0)
        v2 = (v2[0], v2[1], 0.0)
    return angle_between(v1, v2)


# --------------------------------------------------------------------------- #
# Canonical / normalized landmarks
# --------------------------------------------------------------------------- #
def hand_scale(landmarks: Sequence[Point]) -> float:
    """A stable reference length: wrist -> middle-finger MCP distance.

    This span barely changes as fingers open/close, so dividing by it removes
    the effect of how large the hand appears (i.e. its distance from camera).
    """
    size = _dist(landmarks[WRIST], landmarks[MIDDLE_MCP])
    return size if size > 1e-9 else 1e-9


def normalize_landmarks(landmarks: Sequence[Point]) -> List[Tuple[float, float, float]]:
    """Return landmarks in a canonical, orientation-independent frame.

    Steps:

    1. **Translate** so the wrist is at the origin — absolute position in the
       frame no longer matters.
    2. **Scale** by the wrist->middle-MCP distance — apparent hand size /
       distance from the camera no longer matters.
    3. **Rotate** in the image plane so the wrist->middle-MCP vector points
       straight "up" (the -y axis) — in-plane rotation no longer matters, so a
       tilted hand yields the same coordinates as an upright one.

    The z axis is translated and scaled but not rotated (rotation is done only
    in the image plane, which is where MediaPipe's x/y are reliable; z is a
    coarse relative-depth estimate).

    Returns a fresh list of 21 (x, y, z) tuples. Input is left untouched.
    """
    wrist = landmarks[WRIST]
    scale = hand_scale(landmarks)

    # Translate + scale.
    translated = [
        ((p[0] - wrist[0]) / scale, (p[1] - wrist[1]) / scale, (p[2] - wrist[2]) / scale)
        for p in landmarks
    ]

    # Reference direction (wrist -> middle MCP) after translation.
    ref = translated[MIDDLE_MCP]
    theta = math.atan2(ref[1], ref[0])          # current angle of the reference
    target = -math.pi / 2                        # want it pointing "up" (-y)
    rot = target - theta
    cos_r, sin_r = math.cos(rot), math.sin(rot)

    rotated: List[Tuple[float, float, float]] = []
    for x, y, z in translated:
        rotated.append((x * cos_r - y * sin_r, x * sin_r + y * cos_r, z))
    return rotated


# --------------------------------------------------------------------------- #
# Angle features (the classification signal)
# --------------------------------------------------------------------------- #
def finger_curl_angles(landmarks: Sequence[Point]) -> Dict[str, float]:
    """Interior PIP angle (degrees) for each of the four curling fingers.

    ~180 = straight/extended, small = folded/curled. Measured in the image
    plane (see :func:`joint_angle`): rotation-, scale- and translation-
    invariant, and robust to the hand tilting toward/away from the camera
    (unlike a projected tip-to-wrist distance, which foreshortens).
    """
    angles: Dict[str, float] = {}
    for name, (mcp, pip, tip) in FINGER_ANGLE_JOINTS.items():
        angles[name] = joint_angle(
            landmarks[mcp], landmarks[pip], landmarks[tip], use_z=False
        )
    return angles


def thumb_curl_angle(landmarks: Sequence[Point]) -> float:
    """Interior angle at the thumb IP joint (MCP-IP-TIP), in degrees (2-D)."""
    return joint_angle(
        landmarks[THUMB_MCP], landmarks[THUMB_IP], landmarks[THUMB_TIP], use_z=False
    )


def feature_vector(landmarks: Sequence[Point]) -> List[float]:
    """A compact invariant feature vector suitable for a trained classifier.

    Concatenates the flattened normalized landmarks (21 * 3 = 63 values) with
    the five finger curl angles (normalized to [0, 1]). Not used by the current
    rule-based classifier, but provided so a future SVM / small NN can be
    trained on rotation/scale-invariant inputs rather than raw coordinates.
    """
    norm = normalize_landmarks(landmarks)
    vec: List[float] = [c for p in norm for c in p]
    curls = finger_curl_angles(landmarks)
    for name in ("index", "middle", "ring", "pinky"):
        vec.append(curls[name] / 180.0)
    vec.append(thumb_curl_angle(landmarks) / 180.0)
    return vec


# --------------------------------------------------------------------------- #
# Augmentation (data-augmentation for ML + invariance testing)
# --------------------------------------------------------------------------- #
def augment_landmarks(
    landmarks: Sequence[Point],
    rotation_deg: float = 0.0,
    scale: float = 1.0,
    translate: Tuple[float, float] = (0.0, 0.0),
    jitter: float = 0.0,
    mirror: bool = False,
    rng: Optional[random.Random] = None,
) -> List[Tuple[float, float, float]]:
    """Return a synthetically transformed copy of a landmark set.

    Rotations/scaling are applied about the wrist so the hand's pose changes
    without teleporting it. This synthesises the off-angle / different-distance
    / left-vs-right-hand variety that a physically re-collected dataset would
    provide, and lets tests confirm the classifier is invariant to it.

    Args:
        rotation_deg: in-plane rotation about the wrist, degrees.
        scale: multiply hand size about the wrist (distance-from-camera proxy).
        translate: (dx, dy) shift applied after rotation/scale.
        jitter: stddev of Gaussian noise added to each coordinate (landmark
            detection noise).
        mirror: flip horizontally about the wrist (emulate the other hand).
        rng: optional random.Random for reproducible jitter.
    """
    wrist = landmarks[WRIST]
    rot = math.radians(rotation_deg)
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    r = rng or random
    dx, dy = translate

    out: List[Tuple[float, float, float]] = []
    for p in landmarks:
        # Work relative to the wrist.
        x, y, z = p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]
        if mirror:
            x = -x
        # Scale about the wrist.
        x, y, z = x * scale, y * scale, z * scale
        # Rotate in the image plane about the wrist.
        rx = x * cos_r - y * sin_r
        ry = x * sin_r + y * cos_r
        # Back to absolute coords + translation.
        nx = rx + wrist[0] + dx
        ny = ry + wrist[1] + dy
        nz = z + wrist[2]
        if jitter:
            nx += r.gauss(0.0, jitter)
            ny += r.gauss(0.0, jitter)
            nz += r.gauss(0.0, jitter)
        out.append((nx, ny, nz))
    return out
