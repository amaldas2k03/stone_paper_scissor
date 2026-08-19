"""
detect_sign.py
--------------
Standalone Rock-Paper-Scissors hand-sign detector.

Opens the webcam, runs MediaPipe Hands on each frame, classifies the gesture
with the orientation-invariant rule-based classifier, and shows the result live
(and/or prints it to the console). No web server, no frontend — just detection.

The detection itself is delegated to the shared, unit-tested modules that live
alongside this file:

    features.py            -> orientation/scale/tilt-invariant joint-angle features
    gesture_classifier.py  -> classify() -> (gesture, confidence, finger_states)
    pipeline.py            -> analyze() -> one-frame detect + classify (also used
                              by the web UI, so the two can't drift)

so this script stays purely about capture + display.

Usage
=====
    python detect_sign.py                 # windowed, default camera
    python detect_sign.py --camera 1      # pick a different webcam
    python detect_sign.py --no-window     # headless: just print to stdout
    python detect_sign.py --min-confidence 0.5
    python detect_sign.py --flip          # mirror the view (selfie-style)

Controls (windowed mode): press 'q' or Esc to quit.

Requirements: mediapipe, opencv-python, numpy (see requirements.txt). Run from
the backend/ directory, or with backend/ on PYTHONPATH, so the imports resolve.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter, deque
from typing import Deque, Optional, Tuple

# Ensure sibling modules import when the script is launched from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from hand_detector import HandDetector  # noqa: E402
from pipeline import analyze  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("detect_sign")

EMOJI = {"rock": "fist", "paper": "open", "scissors": "V", "unknown": "?", "none": "-"}
# BGR colours for the on-screen label per gesture.
COLOR = {
    "rock": (80, 80, 240),
    "paper": (100, 220, 100),
    "scissors": (240, 180, 60),
    "unknown": (150, 150, 150),
    "none": (120, 120, 120),
}


def _smooth(history: Deque[str]) -> str:
    """Majority vote over the recent gesture history for a stable read.

    Mirrors the frontend's shoot-moment voting: momentary misdetections get
    outvoted by the surrounding frames.
    """
    if not history:
        return "none"
    return Counter(history).most_common(1)[0][0]


def _draw(
    frame: np.ndarray,
    detector: HandDetector,
    landmarks,
    gesture: str,
    confidence: float,
    smoothed: str,
) -> None:
    """Draw the hand skeleton and a gesture label onto ``frame`` in place."""
    h, w = frame.shape[:2]

    if landmarks:
        pts = [(int(x * w), int(y * h)) for x, y, _ in landmarks]
        for a, b in detector.connections:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (200, 160, 255), 2)
        for p in pts:
            cv2.circle(frame, p, 4, (255, 120, 90), -1)

    color = COLOR.get(smoothed, (200, 200, 200))
    label = f"{smoothed.upper()}  ({EMOJI.get(smoothed, '?')})"
    sub = f"frame: {gesture} {confidence:.2f}"
    cv2.rectangle(frame, (0, 0), (w, 64), (30, 30, 30), -1)
    cv2.putText(frame, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, sub, (14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1)


def detect_once(
    detector: HandDetector, frame_bgr: np.ndarray, min_confidence: float
) -> Tuple[str, float, Optional[list]]:
    """Detect + classify the primary hand in one frame.

    Returns (gesture, confidence, landmarks). ``gesture`` is 'none' when no hand
    is present and 'unknown' when the pose is ambiguous or below the confidence
    floor. ``landmarks`` is None when no hand is found.
    """
    det = analyze(detector, frame_bgr, min_confidence)
    return det.gesture, det.confidence, det.landmarks


def run(camera: int, min_confidence: float, show_window: bool, flip: bool) -> int:
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        logger.error(
            "Could not open camera %d. Is a webcam connected and free "
            "(no other app / browser tab using it)?",
            camera,
        )
        return 2

    detector = HandDetector(max_num_hands=1)
    history: Deque[str] = deque(maxlen=6)
    last_printed = None
    logger.info("Detecting… (press 'q' or Esc in the window to quit)")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Dropped a camera frame.")
                time.sleep(0.02)
                continue
            if flip:
                frame = cv2.flip(frame, 1)

            gesture, confidence, landmarks = detect_once(detector, frame, min_confidence)
            history.append(gesture)
            smoothed = _smooth(history)

            # Console output: only when the stable reading changes, so stdout
            # stays readable in headless mode.
            if smoothed != last_printed:
                logger.info("sign = %-8s (latest frame: %s %.2f)", smoothed, gesture, confidence)
                last_printed = smoothed

            if show_window:
                _draw(frame, detector, landmarks, gesture, confidence, smoothed)
                cv2.imshow("RPS sign detector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or Esc
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        detector.close()
        if show_window:
            cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone RPS hand-sign detector.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0).")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.35,
        help="Below this, a gesture is reported as 'unknown' (default 0.35).",
    )
    parser.add_argument(
        "--no-window", action="store_true", help="Headless: print to stdout only."
    )
    parser.add_argument(
        "--flip", action="store_true", help="Mirror the view (selfie-style)."
    )
    args = parser.parse_args()
    return run(
        camera=args.camera,
        min_confidence=args.min_confidence,
        show_window=not args.no_window,
        flip=args.flip,
    )


if __name__ == "__main__":
    raise SystemExit(main())
