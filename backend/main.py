"""
main.py
-------
FastAPI server for the camera-based Rock-Paper-Scissors game.

Endpoints:
    GET  /                -> serves the frontend single-page app
    GET  /api/health      -> liveness/readiness check
    POST /api/play        -> resolve one round given the player's move
    WS   /ws              -> stream base64 JPEG frames, receive hand analysis

Frame analysis flow (per WebSocket message):
    client sends {"type": "frame", "data": "<base64 jpeg>"}
    server replies {"type": "analysis", gesture, confidence, landmarks, ...}

The heavy MediaPipe work is run in a threadpool so the async event loop is not
blocked while a frame is processed.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import os
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from game_logic import MOVES, play_round
from features import finger_curl_angles
from hand_detector import HandDetector
from pipeline import analyze

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("rps.server")

# Gestures at or below this confidence are downgraded to "unknown" server-side
# so the client never acts on a low-confidence guess (the client applies its own
# stricter threshold + multi-frame voting at the "Shoot!" moment). Overridable
# via the RPS_MIN_CONFIDENCE env var for tuning.
MIN_CONFIDENCE = float(os.environ.get("RPS_MIN_CONFIDENCE", "0.35"))

# Set RPS_LOG_FAILURES=1 to log low-confidence / unknown detections (with their
# raw landmarks) so failure patterns can be inspected during testing.
LOG_FAILURES = os.environ.get("RPS_LOG_FAILURES", "").strip().lower() in {"1", "true", "yes"}

# Set RPS_DEBUG_ANGLES=1 to log the per-finger curl angles of every detected
# hand — a diagnostic for tuning the extended/curled thresholds against real
# camera data.
DEBUG_ANGLES = os.environ.get("RPS_DEBUG_ANGLES", "").strip().lower() in {"1", "true", "yes"}

# Set RPS_DUMP_DIR=<path> to save the first few received frames as PNGs, so the
# exact image fed to MediaPipe can be inspected (resolution, distortion, etc.).
DUMP_DIR = os.environ.get("RPS_DUMP_DIR", "").strip()
DUMP_MAX = 12
_dump_count = 0

# Resolve the frontend directory relative to this file (../frontend).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "frontend"))

app = FastAPI(title="Camera Rock-Paper-Scissors", version="1.0.0")


@app.middleware("http")
async def _no_cache(request: Request, call_next):
    """Serve HTML/JS/CSS uncached so the browser can't run a stale frontend.

    Without this, browsers hold on to old copies of index.html / app.js /
    style.css (even the cache-busting query trick fails, because the cached
    index.html keeps referencing the old asset URLs). Cheap and correct for a
    local single-user dev app; drop or scope to assets if this is ever put
    behind a CDN.
    """
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


# --------------------------------------------------------------------------- #
# REST models and endpoints
# --------------------------------------------------------------------------- #
class PlayRequest(BaseModel):
    player_move: str

    @field_validator("player_move")
    @classmethod
    def _valid_move(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in MOVES:
            raise ValueError(f"player_move must be one of {MOVES}")
        return value


@app.get("/api/health")
async def health() -> JSONResponse:
    """Report server health and whether the detection model loaded."""
    return JSONResponse(
        {"status": "ok", "model_loaded": _model_ok, "moves": list(MOVES)}
    )


@app.post("/api/play")
async def play(req: PlayRequest) -> JSONResponse:
    """
    Resolve a single round. The frontend calls this once it has captured a
    valid player gesture at the end of the countdown; the server picks the
    computer's move and returns the outcome.
    """
    outcome = play_round(req.player_move)
    logger.info(
        "Round: player=%s computer=%s -> %s",
        outcome["player_move"],
        outcome["computer_move"],
        outcome["result"],
    )
    return JSONResponse(outcome)


# --------------------------------------------------------------------------- #
# Frame decoding + analysis helpers
# --------------------------------------------------------------------------- #
def _decode_frame(data_url_or_b64: str) -> Optional[np.ndarray]:
    """
    Decode a base64 (optionally data-URL-prefixed) JPEG into a BGR image.

    Returns None on malformed input so the caller can report an error without
    tearing down the connection.
    """
    try:
        payload = data_url_or_b64
        if payload.startswith("data:"):
            # Strip a "data:image/jpeg;base64," prefix if present.
            payload = payload.split(",", 1)[1]
        raw = base64.b64decode(payload, validate=False)
        buffer = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        return image
    except (binascii.Error, ValueError, IndexError):
        return None


def _analyse(detector: HandDetector, image_bgr: np.ndarray) -> dict:
    """
    Detect and classify the primary hand in a frame.

    Runs synchronously (called inside a threadpool). Returns a JSON-serialisable
    analysis dict.
    """
    # Optional diagnostic: dump the first few received frames to disk.
    global _dump_count
    if DUMP_DIR and _dump_count < DUMP_MAX:
        try:
            os.makedirs(DUMP_DIR, exist_ok=True)
            path = os.path.join(DUMP_DIR, f"frame_{_dump_count:03d}.png")
            cv2.imwrite(path, image_bgr)
            _dump_count += 1
        except Exception:
            logger.exception("Frame dump failed")

    # Shared detect + classify path (identical to the standalone detector).
    det = analyze(detector, image_bgr, MIN_CONFIDENCE)

    if DEBUG_ANGLES:
        logger.info(
            "frame shape=%s hands=%d gesture=%s conf=%.2f",
            getattr(image_bgr, "shape", None), det.hand_count, det.gesture, det.confidence,
        )

    if det.gesture == "none":
        return {
            "type": "analysis",
            "gesture": "none",
            "confidence": 0.0,
            "landmarks": [],
            "connections": detector.connections,
            "hand_count": 0,
            "handedness": None,
            "message": "No hand detected — show your hand clearly.",
        }

    landmarks = det.landmarks or []

    if DEBUG_ANGLES:
        angs = finger_curl_angles(landmarks)
        logger.info(
            "angles idx=%.0f mid=%.0f ring=%.0f pinky=%.0f -> %s (%.2f)",
            angs["index"], angs["middle"], angs["ring"], angs["pinky"],
            det.gesture, det.confidence,
        )

    # Log failures (unknown / low-confidence, both surface as 'unknown') for
    # offline inspection when asked.
    if LOG_FAILURES and det.gesture == "unknown":
        logger.info(
            "Low-confidence/unknown detection: states=%s conf=%.3f landmarks=%s",
            det.finger_states,
            det.confidence,
            [[round(x, 4), round(y, 4), round(z, 4)] for x, y, z in landmarks],
        )

    message = None
    if det.hand_count > 1:
        message = "Multiple hands detected — using the largest one."
    elif det.gesture == "unknown":
        message = "Gesture unclear — reposition your hand and make a clear rock, paper or scissors."

    return {
        "type": "analysis",
        "gesture": det.gesture,
        "confidence": det.confidence,
        # Send x,y (and z) so the frontend can draw the skeleton overlay.
        "landmarks": [[round(x, 5), round(y, 5), round(z, 5)] for x, y, z in landmarks],
        "connections": detector.connections,
        "hand_count": det.hand_count,
        "handedness": det.handedness,
        "message": message,
    }


# --------------------------------------------------------------------------- #
# WebSocket endpoint
# --------------------------------------------------------------------------- #
@app.websocket("/ws")
async def ws_frames(websocket: WebSocket) -> None:
    """
    Receive a stream of camera frames and return hand analysis for each.

    A dedicated HandDetector is created per connection (MediaPipe state is not
    shareable across connections). The connection is cleaned up on disconnect
    or error.
    """
    await websocket.accept()
    peer = websocket.client.host if websocket.client else "unknown"
    logger.info("WebSocket connected: %s", peer)

    try:
        detector = HandDetector()
    except Exception:
        logger.exception("Could not create HandDetector for %s", peer)
        await websocket.send_json(
            {"type": "error", "message": "Hand detection model failed to load."}
        )
        await websocket.close()
        return

    try:
        while True:
            message = await websocket.receive_json()

            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if message.get("type") != "frame" or "data" not in message:
                await websocket.send_json(
                    {"type": "error", "message": "Expected {type:'frame', data:...}."}
                )
                continue

            image = _decode_frame(message["data"])
            if image is None:
                await websocket.send_json(
                    {"type": "error", "message": "Malformed frame — could not decode."}
                )
                continue

            # Offload the CPU-heavy detection to a worker thread.
            analysis = await asyncio.to_thread(_analyse, detector, image)
            await websocket.send_json(analysis)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", peer)
    except Exception:  # pragma: no cover - defensive catch-all
        logger.exception("Unexpected WebSocket error for %s", peer)
        try:
            await websocket.send_json(
                {"type": "error", "message": "Internal server error."}
            )
        except Exception:
            pass
    finally:
        detector.close()


# --------------------------------------------------------------------------- #
# Frontend static hosting (mounted last so /api and /ws take precedence)
# --------------------------------------------------------------------------- #
@app.get("/")
async def index() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if os.path.isdir(FRONTEND_DIR):
    # Serve app.js / style.css and any other static assets.
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
else:  # pragma: no cover
    logger.warning("Frontend directory not found at %s", FRONTEND_DIR)


# Attempt an eager model load so /api/health can report readiness and any
# model-load failure surfaces at startup rather than on the first frame.
_model_ok = False


@app.on_event("startup")
async def _warm_up() -> None:
    global _model_ok
    try:
        probe = HandDetector()
        # Run one tiny blank frame through to force graph initialisation.
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        await asyncio.to_thread(probe.detect, blank)
        probe.close()
        _model_ok = True
        logger.info("MediaPipe Hands model warmed up successfully.")
    except Exception:
        logger.exception("Model warm-up failed")
        _model_ok = False


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
