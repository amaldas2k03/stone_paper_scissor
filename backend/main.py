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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from game_logic import MOVES, play_round
from gesture_classifier import classify
from hand_detector import HandDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("rps.server")

# Resolve the frontend directory relative to this file (../frontend).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "frontend"))

app = FastAPI(title="Camera Rock-Paper-Scissors", version="1.0.0")


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
    hands_landmarks, handedness = detector.detect(image_bgr)
    hand_count = len(hands_landmarks)

    if hand_count == 0:
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

    # With multiple hands, play with the largest (closest to camera).
    idx = detector.largest_hand_index(hands_landmarks) or 0
    landmarks = hands_landmarks[idx]
    gesture, confidence, _states = classify(landmarks)

    message = None
    if hand_count > 1:
        message = "Multiple hands detected — using the largest one."
    elif gesture == "unknown":
        message = "Gesture unclear — make a clear rock, paper or scissors."

    return {
        "type": "analysis",
        "gesture": gesture,
        "confidence": confidence,
        # Send x,y (and z) so the frontend can draw the skeleton overlay.
        "landmarks": [[round(x, 5), round(y, 5), round(z, 5)] for x, y, z in landmarks],
        "connections": detector.connections,
        "hand_count": hand_count,
        "handedness": handedness[idx] if idx < len(handedness) else None,
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
