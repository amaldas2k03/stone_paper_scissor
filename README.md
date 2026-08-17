# ✊ ✋ ✌️ Camera Rock · Paper · Scissors

Play Rock-Paper-Scissors against the computer using your **webcam**. You show a
hand gesture to the camera, a Python backend recognises it with
[MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html), and
the browser runs the game — countdown, reveal, scoreboard and history.

- **Frontend** — plain HTML/CSS/JS single-page app: webcam capture, hand-skeleton
  overlay, `Rock… Paper… Scissors… Shoot!` countdown, scoreboard + match history.
- **Backend** — FastAPI server with a **WebSocket** endpoint that receives camera
  frames, detects the 21 hand landmarks, and classifies them into
  `rock` / `paper` / `scissors` / `unknown` with a rule-based classifier.

```
┌──────────────┐   base64 JPEG frames (WebSocket)   ┌───────────────────────┐
│   Browser    │ ─────────────────────────────────► │   FastAPI backend     │
│  (camera +   │                                    │  MediaPipe Hands →    │
│   game UI)   │ ◄───────────────────────────────── │  rule-based classifier│
└──────────────┘   gesture + landmarks (JSON)        └───────────────────────┘
        │  POST /api/play { player_move }  →  { computer_move, result }
        └──────────────────────────────────────────────────────────────────►
```

---

## Project layout

```
stone_paper_scissor/
├── backend/
│   ├── main.py               # FastAPI app: WebSocket /ws, REST /api/*, serves frontend
│   ├── hand_detector.py      # MediaPipe Hands wrapper (21 landmarks per hand)
│   ├── gesture_classifier.py # Rule-based landmark → rock/paper/scissors classifier
│   ├── game_logic.py         # Computer move + winner rules (pure, unit-tested)
│   ├── requirements.txt
│   └── tests/                # pytest unit tests (classifier + game logic)
├── frontend/
│   ├── index.html
│   ├── app.js                # Camera, WebSocket streaming, game state machine
│   └── style.css
└── README.md
```

---

## Requirements

- **Python 3.9–3.12** (tested on 3.11). MediaPipe does not yet support 3.13.
- A **webcam**.
- A modern browser (Chrome, Edge, or Firefox).

> **Camera + browser note:** `getUserMedia` only works in a *secure context*.
> `http://localhost` and `http://127.0.0.1` count as secure, so opening the app
> at `http://127.0.0.1:8000` works out of the box. Serving it from any other
> host requires HTTPS. The first time you open it the browser will ask for
> camera permission — click **Allow**.

---

## Setup & run

### 1. Backend

From the project root:

```bash
cd backend
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd):
# .venv\Scripts\activate.bat
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Start the server:

```bash
python main.py
```

or equivalently:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

You should see `MediaPipe Hands model warmed up successfully.` in the logs.

### 2. Frontend

The backend **serves the frontend for you**. Just open:

```
http://127.0.0.1:8000
```

Allow camera access when prompted, then press **Start Round**.

*(You don't need a separate web server — but if you prefer, you can serve the
`frontend/` folder with any static server and it will still connect back to the
WebSocket at the same host.)*

---

## How to play

1. Open the app and allow camera access — you'll see your mirrored feed with a
   hand skeleton drawn on top when a hand is detected.
2. Press **Start Round**. A `Rock… Paper… Scissors… Shoot!` countdown plays.
3. At **Shoot!**, hold your gesture clearly:
   - ✊ **Rock** — closed fist (all fingers curled)
   - ✋ **Paper** — open hand (all fingers extended)
   - ✌️ **Scissors** — index + middle fingers extended, others curled
4. Both moves are revealed and the round outcome (Win / Lose / Tie) is shown.
5. The **scoreboard** and **match history** update automatically and persist for
   the session (via `localStorage`). Use **Reset Score** to start over.

If no clear hand or an ambiguous gesture is detected at *Shoot!*, the round is
not counted — you'll be asked to try again.

---

## How gesture recognition works

The backend runs MediaPipe Hands on each frame to get **21 landmarks** per hand.
`gesture_classifier.py` then decides, for each of the four non-thumb fingers,
whether it is **extended** or **curled**:

> A finger is *extended* when its fingertip is **farther from the wrist** than
> its middle (PIP) joint, and *curled* when the tip folds back closer to the
> wrist. Using distances to the wrist (rather than raw up/down positions) keeps
> the test working whether the hand points up, sideways or is tilted.

From the finger states:

| Gesture  | Index | Middle | Ring | Pinky |
|----------|:-----:|:------:|:----:|:-----:|
| rock     |  ✗    |   ✗    |  ✗   |  ✗    |
| paper    |  ✓    |   ✓    |  ✓   |  ✓    |
| scissors |  ✓    |   ✓    |  ✗   |  ✗    |

Anything else is reported as `unknown`. A **confidence** score reflects how
decisively each finger sat on its extended/curled boundary, so half-curled,
in-between poses score low and are rejected. If **multiple hands** are in frame,
the largest (closest to the camera) is used. See the docstring in
[`gesture_classifier.py`](backend/gesture_classifier.py) for the full details.

---

## API

| Method | Path          | Description                                                     |
|--------|---------------|-----------------------------------------------------------------|
| `GET`  | `/`           | Serves the frontend app.                                         |
| `GET`  | `/api/health` | `{ status, model_loaded, moves }`.                              |
| `POST` | `/api/play`   | Body `{ "player_move": "rock" }` → `{ player_move, computer_move, result }`. |
| `WS`   | `/ws`         | Send `{ type:"frame", data:"<base64 jpeg>" }`; receive `{ type:"analysis", gesture, confidence, landmarks, connections, hand_count, handedness, message }`. |

---

## Testing

The classifier and game logic are decoupled from the web/camera layers and can
be tested on their own (no camera or model needed for `game_logic`; the
classifier tests use synthetic landmarks):

```bash
cd backend
pip install pytest
python -m pytest tests/ -q
```

---

## Assumptions & simplifications

- **Rule-based** classifier (the required path). The prompt's optional
  ML-classifier stretch goal is not implemented; the geometric rules are robust
  for clear, well-lit gestures. The rule engine is isolated in
  `gesture_classifier.py`, so a trained model could be dropped in behind the same
  `classify()` interface with the rules as a fallback.
- The game runs on **localhost over `ws://`**. For remote/HTTPS deployment you'd
  serve over `wss://` behind TLS.
- Score/history persistence is **per-browser** via `localStorage` (session-scoped
  as requested); there's no server-side database.
- Designed for **desktop/laptop webcams**; mobile is not specifically targeted.
- The computer plays a **uniformly random** move each round.

---

## Troubleshooting

- **"Camera permission was denied."** — Allow camera access in the browser's site
  settings and click **Try again**. Make sure you're on `http://127.0.0.1:8000`
  (or `localhost`), not a file path or another host.
- **"Disconnected" badge / rounds error out** — the backend isn't running or
  crashed; check the terminal running `python main.py`.
- **`mediapipe` install fails** — confirm your Python version is 3.9–3.12 and
  your `pip` is up to date (`python -m pip install --upgrade pip`).
- **Camera in use** — close other apps (Zoom, Teams, etc.) that hold the webcam.
