/* =========================================================================
   Camera Rock-Paper-Scissors — frontend game logic
   -------------------------------------------------------------------------
   Responsibilities:
     * Acquire the webcam via getUserMedia and show the live feed.
     * Stream JPEG frames to the backend over a WebSocket and receive, per
       frame, the recognised gesture + hand landmarks.
     * Draw the hand skeleton overlay so the player sees they are tracked.
     * Run the "Rock… Paper… Scissors… Shoot!" countdown, capture the gesture
       at the shoot moment, resolve the round against the backend, and animate
       the reveal.
     * Maintain the session scoreboard (persisted in localStorage) and history.

   The backend does all detection/classification/round logic; this file owns
   only the camera, the game state machine, and the UI.
   ========================================================================= */

(() => {
  "use strict";

  // ------------------------------------------------------------------ config
  const WS_URL = `ws://${location.host}/ws`;
  const API_PLAY = "/api/play";
  const FRAME_INTERVAL_MS = 130;      // ~7-8 fps sent to backend
  const CAPTURE_W = 480;              // downscaled frame size for low latency
  const CAPTURE_H = 360;
  const JPEG_QUALITY = 0.6;
  const CHANT = ["Rock", "Paper", "Scissors", "Shoot!"];
  const CHANT_BEAT_MS = 650;
  const MIN_CONFIDENCE = 0.3;        // reject very ambiguous gestures at shoot
  const GESTURE_BUFFER = 6;          // frames to vote over at the shoot moment
  const STORAGE_KEY = "rps.session.v1";

  const EMOJI = { rock: "✊", paper: "✋", scissors: "✌️", none: "🖐️", unknown: "❓" };

  // -------------------------------------------------------------------- dom
  const $ = (id) => document.getElementById(id);
  const el = {
    video: $("video"),
    overlay: $("overlay"),
    liveBadge: $("liveBadge"),
    liveEmoji: $("liveEmoji"),
    liveLabel: $("liveLabel"),
    countdown: $("countdown"),
    countdownText: $("countdownText"),
    camError: $("camError"),
    camErrorText: $("camErrorText"),
    retryCamBtn: $("retryCamBtn"),
    statusMessage: $("statusMessage"),
    conn: $("connection"),
    connText: $("connText"),
    playBtn: $("playBtn"),
    resetBtn: $("resetBtn"),
    playerHand: $("playerHand"),
    computerHand: $("computerHand"),
    playerMove: $("playerMove"),
    computerMove: $("computerMove"),
    resultText: $("resultText"),
    scoreWins: $("scoreWins"),
    scoreLosses: $("scoreLosses"),
    scoreTies: $("scoreTies"),
    historyList: $("historyList"),
  };

  const octx = el.overlay.getContext("2d");
  // Offscreen canvas used to grab & downscale frames before sending.
  const grabCanvas = document.createElement("canvas");
  grabCanvas.width = CAPTURE_W;
  grabCanvas.height = CAPTURE_H;
  const grabCtx = grabCanvas.getContext("2d");

  // ------------------------------------------------------------------ state
  const state = {
    ws: null,
    wsReady: false,
    streaming: false,
    roundInProgress: false,
    latest: null,                 // most recent analysis message
    recentGestures: [],           // ring buffer for shoot-moment voting
    frameTimer: null,
    reconnectTimer: null,
    score: { wins: 0, losses: 0, ties: 0 },
    history: [],
    connections: null,            // landmark connection pairs from backend
  };

  // ============================================================= persistence
  function loadSession() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const data = JSON.parse(raw);
      if (data.score) state.score = data.score;
      if (Array.isArray(data.history)) state.history = data.history;
    } catch (err) {
      console.warn("Could not load saved session", err);
    }
  }

  function saveSession() {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ score: state.score, history: state.history })
      );
    } catch (err) {
      console.warn("Could not save session", err);
    }
  }

  // ================================================================== status
  function setStatus(text, kind = "") {
    el.statusMessage.textContent = text;
    el.statusMessage.parentElement.className =
      "status-bar" + (kind ? " is-" + kind : "");
  }

  function setConnection(cls, text) {
    el.conn.className = "conn conn--" + cls;
    el.connText.textContent = text;
  }

  // ================================================================== camera
  async function initCamera() {
    el.camError.hidden = true;
    setStatus("Requesting camera access…");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      el.video.srcObject = stream;
      await el.video.play();
      // Size the overlay canvas to the video's intrinsic resolution.
      el.overlay.width = el.video.videoWidth || 1280;
      el.overlay.height = el.video.videoHeight || 720;
      setStatus("Camera ready. Press “Start Round” when you’re set.", "ok");
      el.playBtn.disabled = !state.wsReady;
      startFrameLoop();
    } catch (err) {
      console.error("getUserMedia failed", err);
      showCameraError(err);
    }
  }

  function showCameraError(err) {
    el.camError.hidden = false;
    el.retryCamBtn.hidden = false;
    let msg = "Could not access the camera.";
    if (err && (err.name === "NotAllowedError" || err.name === "SecurityError")) {
      msg = "Camera permission was denied. Allow camera access in your browser, then try again.";
    } else if (err && err.name === "NotFoundError") {
      msg = "No camera was found. Connect a webcam and try again.";
    } else if (err && err.name === "NotReadableError") {
      msg = "The camera is in use by another application. Close it and try again.";
    }
    el.camErrorText.textContent = msg;
    setStatus(msg, "error");
    el.playBtn.disabled = true;
  }

  // ============================================================ frame stream
  function startFrameLoop() {
    if (state.frameTimer) return;
    state.frameTimer = setInterval(sendFrame, FRAME_INTERVAL_MS);
  }

  function sendFrame() {
    if (!state.wsReady || el.video.readyState < 2) return;
    try {
      // Draw the current video frame downscaled, then encode as JPEG.
      grabCtx.drawImage(el.video, 0, 0, CAPTURE_W, CAPTURE_H);
      const dataUrl = grabCanvas.toDataURL("image/jpeg", JPEG_QUALITY);
      state.ws.send(JSON.stringify({ type: "frame", data: dataUrl }));
    } catch (err) {
      // A transient encode/send error should not kill the loop.
      console.debug("frame send skipped", err);
    }
  }

  // ============================================================== websocket
  function connectWs() {
    setConnection("connecting", "Connecting…");
    let ws;
    try {
      ws = new WebSocket(WS_URL);
    } catch (err) {
      console.error("WebSocket construction failed", err);
      scheduleReconnect();
      return;
    }
    state.ws = ws;

    ws.onopen = () => {
      state.wsReady = true;
      setConnection("open", "Connected");
      if (el.video.srcObject) el.playBtn.disabled = false;
    };

    ws.onmessage = (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      handleServerMessage(msg);
    };

    ws.onerror = (event) => {
      console.warn("WebSocket error", event);
    };

    ws.onclose = () => {
      state.wsReady = false;
      setConnection("closed", "Disconnected");
      el.playBtn.disabled = true;
      scheduleReconnect();
    };
  }

  function scheduleReconnect() {
    if (state.reconnectTimer) return;
    state.reconnectTimer = setTimeout(() => {
      state.reconnectTimer = null;
      connectWs();
    }, 1500);
  }

  function handleServerMessage(msg) {
    if (msg.type === "error") {
      console.warn("server error:", msg.message);
      return;
    }
    if (msg.type !== "analysis") return;

    state.latest = msg;
    if (msg.connections) state.connections = msg.connections;

    // Maintain a short voting buffer of recent gestures for the shoot moment.
    state.recentGestures.push(msg.gesture);
    if (state.recentGestures.length > GESTURE_BUFFER) {
      state.recentGestures.shift();
    }

    drawOverlay(msg);
    updateLiveBadge(msg);

    // Surface hints (multiple hands, unclear gesture) unless mid-round.
    if (!state.roundInProgress) {
      if (msg.gesture === "none") {
        setStatus("No hand detected — show your hand clearly.");
      } else if (msg.message) {
        setStatus(msg.message, "warn");
      } else {
        setStatus("Hand tracked. Press “Start Round” to play.", "ok");
      }
    }
  }

  // ============================================================ overlay draw
  function drawOverlay(msg) {
    octx.clearRect(0, 0, el.overlay.width, el.overlay.height);
    const lms = msg.landmarks;
    if (!lms || lms.length === 0) return;

    const W = el.overlay.width;
    const H = el.overlay.height;
    // Landmarks are normalised; the canvas is CSS-mirrored together with the
    // video, so drawing with raw x keeps the skeleton aligned with the feed.
    const px = (p) => [p[0] * W, p[1] * H];

    // Bones.
    octx.lineWidth = 3;
    octx.strokeStyle = "rgba(91, 140, 255, 0.9)";
    const conns = state.connections || [];
    for (const [a, b] of conns) {
      if (a >= lms.length || b >= lms.length) continue;
      const [x1, y1] = px(lms[a]);
      const [x2, y2] = px(lms[b]);
      octx.beginPath();
      octx.moveTo(x1, y1);
      octx.lineTo(x2, y2);
      octx.stroke();
    }

    // Joints.
    for (const p of lms) {
      const [x, y] = px(p);
      octx.beginPath();
      octx.arc(x, y, 4.5, 0, Math.PI * 2);
      octx.fillStyle = "#7c5cff";
      octx.fill();
      octx.lineWidth = 1.5;
      octx.strokeStyle = "rgba(255,255,255,0.85)";
      octx.stroke();
    }
  }

  function updateLiveBadge(msg) {
    const g = msg.gesture;
    el.liveEmoji.textContent = EMOJI[g] || "🖐️";
    if (g === "none") {
      el.liveLabel.textContent = "No hand";
      el.liveBadge.className = "live-badge is-none";
    } else if (g === "unknown") {
      el.liveLabel.textContent = "Unclear";
      el.liveBadge.className = "live-badge is-detected";
    } else {
      const pct = Math.round((msg.confidence || 0) * 100);
      el.liveLabel.textContent = `${g} · ${pct}%`;
      el.liveBadge.className = "live-badge is-detected";
    }
  }

  // ============================================================== game round
  async function playRound() {
    if (state.roundInProgress || !state.wsReady) return;
    state.roundInProgress = true;
    el.playBtn.disabled = true;
    el.resetBtn.disabled = true;

    // Reset the reveal area.
    el.resultText.textContent = "";
    el.resultText.className = "reveal__result";
    el.playerMove.textContent = "—";
    el.computerMove.textContent = "—";
    el.playerHand.textContent = "🖐️";
    el.computerHand.textContent = "🤖";
    state.recentGestures = [];

    await runCountdown();

    // Vote over the recent gesture buffer for a stable read at "Shoot!".
    const move = pickPlayerMove();

    hideCountdown();

    if (!move) {
      setStatus(
        "No clear gesture at “Shoot!” — that round didn’t count. Try again.",
        "warn"
      );
      el.resultText.textContent = "Retry";
      el.playerHand.textContent = "❓";
      endRound();
      return;
    }

    el.playerMove.textContent = move;
    el.playerHand.textContent = EMOJI[move];

    try {
      const res = await fetch(API_PLAY, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ player_move: move }),
      });
      if (!res.ok) throw new Error(`server ${res.status}`);
      const data = await res.json();
      revealResult(move, data.computer_move, data.result);
    } catch (err) {
      console.error("round resolution failed", err);
      setStatus("Could not reach the game server. Check the backend and retry.", "error");
      el.resultText.textContent = "Error";
    }
    endRound();
  }

  function endRound() {
    state.roundInProgress = false;
    el.playBtn.disabled = !state.wsReady;
    el.resetBtn.disabled = false;
  }

  function runCountdown() {
    return new Promise((resolve) => {
      el.countdown.hidden = false;
      let i = 0;
      const tick = () => {
        const word = CHANT[i];
        el.countdownText.textContent = word;
        el.countdownText.className =
          "countdown__text" + (word === "Shoot!" ? " shoot" : "");
        // restart the pop animation
        el.countdownText.style.animation = "none";
        void el.countdownText.offsetWidth;
        el.countdownText.style.animation = "";
        i += 1;
        if (i < CHANT.length) {
          setTimeout(tick, CHANT_BEAT_MS);
        } else {
          // Hold "Shoot!" briefly, then resolve so the last frames are captured.
          setTimeout(resolve, 250);
        }
      };
      tick();
    });
  }

  function hideCountdown() {
    el.countdown.hidden = true;
  }

  // Majority vote over recent frames; ignores none/unknown and requires a
  // minimum confidence on the latest read of the chosen gesture.
  function pickPlayerMove() {
    const counts = { rock: 0, paper: 0, scissors: 0 };
    for (const g of state.recentGestures) {
      if (g in counts) counts[g] += 1;
    }
    let best = null;
    let bestCount = 0;
    for (const g of Object.keys(counts)) {
      if (counts[g] > bestCount) {
        best = g;
        bestCount = counts[g];
      }
    }
    if (!best || bestCount === 0) return null;
    // Guard against a fluke single frame with very low confidence.
    if (
      state.latest &&
      state.latest.gesture === best &&
      (state.latest.confidence || 0) < MIN_CONFIDENCE &&
      bestCount < 2
    ) {
      return null;
    }
    return best;
  }

  function revealResult(playerMove, computerMove, result) {
    el.computerMove.textContent = computerMove;
    el.computerHand.textContent = EMOJI[computerMove];

    // Little "shake" like a real hand throw.
    el.playerHand.classList.remove("shake");
    el.computerHand.classList.remove("shake");
    void el.playerHand.offsetWidth;
    el.playerHand.classList.add("shake");
    el.computerHand.classList.add("shake");

    const label = { win: "You win! 🎉", lose: "You lose 😑", tie: "Tie 🤝" }[result];
    el.resultText.textContent = label;
    el.resultText.className = "reveal__result " + result;

    // Update score.
    if (result === "win") state.score.wins += 1;
    else if (result === "lose") state.score.losses += 1;
    else state.score.ties += 1;

    // Prepend to history (cap at 20 entries).
    state.history.unshift({ playerMove, computerMove, result, t: Date.now() });
    state.history = state.history.slice(0, 20);

    renderScore();
    renderHistory();
    saveSession();

    setStatus(
      `You threw ${playerMove}, computer threw ${computerMove}.`,
      result === "win" ? "ok" : result === "lose" ? "error" : "warn"
    );
  }

  // ============================================================== rendering
  function renderScore() {
    el.scoreWins.textContent = state.score.wins;
    el.scoreLosses.textContent = state.score.losses;
    el.scoreTies.textContent = state.score.ties;
  }

  function renderHistory() {
    if (state.history.length === 0) {
      el.historyList.innerHTML =
        '<li class="history__empty">No rounds played yet.</li>';
      return;
    }
    el.historyList.innerHTML = state.history
      .map((h) => {
        const badge = { win: "Win", lose: "Loss", tie: "Tie" }[h.result];
        return `<li>
          <span title="you">${EMOJI[h.playerMove]}</span>
          <span class="history__vs">vs</span>
          <span title="computer">${EMOJI[h.computerMove]}</span>
          <span class="history__badge ${h.result}">${badge}</span>
        </li>`;
      })
      .join("");
  }

  function resetScore() {
    state.score = { wins: 0, losses: 0, ties: 0 };
    state.history = [];
    renderScore();
    renderHistory();
    saveSession();
    setStatus("Score reset. New game — good luck!", "ok");
  }

  // =================================================================== init
  function init() {
    loadSession();
    renderScore();
    renderHistory();

    el.playBtn.addEventListener("click", playRound);
    el.resetBtn.addEventListener("click", resetScore);
    el.retryCamBtn.addEventListener("click", initCamera);

    connectWs();
    initCamera();
  }

  // Kick off once the DOM is ready.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
