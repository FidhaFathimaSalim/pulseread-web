/* ==========================================================================
   PulseRead — live webcam capture
   Streams frames (base64 JPEG, downsampled) to live_server.py over
   Socket.IO. The server keeps a rolling buffer per connection and
   periodically runs the open-rppg model over the current window,
   emitting 'live_update' events with the latest HR/HRV/stress estimate.

   FIXES IN THIS VERSION:
     * Higher, steadier frame rate (rPPG needs ~20+ fps, 6 was too low).
       Uses requestVideoFrameCallback when available (in sync with the
       camera), otherwise falls back to setInterval.
     * Smaller frames (320 px wide) so 20 s of buffered frames stays light.
     * Clear on-screen errors if the Socket.IO script is missing.
     * No duplicate capture loops on reconnect / double-click.
     * BPM chip no longer flickers to "--" when a model run has no result.
   ========================================================================== */

window.PULSEREAD_WEBCAM_VERSION = 2;   // features.js checks this to detect a stale webcam.js
let webcamStream = null;
let webcamSocket = null;
let captureInterval = null;
let recTimerInterval = null;
let recSeconds = 0;
let capturing = false;
let lastSentAt = 0;
let lastDrawnRun = null;

// ~20 fps is a good balance for the model without overloading the server.
// Raise to 30 if your machine keeps up (check "eff_fps" in the server log).
const CAPTURE_FPS = 20;
const JPEG_QUALITY = 0.95;  // high quality: compression noise can drown the tiny pulse signal
const FRAME_MAX_WIDTH = 320;

// Below this signal quality (%) we show "--" instead of a BPM, because a reading
// from a very weak signal is mostly noise. Set to 0 to always show the number.
const MIN_QUALITY_PCT = 40;   // log evidence: 20-30 s windows above ~40 % cluster at the true value, below ~20 % are noise
window.PULSEREAD_MIN_QUALITY = MIN_QUALITY_PCT;   // graphs use the same cut-off

// ---- AUTO-STOP (single + multi-person) -------------------------------------------------------
// Recording stops by itself once a reading has signal quality ABOVE this value AND every value
// (heart rate, HRV, stress, breathing) is there. Not while a Physiological Challenge is pending/running.
const AUTO_STOP_QUALITY = 50;
// breathing needs a 30 s window and is not always trusted; after this many seconds of recording we stop
// without it rather than recording forever
const BREATH_GIVEUP_SEC = 90;     // the 30 s breathing run normally finishes around 35-45 s
window.PULSEREAD_AUTOSTOP_QUALITY = AUTO_STOP_QUALITY;
window.PULSEREAD_REC_SECONDS = 0;

// d = a live result (single payload, or one person's person_update)
window.PULSEREAD_IS_COMPLETE = function (d) {
  if (!d || d.no_face) return false;
  const q = d.signal_quality_pct;
  if (d.heart_rate_bpm == null || q == null || !(q > AUTO_STOP_QUALITY)) return false;
  if (!d.hrv_available || d.hrv_rmssd_ms == null || d.hrv_sdnn_ms == null) return false;
  if (!d.stress_level) return false;
  // Breathing counts as captured only when the real 30 s breathing run has finished AND its waveform exists
  // (the Breathing graph needs it). A bare breathing number from the shorter window is not enough.
  const hasBreathingGraph = !!(d.resp && d.resp.wave && d.resp.wave.t && d.resp.wave.t.length);
  if (!hasBreathingGraph && (window.PULSEREAD_REC_SECONDS || 0) < BREATH_GIVEUP_SEC) return false;
  return true;
};

// Stops the camera + socket but leaves every result on screen. Returns true if it stopped.
window.PULSEREAD_AUTOSTOP = function (message, lastBpm) {
  if (window.PULSEREAD_CHALLENGE_BUSY) return false;
  if (!webcamStream && !webcamSocket) return false;
  stopWebcamCapture();
  connStatus.textContent = message || 'Done · all values captured';
  if (lastBpm != null) liveBpmChip.textContent = `${lastBpm}`;
  return true;
};

const startBtn = document.getElementById('startWebcamBtn');
const stopBtn = document.getElementById('stopWebcamBtn');
const webcamVideo = document.getElementById('webcamVideo');
const webcamStandby = document.getElementById('webcamStandby');
const webcamDot = document.getElementById('webcamDot');
const recBadge = document.getElementById('recBadge');
const recTimer = document.getElementById('recTimer');
const connStatus = document.getElementById('connStatus');
const liveBpmChip = document.getElementById('liveBpmChip');
const bufferChip = document.getElementById('bufferChip');

const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

startBtn.addEventListener('click', startWebcamCapture);
stopBtn.addEventListener('click', stopWebcamCapture);

async function startWebcamCapture() {
  if (webcamStream || webcamSocket) return; // already running

  if (typeof io === 'undefined') {
    connStatus.textContent = 'Socket.IO script missing';
    alert('The Socket.IO client script did not load. Check the <script> tag for socket.io in dashboard.html.');
    return;
  }

  try {
    // multi-person: ask for a sharper picture, so 3-4 faces each keep enough pixels (single mode is unchanged: 640x480)
    const multi = (window.PULSEREAD_MODE || 'single') === 'multi';
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: multi ? { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } }
                   : { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      audio: false,
    });
  } catch (err) {
    webcamStream = null;
    alert(`Could not access the webcam: ${err.message}`);
    return;
  }

  webcamVideo.srcObject = webcamStream;
  try { await webcamVideo.play(); } catch (e) { /* autoplay attribute handles it */ }

  webcamStandby.style.display = 'none';
  webcamVideo.style.display = 'block';
  recBadge.style.display = 'flex';
  webcamDot.classList.remove('dot-idle');
  webcamDot.classList.add('dot-live');
  startBtn.classList.add('hidden');
  stopBtn.classList.remove('hidden');

  recSeconds = 0;
  window.PULSEREAD_REC_SECONDS = 0;
  lastDrawnRun = null;
  recTimer.textContent = '00:00';
  recTimerInterval = setInterval(() => {
    recSeconds += 1;
    window.PULSEREAD_REC_SECONDS = recSeconds;
    const m = String(Math.floor(recSeconds / 60)).padStart(2, '0');
    const s = String(recSeconds % 60).padStart(2, '0');
    recTimer.textContent = `${m}:${s}`;
  }, 1000);

  if (window.PULSEREAD_LIVE_BEGIN) window.PULSEREAD_LIVE_BEGIN();   // clears Key Results + graphs for this session
  connectLiveSocket();
}

function connectLiveSocket() {
  connStatus.textContent = 'Connecting…';
  webcamSocket = io(window.PULSEREAD_CONFIG.LIVE_BASE, { transports: ['websocket', 'polling'] });
  // v2: let challenge.js / people.js attach their own listeners to this socket
  (window.PULSEREAD_SOCKET_HOOKS || []).forEach(fn => { try { fn(webcamSocket); } catch (e) { console.error(e); } });

  webcamSocket.on('connect', () => {
    connStatus.textContent = 'Live';
    webcamSocket.emit('config', { mode: window.PULSEREAD_MODE || 'single' });   // v2: single / multi-person
    beginFrameCapture();
  });

  webcamSocket.on('disconnect', () => {
    connStatus.textContent = 'Disconnected';
    capturing = false;
  });

  webcamSocket.on('connect_error', (err) => {
    connStatus.textContent = 'Connection failed';
    console.error('live_server connection error — is live_server.py running on',
                  window.PULSEREAD_CONFIG.LIVE_BASE, err);
  });

  // Show empty tiles (with a reason) instead of a misleading number.
  function renderEmpty(note, quality, extra) {
    if (!window.PULSEREAD_RENDER) return;
    window.PULSEREAD_RENDER({
      status: 'ok',
      heart_rate_bpm: null,
      hrv_available: false,
      hrv_note: note,
      signal_quality_pct: quality,
      stress_level: null,
      recommendation: null,
      waveform_plot_url: null,
      ...(extra || {}),
    }, { live: true });
  }

  webcamSocket.on('live_update', (payload) => {
    // multi-person: ignore updates that belong to a person who is not the one shown on the dashboard
    if (window.PULSEREAD_ACTIVE_PERSON != null && payload.person_id != null &&
        payload.person_id !== window.PULSEREAD_ACTIVE_PERSON) return;
    if (window.PULSEREAD_LIVE_UPDATE) window.PULSEREAD_LIVE_UPDATE(payload);   // feeds the three graphs
    const buf = payload.buffer_sec != null ? payload.buffer_sec.toFixed(0) : 0;
    bufferChip.textContent = `${buf}s / ${payload.hrv_min_buffer_sec || 20}s`;

    const noFace = payload.no_face === true;
    const hasBpm = payload.heart_rate_bpm != null;
    const q = payload.signal_quality_pct;
    const tooWeak = hasBpm && q != null && q < MIN_QUALITY_PCT;

    if (noFace) {
      liveBpmChip.textContent = '--';
      connStatus.textContent = payload.multi_face ? 'More than one face: one person only' : 'No face detected';
      renderEmpty(payload.hrv_note, null);
      return;
    }
    if (tooWeak) {
      liveBpmChip.textContent = '--';
      connStatus.textContent = 'Signal too weak';
      renderEmpty(`Signal too weak (${q.toFixed(0)}%) — face the light and hold still.`, q);
      return;
    }

    liveBpmChip.textContent = hasBpm ? `${payload.heart_rate_bpm}` : '--';
    connStatus.textContent = hasBpm ? 'Live' : 'Live · warming up';

    if (hasBpm && window.PULSEREAD_RENDER) {
      window.PULSEREAD_RENDER({
        status: 'ok',
        heart_rate_bpm: payload.heart_rate_bpm,
        bpm_reliability: payload.bpm_reliability,
        hrv_rmssd_ms: payload.hrv_rmssd_ms,
        hrv_sdnn_ms: payload.hrv_sdnn_ms,
        mean_ibi_ms: payload.mean_ibi_ms,
        hrv_available: payload.hrv_available,
        hrv_note: payload.hrv_note,
        signal_quality_pct: payload.signal_quality_pct,
        stress_level: payload.stress_level,
        breathing_rate_per_min: payload.breathing_rate_per_min,
        breathing_source: payload.breathing_source,
        resp: payload.resp,
        recommendation: payload.recommendation,
        waveform_plot_url: null, // no static waveform image in live mode
      });
    }
    if (payload.hrv_note) console.debug('[live]', payload.hrv_note);

    // The graphs are drawn BEFORE the tiles are updated (dashboard.js gets the update first), so when a new model run
    // arrives the Breathing / rPPG graph would still show the previous run. Redraw once per new run, after the render above.
    if (hasBpm && payload.run_id != null && payload.run_id !== lastDrawnRun) {
      lastDrawnRun = payload.run_id;
      if (window.renderSignalGraph) { try { window.renderSignalGraph(); } catch (e) { console.error(e); } }
    }

    // single-person: stop recording once everything is captured at > 50 % quality
    // (multi-person does the same per person in people.js)
    if ((window.PULSEREAD_MODE || 'single') === 'single' && window.PULSEREAD_IS_COMPLETE(payload)) {
      window.PULSEREAD_AUTOSTOP(`Done · all values captured (quality ${q.toFixed(0)}%)`, payload.heart_rate_bpm);
    }
  });

  webcamSocket.on('live_error', (payload) => {
    connStatus.textContent = (payload && payload.message) || 'Error';
  });
}

function sendFrame() {
  if (!webcamSocket || !webcamSocket.connected) return;
  if (webcamVideo.readyState < 2 || !webcamVideo.videoWidth) return;
  // timestamp NOW, before the (slow, uneven) JPEG encoding: the pulse is sampled at the moment the frame was captured,
  // not at the moment it finished encoding. Encode time varies by 10-40 ms and that jitter is noise in the pulse signal.
  const capturedAt = Date.now() / 1000;

  // v2: multi-person sends wider frames (never wider than the camera really gives: upscaling adds no detail)
  const frameW = Math.min(window.PULSEREAD_FRAME_WIDTH || FRAME_MAX_WIDTH, webcamVideo.videoWidth);
  const targetH = Math.round(frameW * (webcamVideo.videoHeight / webcamVideo.videoWidth));
  if (captureCanvas.width !== frameW || captureCanvas.height !== targetH) {
    captureCanvas.width = frameW;
    captureCanvas.height = targetH;
  }

  captureCtx.drawImage(webcamVideo, 0, 0, captureCanvas.width, captureCanvas.height);
  // multi-person: 4 faces at 960 px already carry the pulse; q 0.95 only makes the frames ~2x bigger and slower to encode/send
  const multiMode = (window.PULSEREAD_MODE || 'single') === 'multi';
  const dataUrl = captureCanvas.toDataURL('image/jpeg', (multiMode && captureCanvas.width > 700) ? 0.92 : JPEG_QUALITY);
  // volatile: if the connection is busy, drop this frame instead of queueing it
  webcamSocket.volatile.emit('frame', { image: dataUrl, ts: capturedAt });
}

function beginFrameCapture() {
  if (capturing) return; // avoid duplicate loops when the socket reconnects
  capturing = true;
  lastSentAt = 0;
  const minGap = 1000 / CAPTURE_FPS;

  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    const onFrame = (now) => {
      if (!capturing) return;
      if (now - lastSentAt >= minGap - 3) {
        lastSentAt = now;
        sendFrame();
      }
      webcamVideo.requestVideoFrameCallback(onFrame);
    };
    webcamVideo.requestVideoFrameCallback(onFrame);
  } else {
    captureInterval = setInterval(sendFrame, minGap);
  }
}

function stopWebcamCapture() {
  capturing = false;
  if (window.PULSEREAD_LIVE_STOP) window.PULSEREAD_LIVE_STOP();
  if (captureInterval) { clearInterval(captureInterval); captureInterval = null; }
  if (recTimerInterval) { clearInterval(recTimerInterval); recTimerInterval = null; }
  if (webcamSocket) { webcamSocket.disconnect(); webcamSocket = null; }
  if (webcamStream) { webcamStream.getTracks().forEach(t => t.stop()); webcamStream = null; }
  webcamVideo.srcObject = null;

  webcamVideo.style.display = 'none';
  webcamStandby.style.display = '';   // back to whatever the CSS defines
  recBadge.style.display = 'none';
  webcamDot.classList.remove('dot-live');
  webcamDot.classList.add('dot-idle');
  startBtn.classList.remove('hidden');
  stopBtn.classList.add('hidden');
  connStatus.textContent = 'Not connected';
  liveBpmChip.textContent = '--';
  bufferChip.textContent = '0s / 20s';
}

// Called by dashboard.js when the user switches back to the Upload tab
window.stopWebcamCapture = stopWebcamCapture;

// When the mode changes, the camera chips must not keep the last mode's numbers / "Done" message.
(window.PULSEREAD_CLEAR_HOOKS = window.PULSEREAD_CLEAR_HOOKS || []).push(() => {
  if (webcamStream || webcamSocket) return;       // a camera is running: leave it alone
  connStatus.textContent = 'Not connected';
  liveBpmChip.textContent = '--';
  bufferChip.textContent = '0s / 20s';
});
