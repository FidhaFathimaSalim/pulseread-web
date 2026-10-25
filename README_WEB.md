# PulseRead Web — setup guide

A web frontend (landing page + dashboard) for your existing rPPG pipeline,
with **both** a file-upload flow and a **true live, frame-by-frame webcam**
flow. This wraps your existing Python code — it doesn't replace it.

```
pulseread-web/
  frontend/
    index.html          landing page
    dashboard.html       upload + live webcam + results dashboard
    css/styles.css
    js/dashboard.js       upload flow, charts, result rendering, PDF button
    js/webcam.js           getUserMedia + live socket streaming
  backend/
    app.py                Flask #1 — TensorFlow-side (file uploads)
    live_server.py          Flask-SocketIO #2 — JAX-side (live webcam)
    report_builder.py        PDF report generation (reportlab)
    requirements_web.txt
```

## Why two backend processes

Your project already solved this: **Keras 3 locks its backend (TensorFlow
vs JAX) per Python process.** `main.py` runs open-rppg as a subprocess for
exactly this reason. The web layer keeps the same split:

| Process | Backend | Handles |
|---|---|---|
| `app.py` (port 5000) | TensorFlow | Serves the site, file-upload analysis (imports your `main.py` unchanged, which still spawns open-rppg as a subprocess internally) |
| `live_server.py` (port 5001) | JAX | Live webcam — imports `openrppg_pipeline.py` directly, streams results over Socket.IO |

Run **both** processes at once, each in the venv that already has the
right backend installed (`venv_pulseread` and `venv_openrppg` from your
existing setup).

## 1. Copy your project files into `backend/`

`app.py` imports `main.py` as-is, and `live_server.py` imports
`openrppg_pipeline.py` as-is, so they need to sit alongside their
dependencies:

```
backend/
  app.py, live_server.py, report_builder.py     ← already here
  main.py, pulseread_pipeline.py, explain.py,
  recommendations.py, openrppg_pipeline.py       ← copy from your project
  face_landmarker.task
  models/pulseread_best.keras
```

Two things to check in your copied `main.py`:
- `OPENRPPG_PYTHON` is still hardcoded to your machine's venv path — leave
  it pointing at your `venv_openrppg\Scripts\python.exe`.
- Nothing else needs to change; `app.py` calls `main.analyze(video_path)`
  directly.

## 2. Install dependencies

```bash
# TensorFlow side (app.py) — your existing venv_pulseread
venv_pulseread\Scripts\activate
pip install -r requirements.txt            # your existing project deps
pip install -r backend\requirements_web.txt

# JAX side (live_server.py) — your existing venv_openrppg
venv_openrppg\Scripts\activate
pip install -r backend\requirements_web.txt
```

## 3. Run both servers

```bash
# Terminal 1
venv_pulseread\Scripts\activate
cd backend
python app.py           # -> http://localhost:5000

# Terminal 2
venv_openrppg\Scripts\activate
cd backend
python live_server.py   # -> http://localhost:5001
```

Then open **http://localhost:5000** — `app.py` serves the whole frontend,
so you don't need a separate static server. The dashboard's Upload tab
talks to port 5000; the Webcam tab talks to port 5001 directly.

If you ever serve the frontend from somewhere else, edit
`frontend/js/dashboard.js`:

```js
window.PULSEREAD_CONFIG = {
  API_BASE: 'http://localhost:5000',
  LIVE_BASE: 'http://localhost:5001',
};
```

## How live webcam mode actually works

`model.video_capture(0)` (used by your `run_webcam_live.py`) only works
for a camera physically attached to the machine running Python — it can't
be pointed at a browser tab's webcam. So for the web version:

1. The browser captures webcam frames via `getUserMedia` + a hidden
   canvas, and streams them as JPEGs over Socket.IO (~6 fps, downsampled)
   to `live_server.py`.
2. `live_server.py` keeps a rolling ~25s buffer of frames per connection.
3. A background loop re-runs `model.process_video_tensor(frames, fps=...)`
   over the current window every second — a quick BPM-only pass once ~5s
   is buffered, and a full HR+HRV pass every 10s once 20s is buffered.
   This mirrors the refresh cadence your own `run_webcam_live.py` already
   uses, just fed by the network instead of a local device.
4. Results stream back over the same socket as `live_update` events and
   populate the same Key Results / HRV / stress-gauge / insights UI the
   upload flow uses.

This is genuinely continuous/real-time (not record-a-clip-then-upload),
at the cost of recomputing over the sliding window each tick rather than
true incremental streaming inference — there's no documented public API
in `open-rppg` for injecting arbitrary external frames into a persistent
internal buffer, so this sliding-window approach is the closest
faithful equivalent using the library's supported tensor-input API.

**One thing to verify once you run it:** `process_video()` presumably
handles color-channel conversion internally when reading a video file.
Browser frames arrive here as OpenCV's default BGR. If live BPM readings
look clearly wrong while file uploads are fine, try swapping the channel
order in `live_server.py`'s `decode_frame()` (`cv2.cvtColor(frame,
cv2.COLOR_BGR2RGB)`) — I can't verify open-rppg's internal expectation
without running its source.

**Single-user note:** the live server shares one `rppg.Model()` instance
across all connections (same pattern as the rest of this project). Fine
for local/dissertation use; add a per-session model if you need several
people using live webcam mode at once.

## Notes on the PDF report

`report_builder.py` builds a summary PDF from whatever result JSON is
currently shown on the dashboard (upload or live) — heart rate, HRV,
stress state, and the same breathing/exercise/sleep/nutrition guidance
from your `recommendations.py`. It's independent of the "evidence log"
`main.py` already writes to `logs/comparison_log.jsonl`.
