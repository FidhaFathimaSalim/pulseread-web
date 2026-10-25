window.PULSEREAD_JS_VERSION = 2;   // features.js checks this to detect a stale cached/old dashboard.js
/* ==========================================================================
   PulseRead dashboard — shared config + upload flow + result rendering
   ========================================================================== */

// ---- Backend endpoints. Adjust if you deploy the two Flask processes
// on different hosts/ports than the defaults in backend/app.py / live_server.py.
window.PULSEREAD_CONFIG = {
  API_BASE: 'http://localhost:5000',   // app.py — file-upload analysis (TF + subprocess open-rppg)
  LIVE_BASE: 'http://localhost:5001',  // live_server.py — live webcam analysis (JAX, socket.io)
};

let lastResult = null;
let signalChart = null;
let hrvChart = null;
let sessionHistory = [];     // past sessions from /api/history (for the HRV Analysis card)
let currentGraph = null;     // graph data of the latest analysed recording (data.graph)

// ---- Live (webcam) state ----
let liveActive = false;      // true while a webcam session is streaming
let liveSeries = [];         // one point per finished model run: { t, bpm, rmssd, sdnn }
let liveGraph = null;        // graph data of the latest 20 s window
let liveLastRunId = null;
let liveStartMs = null;

// ---- Separate results for the Upload tab and the Webcam tab ----
let currentMode = 'upload';                 // which tab is showing
const modeStore = { upload: null, webcam: null };   // what each tab last showed
const DEFAULT_INSIGHTS_HTML = document.getElementById('insightsList').innerHTML;
const DEFAULT_INSIGHTS_SUMMARY = document.getElementById('insightsSummary').textContent;
const EMPTY_RESULT = {
  heart_rate_bpm: null, bpm_reliability: null, hrv_rmssd_ms: null, hrv_sdnn_ms: null,
  mean_ibi_ms: null, hrv_available: false, hrv_note: 'Awaiting data',
  signal_quality_pct: null, stress_level: null, recommendation: null,
};

// -------------------- Tab switching (Upload vs Webcam) --------------------
const tabUpload = document.getElementById('tabUpload');
const tabWebcam = document.getElementById('tabWebcam');
const panelUpload = document.getElementById('panelUpload');
const panelWebcam = document.getElementById('panelWebcam');

tabUpload.addEventListener('click', () => switchMode('upload'));
tabWebcam.addEventListener('click', () => switchMode('webcam'));

// Remember what the tab we are leaving was showing, then show the other tab's own
// results (or an empty dashboard if it has none yet). Nothing is shared between tabs.
function captureMode(mode) {
  modeStore[mode] = { result: lastResult, currentGraph, liveGraph, liveSeries: liveSeries.slice(), liveLastRunId };
}

function restoreMode(mode) {
  const s = modeStore[mode];
  lastResult = s ? s.result : null;
  currentGraph = s ? s.currentGraph : null;
  liveGraph = s ? s.liveGraph : null;
  liveSeries = s ? s.liveSeries.slice() : [];
  liveLastRunId = s ? s.liveLastRunId : null;
  graphMode = mode === 'webcam' ? 'live' : 'recording';

  paintTiles(lastResult || EMPTY_RESULT);
  setBadges(mode === 'upload' && !!lastResult);
  document.getElementById('downloadReportBtn').disabled = !(lastResult && lastResult.heart_rate_bpm != null);
  renderSignalGraph();
  renderHrvCard();
}

function switchMode(mode) {
  if (mode === currentMode) return;
  if (currentMode === 'webcam' && window.stopWebcamCapture) window.stopWebcamCapture();
  currentMode = mode;
  tabUpload.classList.toggle('active', mode === 'upload');
  tabWebcam.classList.toggle('active', mode === 'webcam');
  panelUpload.classList.toggle('hidden', mode !== 'upload');
  panelWebcam.classList.toggle('hidden', mode !== 'webcam');
  graphMode = mode === 'webcam' ? 'live' : 'recording';
  clearAllOutputs();          // every mode starts EMPTY: nothing is carried over from the mode we left
}

// Erases every result on the page and puts the Upload box back to its first-load state.
// Called when switching Upload <-> One-person <-> Multi-person (people.js calls it for One <-> Multi).
// Scripts that own other output (challenge.js, webcam.js ...) add a function to PULSEREAD_CLEAR_HOOKS.
window.PULSEREAD_CLEAR_HOOKS = window.PULSEREAD_CLEAR_HOOKS || [];
let modeEpoch = 0;            // bumped on every clear: an upload analysis that finishes later is thrown away
function clearAllOutputs() {
  modeEpoch += 1;
  modeStore.upload = null;
  modeStore.webcam = null;
  lastResult = null;
  currentGraph = null;
  liveGraph = null;
  liveSeries = [];
  liveLastRunId = null;
  liveActive = false;

  paintTiles(EMPTY_RESULT);
  setBadges(false);
  document.getElementById('downloadReportBtn').disabled = true;
  renderSignalGraph();
  renderHrvCard();

  // Upload box back to "Upload Your Facial Video"
  selectedFile = null;
  previewVideo.removeAttribute('src');
  try { previewVideo.load(); } catch (e) { /* ignore */ }
  fileInput.value = '';
  uploadBox.classList.remove('has-preview');
  analyzeBtn.classList.add('hidden');
  uploadStatus.textContent = '';

  window.PULSEREAD_CLEAR_HOOKS.forEach(fn => { try { fn(); } catch (e) { console.error(e); } });
}
window.PULSEREAD_CLEAR_OUTPUT = clearAllOutputs;

function setBadges(show) {
  ['badgeFace', 'badgeRoi'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('hidden', !show);
  });
}

// -------------------- Upload flow --------------------
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const previewPanel = document.getElementById('previewPanel');
const previewVideo = document.getElementById('previewVideo');
const changeFileBtn = document.getElementById('changeFileBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const uploadStatus = document.getElementById('uploadStatus');

browseBtn.addEventListener('click', () => fileInput.click());

uploadBox.addEventListener('dragover', (e) => { e.preventDefault(); uploadBox.classList.add('dragover'); });
uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('dragover'));
uploadBox.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadBox.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFileSelected(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (e) => {
  if (e.target.files.length) handleFileSelected(e.target.files[0]);
});

let selectedFile = null;

// Removes the previous upload's numbers + graphs (used when the file is changed or replaced).
function clearUploadResults() {
  modeEpoch += 1;                    // an analysis still running for the OLD file is thrown away
  lastResult = null;
  currentGraph = null;
  paintTiles(EMPTY_RESULT);
  setBadges(false);
  document.getElementById('downloadReportBtn').disabled = true;
  renderSignalGraph();
  renderHrvCard();
}

function handleFileSelected(file) {
  clearUploadResults();              // a new file must not show the old file's values
  selectedFile = file;
  previewVideo.src = URL.createObjectURL(file);
  uploadBox.classList.add('has-preview');
  analyzeBtn.classList.remove('hidden');
  uploadStatus.textContent = `Selected: ${file.name} (${(file.size / 1e6).toFixed(1)} MB)`;
  // Reset detection badges until next successful analysis
  const bF = document.getElementById('badgeFace');
  const bR = document.getElementById('badgeRoi');
  if (bF) bF.classList.add('hidden');
  if (bR) bR.classList.add('hidden');
}

// "Change file" button — revert box back to upload prompt
changeFileBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  clearUploadResults();
  selectedFile = null;
  previewVideo.src = '';
  fileInput.value = '';
  uploadBox.classList.remove('has-preview');
  analyzeBtn.classList.add('hidden');
  uploadStatus.textContent = '';
  setBadges(false);
});

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing… this can take up to a minute';
  uploadStatus.textContent = 'Uploading and running the rPPG pipeline (both models)…';

  const form = new FormData();
  form.append('video', selectedFile);
  const myEpoch = modeEpoch;     // if the user changes mode while this runs, the result is discarded

  try {
    const res = await fetch(`${window.PULSEREAD_CONFIG.API_BASE}/api/analyze`, {
      method: 'POST',
      body: form,
    });
    const data = await res.json();
    if (myEpoch !== modeEpoch) return;     // user left the Upload tab meanwhile: do not show this result anywhere

    if (!res.ok || data.status !== 'ok') {
      uploadStatus.textContent = `Error: ${data.error_message || 'analysis failed'}`;
      return;
    }
    uploadStatus.textContent = 'Analysis complete.';
    renderResult(data);
  } catch (err) {
    if (myEpoch !== modeEpoch) return;
    uploadStatus.textContent = `Could not reach the backend at ${window.PULSEREAD_CONFIG.API_BASE}. Is app.py running? (${err.message})`;
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze Video';
  }
});

// -------------------- Result rendering (shared by upload + live webcam) --------------------

function classifyBpmTag(bpm) {
  if (bpm == null) return { text: 'Awaiting data', cls: '' };
  if (bpm < 60) return { text: 'Low', cls: '' };
  if (bpm <= 100) return { text: 'Normal', cls: '' };
  return { text: 'Elevated', cls: '' };
}

function renderResult(data, opts) {
  lastResult = data;
  paintTiles(data);
  renderModeExtras(data);
}

// Fills Key Results / HRV numbers / stress gauge / insights from one result object.
function paintTiles(data) {
  // Heart rate
  document.getElementById('valBpm').textContent = data.heart_rate_bpm != null ? data.heart_rate_bpm : '--';
  const bpmTag = classifyBpmTag(data.heart_rate_bpm);
  document.getElementById('subBpm').textContent = data.bpm_reliability
    ? `${bpmTag.text} · ${data.bpm_reliability} confidence`
    : bpmTag.text;

  // HRV
  document.getElementById('valHrv').textContent = data.hrv_rmssd_ms != null ? data.hrv_rmssd_ms : '--';
  document.getElementById('subHrv').textContent = data.hrv_available
    ? (data.hrv_confidence === 'Low' ? 'Rough estimate · signal quality is only fair' : 'Good')
    : (data.hrv_note || 'Unavailable');

  // Stress
  document.getElementById('valStress').textContent = data.stress_level || '--';
  document.getElementById('subStress').textContent = stressDescriptor(data.stress_level) +
    (data.hrv_available && data.hrv_confidence === 'Low' ? ' · rough estimate' : '');

  // Signal quality
  document.getElementById('valQuality').textContent = data.signal_quality_pct != null ? data.signal_quality_pct : '--';
  document.getElementById('subQuality').textContent = qualityDescriptor(data.signal_quality_pct);

  // HRV numbers panel
  document.getElementById('numRmssd').textContent = data.hrv_rmssd_ms != null ? `${data.hrv_rmssd_ms} ms` : '-- ms';
  document.getElementById('numSdnn').textContent = data.hrv_sdnn_ms != null ? `${data.hrv_sdnn_ms} ms` : '-- ms';
  document.getElementById('numMeanrr').textContent = data.mean_ibi_ms != null ? `${Math.round(data.mean_ibi_ms)} ms` : '-- ms';
  const respEl = document.getElementById('numResp');   // v2: breathing rate
  if (respEl) respEl.textContent = data.breathing_rate_per_min != null ? `${data.breathing_rate_per_min} /min` : '-- /min';

  // Gauge
  drawStressGauge(data.stress_level);
  document.getElementById('gaugeLabel').textContent = data.stress_level || '--';
  document.getElementById('stressNote').innerHTML = data.recommendation
    ? `<b>${data.recommendation.state_label}:</b> ${data.recommendation.nervous_system_note}<br><br>${data.recommendation.disclaimer}`
    : 'Run an analysis to see your HRV-based stress estimate here.';

  // Health insights
  if (data.recommendation) {
    document.getElementById('insightsSummary').textContent =
      `Your HRV-based indicator suggests a ${data.recommendation.state_label.toLowerCase()} state.`;
    document.getElementById('insightsList').innerHTML = `
      <div class="insight-item"><div class="ic">💧</div><div><b>Nutrition</b><span>${data.recommendation.nutrition}</span></div></div>
      <div class="insight-item"><div class="ic">🌙</div><div><b>Sleep</b><span>${data.recommendation.sleep}</span></div></div>
      <div class="insight-item"><div class="ic">🏃</div><div><b>Exercise</b><span>${data.recommendation.exercise}</span></div></div>
      <div class="insight-item"><div class="ic">🧘</div><div><b>Breathing</b><span>${data.recommendation.breathing}</span></div></div>
    `;
  } else {
    document.getElementById('insightsSummary').textContent = DEFAULT_INSIGHTS_SUMMARY;
    document.getElementById('insightsList').innerHTML = DEFAULT_INSIGHTS_HTML;
  }

  // v2: range bars + status labels ("is my number high or low?")
  if (window.PULSEREAD_RANGES) window.PULSEREAD_RANGES.update(data);
}

// Everything that depends on WHICH tab produced the result.
function renderModeExtras(data) {
  if (currentMode === 'upload') {
    // Upload tab: graphs come from the recording's real waveform / beats.
    currentGraph = data.graph || null;
    graphMode = 'recording';
    renderSignalGraph();
    refreshHistory();
    setBadges(true);     // a result means the pipeline found a face + ROI
  }
  // Webcam tab: graphs are driven by PULSEREAD_LIVE_UPDATE (see below).
  document.getElementById('downloadReportBtn').disabled = (data.heart_rate_bpm == null);
}

function stressDescriptor(level) {
  switch (level) {
    case 'Very Low': return 'Very Relaxed';
    case 'Low': return 'Relaxed';
    case 'Moderate': return 'Balanced';
    case 'High': return 'Elevated';
    default: return 'Awaiting data';
  }
}
function qualityDescriptor(pct) {
  if (pct == null) return 'Awaiting data';
  if (pct >= 85) return 'Excellent';
  if (pct >= 65) return 'Good';
  if (pct >= 45) return 'Fair';
  return 'Poor';
}

function stressColor(level) {
  switch (level) {
    case 'Very Low': return '#16a34a';
    case 'Low': return '#84cc16';
    case 'Moderate': return '#f59e0b';
    case 'High': return '#ef4444';
    default: return '#9ca3af';
  }
}
function stressFraction(level) {
  // 0 = calmest (left/green), 1 = most stressed (right/red)
  switch (level) {
    case 'Very Low': return 0.10;   // centre of the dark-green segment
    case 'Low': return 0.30;        // light green
    case 'Moderate': return 0.50;   // amber
    case 'High': return 0.90;       // red
    default: return 0.5;
  }
}

// -------------------- Stress gauge (5-colour semicircle with a needle) --------------------
const GAUGE_W = 300, GAUGE_H = 200;
const GAUGE_COLORS = ['#0f8a4b', '#7cc152', '#f6a81c', '#f26b3a', '#ee1f4d'];   // dark green -> red
let gaugeFrac = 0.5;          // where the needle is drawn right now (0 = calm/left, 1 = stressed/right)
let gaugeAnim = null;

function renderGauge(frac, known) {
  const canvas = document.getElementById('gaugeCanvas');
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== GAUGE_W * dpr) {
    canvas.width = GAUGE_W * dpr;
    canvas.height = GAUGE_H * dpr;
    canvas.style.width = '100%';
    canvas.style.maxWidth = GAUGE_W + 'px';
    canvas.style.height = 'auto';
    canvas.style.aspectRatio = `${GAUGE_W} / ${GAUGE_H}`;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, GAUGE_W, GAUGE_H);

  const cx = GAUGE_W / 2, cy = 142, rOut = 132, rIn = 88;
  const gap = 0.045;                                  // radians between segments
  const seg = Math.PI / GAUGE_COLORS.length;

  // five rounded arc segments
  GAUGE_COLORS.forEach((color, i) => {
    const a0 = Math.PI + i * seg + gap / 2;           // PI = left end, 2*PI = right end (canvas y points down)
    const a1 = Math.PI + (i + 1) * seg - gap / 2;
    ctx.beginPath();
    ctx.arc(cx, cy, rOut, a0, a1);
    ctx.arc(cx, cy, rIn, a1, a0, true);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineJoin = 'round';
    ctx.lineWidth = 8;                                // stroke + fill = softly rounded corners
    ctx.stroke();
    ctx.fill();
    // lighter inner band for the two-tone look
    ctx.beginPath();
    ctx.arc(cx, cy, rIn + 3, a0 + 0.02, a1 - 0.02);
    ctx.arc(cx, cy, rIn + 14, a1 - 0.02, a0 + 0.02, true);
    ctx.closePath();
    ctx.fillStyle = 'rgba(255,255,255,0.20)';
    ctx.fill();
  });

  // LOW / HIGH labels
  ctx.font = '800 15px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillStyle = GAUGE_COLORS[0];
  ctx.fillText('LOW', cx - (rOut + rIn) / 2, cy + 26);
  ctx.fillStyle = GAUGE_COLORS[4];
  ctx.fillText('HIGH', cx + (rOut + rIn) / 2, cy + 26);

  // needle: tapered pointer + ring hub
  const angle = Math.PI + frac * Math.PI;
  const len = rOut - 22, baseW = 7;
  const ux = Math.cos(angle), uy = Math.sin(angle);
  const px = -uy, py = ux;                            // perpendicular
  const needle = known ? '#2b2f33' : '#b8bec7';
  ctx.beginPath();
  ctx.moveTo(cx + ux * len, cy + uy * len);
  ctx.lineTo(cx + px * baseW, cy + py * baseW);
  ctx.lineTo(cx - px * baseW, cy - py * baseW);
  ctx.closePath();
  ctx.fillStyle = needle;
  ctx.fill();
  ctx.beginPath();
  ctx.arc(cx, cy, 14, 0, Math.PI * 2);
  ctx.fillStyle = needle;
  ctx.fill();
  ctx.beginPath();
  ctx.arc(cx, cy, 6.5, 0, Math.PI * 2);
  ctx.fillStyle = '#fff';
  ctx.fill();

  // title
  ctx.font = '800 34px Inter, sans-serif';
  ctx.fillStyle = '#2b2f33';
  ctx.fillText('STRESS', cx, GAUGE_H - 6);
}

function drawStressGauge(level) {
  const known = ['Very Low', 'Low', 'Moderate', 'High'].includes(level);
  const target = known ? stressFraction(level) : 0.5;
  if (gaugeAnim) cancelAnimationFrame(gaugeAnim);
  const start = gaugeFrac, t0 = performance.now(), dur = 700;
  const step = (now) => {
    const k = Math.min(1, (now - t0) / dur);
    const ease = 1 - Math.pow(1 - k, 3);
    gaugeFrac = start + (target - start) * ease;
    renderGauge(gaugeFrac, known);
    gaugeAnim = k < 1 ? requestAnimationFrame(step) : null;
  };
  gaugeAnim = requestAnimationFrame(step);
  const label = document.getElementById('gaugeLabel');
  if (label) label.style.color = known ? stressColor(level) : '';
}
// Draw an idle gauge on load
window.addEventListener('DOMContentLoaded', () => drawStressGauge(null));

// -------------------- Graphs --------------------
// All three graphs are drawn from the REAL signal:
//   rPPG Signal : the pulse waveform with every detected beat marked
//   HRV Trend   : beat-to-beat intervals (tachogram) - the raw material of HRV
//   Heart Rate  : instantaneous heart rate beat by beat + the reported average
// (graph data is built by graph_data.py on the backend.)

const GRID = 'rgba(0,0,0,0.06)';
let graphMode = 'recording';   // 'recording' = uploaded video, 'live' = webcam session

function activeTab() {
  const b = document.querySelector('.tabs-mini button.active');
  return (b && b.dataset.graph) || 'rppg';
}

document.querySelectorAll('.tabs-mini button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tabs-mini button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    renderSignalGraph();
  });
});

// ---- small helpers ----
function drawEmptyMessage(canvas, text) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth || 400;
  const h = canvas.height = canvas.clientHeight || 220;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#9ca3af';
  ctx.font = '13px Inter, sans-serif';
  ctx.textAlign = 'center';
  const maxChars = Math.max(20, Math.floor(w / 7.5));
  const lines = [];
  let line = '';
  text.split(' ').forEach(word => {
    if ((line + ' ' + word).trim().length > maxChars) { lines.push(line); line = word; }
    else { line = (line + ' ' + word).trim(); }
  });
  if (line) lines.push(line);
  const y0 = h / 2 - ((lines.length - 1) * 10);
  lines.forEach((l, i) => ctx.fillText(l, w / 2, y0 + i * 20));
}

function chartAvailable(canvas) {
  if (typeof Chart !== 'undefined') return true;
  drawEmptyMessage(canvas, 'Chart library did not load. Check the Chart.js <script> tag in dashboard.html.');
  return false;
}

function validNums(arr) { return arr.filter(v => v != null && isFinite(v)); }
function meanOf(arr) { return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null; }

// Y-axis range that hugs the data but never exaggerates tiny changes
// (a 70.5 -> 70.8 BPM wiggle must not fill the whole chart).
function niceRange(values, minSpan, hardMin, hardMax, step) {
  const v = validNums(values);
  if (!v.length) return { min: hardMin, max: hardMax };
  let lo = Math.min(...v), hi = Math.max(...v);
  const pad = Math.max(step / 2, (hi - lo) * 0.15);
  lo -= pad; hi += pad;
  if (hi - lo < minSpan) { const mid = (lo + hi) / 2; lo = mid - minSpan / 2; hi = mid + minSpan / 2; }
  return {
    min: Math.max(hardMin, Math.floor(lo / step) * step),
    max: Math.min(hardMax, Math.ceil(hi / step) * step),
  };
}

function baseOptions(o) {
  return {
    responsive: true, maintainAspectRatio: false, animation: false,
    parsing: false, normalized: true,
    interaction: { mode: 'nearest', intersect: false },
    plugins: {
      legend: o.legend
        ? { display: true, position: 'top', align: 'end', labels: { boxWidth: 10, boxHeight: 10, usePointStyle: true, font: { size: 11 } } }
        : { display: false },
      title: o.title
        ? { display: true, text: o.title, align: 'start', color: '#6b7280', font: { size: 12, weight: '600' }, padding: { bottom: 8 } }
        : { display: false },
    },
    scales: {
      x: {
        type: 'linear', grid: { color: GRID },
        title: { display: true, text: o.xTitle },
        ticks: { maxTicksLimit: 8, callback: v => +Number(v).toFixed(1) },
      },
      y: Object.assign({ title: { display: true, text: o.yTitle }, grid: { color: GRID } }, o.y || {}),
    },
  };
}

function destroySignalChart() {
  if (signalChart) { signalChart.destroy(); signalChart = null; }
}

// ---- the three chart types ----
function drawRppgChart(canvas, g, title) {
  const fps = g.bvp_fps || 15;
  const wave = g.bvp.map((v, i) => ({ x: i / fps, y: v }));
  const peaks = g.peak_t.map((t, i) => ({ x: t, y: g.peak_y[i] }));
  return new Chart(canvas, {
    type: 'line',
    data: {
      datasets: [
        { label: 'rPPG signal', data: wave, borderColor: '#2563eb', borderWidth: 1.6, pointRadius: 0, tension: 0.2, order: 2 },
        { label: 'Detected beats', data: peaks, showLine: false, pointRadius: 3.5, pointHoverRadius: 5,
          pointBackgroundColor: '#e11d48', pointBorderColor: '#fff', pointBorderWidth: 1, order: 1 },
      ],
    },
    options: baseOptions({ xTitle: 'Time (s)', yTitle: 'Amplitude (normalised)', title, legend: true,
                           y: { suggestedMin: -3, suggestedMax: 3 } }),
  });
}

function drawBeatHrChart(canvas, g, avgBpm) {
  const raw = g.beat_t.map((t, i) => ({ x: t, y: g.hr_inst[i] }));
  const smooth = g.beat_t.map((t, i) => ({ x: t, y: g.hr_smooth[i] }));
  const avg = avgBpm != null ? avgBpm : meanOf(validNums(g.hr_inst));
  const x0 = g.beat_t[0], x1 = g.beat_t[g.beat_t.length - 1];
  const r = niceRange(validNums(g.hr_inst).concat([avg]), 30, 30, 200, 5);
  return new Chart(canvas, {
    type: 'line',
    data: {
      datasets: [
        { label: 'Each beat', data: raw, showLine: false, pointRadius: 2.5, pointBackgroundColor: 'rgba(225,29,72,0.35)', order: 3 },
        { label: '5-beat average', data: smooth, borderColor: '#e11d48', borderWidth: 2.5, pointRadius: 0,
          cubicInterpolationMode: 'monotone', spanGaps: true, order: 1 },
        { label: `Reported ${Math.round(avg)} BPM`, data: [{ x: x0, y: avg }, { x: x1, y: avg }],
          borderColor: '#6b7280', borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0, order: 2 },
      ],
    },
    options: baseOptions({ xTitle: 'Time (s)', yTitle: 'Heart rate (BPM)', legend: true,
                           title: 'Heart rate beat by beat', y: { min: r.min, max: r.max } }),
  });
}

function drawTachogram(canvas, g) {
  const pts = g.beat_t.map((t, i) => ({ x: t, y: g.ibi_ms[i] }));
  const valid = validNums(g.ibi_ms);
  const m = meanOf(valid);
  const x0 = g.beat_t[0], x1 = g.beat_t[g.beat_t.length - 1];
  const r = niceRange(valid, 300, 300, 1600, 50);
  const hrvText = (lastResult && lastResult.hrv_available && lastResult.hrv_rmssd_ms != null)
    ? `RMSSD ${lastResult.hrv_rmssd_ms} ms · SDNN ${lastResult.hrv_sdnn_ms} ms`
    : 'HRV not reliable for this recording';
  return new Chart(canvas, {
    type: 'line',
    data: {
      datasets: [
        { label: 'Beat-to-beat interval', data: pts, borderColor: '#7c5cfc', backgroundColor: '#7c5cfc',
          borderWidth: 2, pointRadius: 3, tension: 0, spanGaps: false, order: 1 },
        { label: `Mean ${Math.round(m)} ms`, data: [{ x: x0, y: m }, { x: x1, y: m }],
          borderColor: '#6b7280', borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0, order: 2 },
      ],
    },
    options: baseOptions({ xTitle: 'Time (s)', yTitle: 'Interval between beats (ms)', legend: true,
                           title: `Beat-to-beat intervals · ${hrvText}`, y: { min: r.min, max: r.max } }),
  });
}

// ---- main entry: redraws the big "Signal Graphs" card ----
const EMPTY_TEXT = {
  rppg: 'Upload a video and press Analyze (or start the webcam) to see your pulse waveform here.',
  hrv: 'Upload a video and press Analyze (or start the webcam) to see your beat-to-beat intervals here.',
  hr: 'Upload a video and press Analyze (or start the webcam) to see your heart rate here.',
  resp: 'Breathing needs at least 25-30 seconds of clean signal. Analyse a video or keep the webcam running.',
};
const TOO_FEW_BEATS = 'Not enough clean heartbeats were detected to draw this graph. Re-record facing a light and hold still.';

function renderSignalGraph() {
  const canvas = document.getElementById('signalCanvas');
  destroySignalChart();
  canvas.style.display = 'block';
  const legacyImg = document.getElementById('waveformImg');
  if (legacyImg) legacyImg.style.display = 'none';

  const mode = activeTab();
  if (mode === 'resp') { renderRespGraph(canvas); return; }     // v2: breathing waveform (same for upload + live)
  if (graphMode === 'live') renderLiveGraph(canvas, mode);
  else renderRecordingGraph(canvas, mode);
}

// v2: breathing waveform from respiration.py (full 0.1-0.5 Hz band, not narrowed)
function renderRespGraph(canvas) {
  const r = lastResult && lastResult.resp;
  if (lastResult && lastResult.status === 'ok' && r === undefined) {
    // the result has no 'resp' field at all -> the BACKEND is still the old version
    drawEmptyMessage(canvas, 'Backend not updated: copy respiration.py, openrppg_pipeline.py and main.py into backend/, then restart app.py (and live_server.py).');
    return;
  }
  if (!r || !r.wave || !r.wave.t || !r.wave.t.length) {
    drawEmptyMessage(canvas, r && r.reason ? `Breathing: ${r.reason}.` : EMPTY_TEXT.resp);
    return;
  }
  if (!chartAvailable(canvas)) return;
  const t0 = r.wave.t[0];
  const pts = r.wave.t.map((t, i) => ({ x: t - t0, y: r.wave.y[i] }));
  const m = r.methods || {};
  const used = ['FM', 'AM', 'BW'].filter(k => m[k] != null).map(k => `${k} ${m[k]}`).join(' · ');
  const rate = r.rate_bpm != null ? r.rate_bpm : '--';
  signalChart = new Chart(canvas, {
    type: 'line',
    data: { datasets: [{ label: 'Breathing waveform', data: pts, borderColor: '#06b6d4', borderWidth: 2,
                         pointRadius: 0, tension: 0.35 }] },
    options: baseOptions({ xTitle: 'Time (s)', yTitle: 'Breathing (normalised)', legend: false,
                           title: `Breathing about ${rate}/min (${r.confidence}) · per method: ${used || 'n/a'}`,
                           y: { suggestedMin: -3, suggestedMax: 3 } }),
  });
}

function renderRecordingGraph(canvas, mode) {
  const g = currentGraph;
  if (!g || !g.bvp || !g.bvp.length) { drawEmptyMessage(canvas, EMPTY_TEXT[mode] || EMPTY_TEXT.rppg); return; }
  if (mode !== 'rppg' && g.n_valid_ibi < 3) { drawEmptyMessage(canvas, TOO_FEW_BEATS); return; }
  if (!chartAvailable(canvas)) return;

  if (mode === 'rppg') {
    const src = lastResult && lastResult.graph_source === 'pulseread' ? ' (PulseRead signal)' : '';
    signalChart = drawRppgChart(canvas, g, `Pulse waveform${src} · ${g.n_beats} beats detected`);
  } else if (mode === 'hr') {
    signalChart = drawBeatHrChart(canvas, g, lastResult ? lastResult.heart_rate_bpm : null);
  } else {
    signalChart = drawTachogram(canvas, g);
  }
}

// Webcam tab: all three graphs now look exactly like the uploaded-video graphs. They are drawn from the
// beats of the LATEST analysis window (liveGraph) and redraw after every finished model run.
// (The one-point-per-run trend lives in the "HRV Analysis" card on the right.)
function renderLiveGraph(canvas, mode) {
  const g = liveGraph;
  if (!g || !g.bvp || !g.bvp.length) {
    drawEmptyMessage(canvas, 'Waiting for the first model result (about 20 seconds of video)…');
    return;
  }
  if (mode !== 'rppg' && g.n_valid_ibi < 3) { drawEmptyMessage(canvas, TOO_FEW_BEATS); return; }
  if (!chartAvailable(canvas)) return;

  if (mode === 'rppg') {
    signalChart = drawRppgChart(canvas, g, `Latest window · ${g.n_beats} beats detected`);
  } else if (mode === 'hr') {
    signalChart = drawBeatHrChart(canvas, g, lastResult ? lastResult.heart_rate_bpm : null);
  } else {
    signalChart = drawTachogram(canvas, g);
  }
}

// -------------------- "HRV Analysis" card: HRV across sessions --------------------
function fmtSessionTime(ts) {
  const d = new Date(ts);
  if (isNaN(d)) return '';
  return d.toLocaleString(undefined, { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' });
}

function renderHrvCard() {
  const canvas = document.getElementById('hrvCanvas');
  if (hrvChart) { hrvChart.destroy(); hrvChart = null; }

  // Same chart style in both tabs:
  //  * Upload tab : RMSSD / SDNN across your recorded sessions (reliable ones, most recent 12)
  //  * Webcam tab : RMSSD / SDNN of THIS live session, one point per finished analysis that had a
  //                 reliable HRV reading (runs without HRV are left out so the line stays connected)
  let labels, rmssd, sdnn, emptyText;
  if (graphMode === 'live') {
    const pts = liveSeries.filter(p => p.rmssd != null || p.sdnn != null);
    labels = pts.map(p => `${p.t}s`);
    rmssd = pts.map(p => p.rmssd);
    sdnn = pts.map(p => p.sdnn);
    emptyText = 'HRV points appear here as each analysis finishes (needs signal quality of at least 50% and about 20 s of video).';
  } else {
    const rows = sessionHistory.filter(s => s.hrv_rmssd_ms != null).slice(-12);
    labels = rows.map(s => fmtSessionTime(s.timestamp));
    rmssd = rows.map(s => s.hrv_rmssd_ms);
    sdnn = rows.map(s => s.hrv_sdnn_ms);
    emptyText = 'HRV from your sessions will appear here after a recording with a reliable HRV reading.';
  }

  if (!rmssd.some(v => v != null)) {
    drawEmptyMessage(canvas, emptyText);
    return;
  }
  if (!chartAvailable(canvas)) return;

  hrvChart = new Chart(canvas, {
    type: 'line',
    data: { labels, datasets: [
      { label: 'RMSSD', data: rmssd, borderColor: '#7c5cfc', backgroundColor: '#7c5cfc', borderWidth: 2, pointRadius: 4,
        cubicInterpolationMode: 'monotone', spanGaps: true },
      { label: 'SDNN', data: sdnn, borderColor: '#06b6d4', backgroundColor: '#06b6d4', borderWidth: 2, pointRadius: 4,
        cubicInterpolationMode: 'monotone', spanGaps: true },
    ] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { position: 'top', align: 'end', labels: { boxWidth: 10, boxHeight: 10, usePointStyle: true, font: { size: 11 } } } },
      scales: {
        x: { grid: { display: false }, ticks: { maxRotation: 40, autoSkip: true, maxTicksLimit: 6, font: { size: 10 } } },
        y: { beginAtZero: true, grace: '10%', title: { display: true, text: 'ms' }, grid: { color: GRID } },
      },
    },
  });
}

async function refreshHistory() {
  try {
    const res = await fetch(`${window.PULSEREAD_CONFIG.API_BASE}/api/history`);
    if (!res.ok) return;
    sessionHistory = await res.json();
    if (graphMode !== 'live') renderHrvCard();
  } catch (err) {
    // backend not reachable yet — upload/live flows still work standalone
  }
}

// -------------------- PDF report download --------------------
document.getElementById('downloadReportBtn').addEventListener('click', async () => {
  if (!lastResult) return;
  const btn = document.getElementById('downloadReportBtn');
  btn.disabled = true;
  btn.textContent = 'Generating…';
  try {
    const res = await fetch(`${window.PULSEREAD_CONFIG.API_BASE}/api/generate-report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // v2: include the challenge report (if one was run) so it lands in the PDF
      body: JSON.stringify({ ...lastResult, graph: undefined, challenge: window.PULSEREAD_CHALLENGE_REPORT || undefined }),
    });
    if (!res.ok) throw new Error('report generation failed');
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'pulseread_report.pdf';
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert(`Could not generate the report: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = '⬇ Download PDF Report';
  }
});

// -------------------- Live (webcam) hooks — called by webcam.js --------------------
function startLive() {
  liveActive = true;
  graphMode = 'live';
  liveSeries = [];
  liveGraph = null;
  liveLastRunId = null;
  liveStartMs = Date.now();
  lastResult = null;
  paintTiles(EMPTY_RESULT);
  document.getElementById('downloadReportBtn').disabled = true;
  renderSignalGraph();
  renderHrvCard();
}

// Called for every live update (about once a second). The graphs only redraw when a
// NEW model run has finished (run_id changed); otherwise nothing has changed.
function updateLiveGraphs(data) {
  if (data.run_id == null || data.run_id === liveLastRunId) return;
  liveLastRunId = data.run_id;

  if (data.graph && Array.isArray(data.graph.bvp) && data.graph.bvp.length) liveGraph = data.graph;

  const q = data.signal_quality_pct;
  const minQ = window.PULSEREAD_MIN_QUALITY != null ? window.PULSEREAD_MIN_QUALITY : 30;
  const strongEnough = q == null || q >= minQ;
  liveSeries.push({
    t: Math.round((Date.now() - (liveStartMs || Date.now())) / 1000),
    bpm: strongEnough ? data.heart_rate_bpm : null,
    rmssd: data.hrv_available ? data.hrv_rmssd_ms : null,
    sdnn: data.hrv_available ? data.hrv_sdnn_ms : null,
  });
  if (liveSeries.length > 120) liveSeries.shift();

  renderSignalGraph();
  renderHrvCard();
}

window.PULSEREAD_LIVE_UPDATE = function (payload) {
  if (!liveActive) startLive();          // first update of a webcam session
  if (payload) updateLiveGraphs(payload);
};

// Webcam stopped: keep the last live graphs on screen, but a new upload may take over.
window.PULSEREAD_LIVE_STOP = function () { liveActive = false; };
window.PULSEREAD_LIVE_BEGIN = startLive;   // webcam.js calls this when Start is pressed

// Expose for webcam.js
window.PULSEREAD_RENDER = renderResult;
window.renderSignalGraph = renderSignalGraph;

// -------------------- First paint --------------------
renderSignalGraph();
renderHrvCard();
refreshHistory();
// If Chart.js had to come from the CDN fallback it may arrive after this script ran.
window.addEventListener('load', () => { renderSignalGraph(); renderHrvCard(); });