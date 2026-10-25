/* ==========================================================================
   PulseRead — Physiological Challenge Mode (baseline -> challenge -> recovery)
   Builds its own panel inside the webcam card, drives the phase timer, tells
   live_server.py when each phase starts/ends, and draws the final report.
   Loaded AFTER webcam.js and dashboard.js (see dashboard.html).
   ========================================================================== */
(function () {
  'use strict';

  const host = document.getElementById('panelWebcam');
  if (!host) return;

  // ---------------------------------------------------------------- state
  let socket = null;
  let protocols = [];
  let fullWindow = 30;
  let bufferSec = 0;
  let faceOk = false;
  let multiFace = false;       // single mode saw 2+ faces
  let run = null;              // { proto, t0, idx, total, starts[], timer }
  let pending = null;          // Start pressed, waiting for webcam / face / 30 s of video: { opened, launched, timer }
  let sampleCount = 0;
  let charts = [];
  let audioCtx = null;

  const PHASE_COLORS = { baseline: 'rgba(37,99,235,0.07)', challenge: 'rgba(239,68,68,0.09)', recovery: 'rgba(22,163,74,0.08)' };

  // ---------------------------------------------------------------- UI
  const style = document.createElement('style');
  style.textContent = `
    .ch-panel{margin-top:16px;padding:16px;border:1px solid var(--border,#e8e9f0);border-radius:14px;background:#fff;color:var(--ink,#0e0f1a);}
    .ch-head{display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:space-between;}
    .ch-head b{font-size:15px;}
    .ch-ctl{display:flex;gap:8px;align-items:center;flex-wrap:wrap;}
    .ch-ctl select{padding:8px 10px;border-radius:8px;border:1px solid var(--border,#e8e9f0);font:inherit;font-size:13px;background:#fff;}
    .ch-note{font-size:12.5px;color:var(--muted,#8a8fa3);margin-top:8px;line-height:1.5;}
    .ch-hint{font-size:12.5px;color:var(--ink-soft,#4b5060);margin-top:8px;}
    .ch-run{margin-top:14px;padding:16px;border-radius:12px;background:var(--bg-light,#f6f7fb);text-align:center;}
    .ch-phase-tag{display:inline-block;font-size:11px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;padding:4px 10px;border-radius:999px;color:#fff;}
    .ch-phase-tag.baseline{background:#2563eb}.ch-phase-tag.challenge{background:#ef4444}.ch-phase-tag.recovery{background:#16a34a}
    .ch-count{font-size:44px;font-weight:800;margin:6px 0 2px;font-variant-numeric:tabular-nums;}
    .ch-instr{font-size:14px;color:var(--ink-soft,#4b5060);max-width:520px;margin:0 auto;line-height:1.5;}
    .ch-bar{height:6px;border-radius:6px;background:#e5e7eb;margin-top:14px;overflow:hidden;}
    .ch-bar>div{height:100%;width:0;background:linear-gradient(90deg,var(--purple,#7c5cfc),var(--pink,#ec4899));transition:width .25s linear;}
    .ch-meta{font-size:12px;color:var(--muted,#8a8fa3);margin-top:8px;}
    .ch-pacer{display:flex;flex-direction:column;align-items:center;gap:6px;margin:10px 0 0;}
    .ch-pacer-dot{width:90px;height:90px;border-radius:50%;background:radial-gradient(circle,#67e8f9,#06b6d4);animation:chBreath 10s ease-in-out infinite;}
    @keyframes chBreath{0%,100%{transform:scale(.55)}50%{transform:scale(1)}}
    .ch-card{margin-top:14px;padding:14px;border:1px solid var(--border,#e8e9f0);border-radius:12px;}
    .ch-card h4{margin:0 0 8px;font-size:14px;}
    .ch-table{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0;}
    .ch-table th{background:var(--purple,#7c5cfc);color:#fff;font-weight:600;padding:6px 8px;text-align:left;}
    .ch-table td{padding:6px 8px;border-bottom:1px solid var(--border,#e8e9f0);}
    .ch-chips{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0;}
    .ch-chip{background:var(--bg-light,#f6f7fb);border-radius:8px;padding:6px 10px;font-size:12px;}
    .ch-chip b{display:block;font-size:15px;}
    .ch-read{font-size:13.5px;line-height:1.55;margin:8px 0;}
    .ch-dim{color:#9ca3af;} .ch-chip small{font-size:10.5px;font-weight:500;}
    .ch-warn{font-size:12px;color:#b45309;margin:4px 0;}
    .ch-chart{height:240px;margin-top:8px;}
    .ch-fine{font-size:11.5px;color:var(--muted,#8a8fa3);margin-top:10px;line-height:1.5;}
    .ch-err{color:#b91c1c;font-size:13px;margin-top:8px;}
    .ch-hidden{display:none !important;}
  `;
  document.head.appendChild(style);

  const root = document.createElement('div');
  root.className = 'ch-panel';
  root.innerHTML = `
    <div class="ch-head">
      <b>🧪 Physiological Challenge</b>
      <div class="ch-ctl">
        <select id="chProtocol"><option value="cognitive">Mental arithmetic stress test</option><option value="paced">Slow paced breathing</option></select>
        <button class="btn btn-primary" id="chStart">Start challenge</button>
        <button class="btn btn-dark ch-hidden" id="chCancel">Cancel</button>
      </div>
    </div>
    <div class="ch-note" id="chNote">Baseline → challenge → recovery. Press Start: the webcam opens by itself and the whole challenge runs to the end (one-person webcam).</div>
    <div class="ch-hint" id="chHint"></div>
    <div class="ch-err ch-hidden" id="chErr"></div>
    <div class="ch-run ch-hidden" id="chRun">
      <span class="ch-phase-tag baseline" id="chTag">Baseline</span>
      <div class="ch-count" id="chCount">0:00</div>
      <div class="ch-instr" id="chInstr"></div>
      <div class="ch-pacer ch-hidden" id="chPacer"><div class="ch-pacer-dot" id="chPacerDot"></div><b id="chPacerText">Breathe in</b></div>
      <div class="ch-bar"><div id="chBar"></div></div>
      <div class="ch-meta" id="chMeta"></div>
    </div>
    <div id="chResult"></div>
  `;
  host.appendChild(root);

  const $ = (id) => root.querySelector('#' + id);
  const sel = $('chProtocol'), startBtn = $('chStart'), cancelBtn = $('chCancel');
  const noteEl = $('chNote'), hintEl = $('chHint'), errEl = $('chErr'), runEl = $('chRun');
  const tagEl = $('chTag'), countEl = $('chCount'), instrEl = $('chInstr'), barEl = $('chBar'), metaEl = $('chMeta');
  const pacerEl = $('chPacer'), pacerDot = $('chPacerDot'), pacerText = $('chPacerText');
  const resultEl = $('chResult');

  const nowSec = () => Date.now() / 1000;
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (v, d = 1) => (v == null || !isFinite(v) ? '--' : (+v).toFixed(d).replace(/\.0+$/, ''));
  const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

  function beep(freq) {
    try {
      audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
      const o = audioCtx.createOscillator(), g = audioCtx.createGain();
      o.frequency.value = freq || 660; g.gain.value = 0.05;
      o.connect(g); g.connect(audioCtx.destination);
      o.start(); o.stop(audioCtx.currentTime + 0.15);
    } catch (e) { /* sound is optional */ }
  }

  function currentProtocol() { return protocols.find(p => p.key === sel.value) || protocols[0]; }

  function showNote() {
    const p = currentProtocol();
    noteEl.textContent = p ? `${p.note} ${p.safety}` : 'Baseline → challenge → recovery. Press Start: the webcam opens by itself and the whole challenge runs to the end.';
  }
  sel.addEventListener('change', showNote);

  function refreshStart() {
    if (run) { startBtn.disabled = true; return; }
    if (pending) {
      startBtn.disabled = true;
      cancelBtn.classList.remove('ch-hidden');
      if (!socket || !socket.connected) hintEl.textContent = 'Opening the webcam… (allow camera access if the browser asks)';
      else if (multiFace) hintEl.textContent = 'More than one face is in view: only you should be in front of the camera.';
      else if (!faceOk) hintEl.textContent = 'Waiting for your face to be in view…';
      else if (bufferSec < fullWindow - 0.5) hintEl.textContent = `Warming up: collecting ${fullWindow}s of video (${bufferSec.toFixed(0)}s so far). The challenge starts automatically.`;
      else hintEl.textContent = 'Starting the challenge…';
      return;
    }
    cancelBtn.classList.add('ch-hidden');
    startBtn.disabled = false;
    hintEl.textContent = 'Press Start. The webcam opens by itself, the challenge runs through all phases, then recording stops.';
  }

  function webcamBtn() { return document.getElementById('startWebcamBtn'); }
  function webcamIsRunning() { const b = webcamBtn(); return !!b && b.classList.contains('hidden'); }
  function setBusy(v) { window.PULSEREAD_CHALLENGE_BUSY = !!v; }

  function clearPending() {
    if (pending && pending.timer) clearTimeout(pending.timer);
    pending = null;
  }

  // everything else stops the camera when it is done; so does the challenge
  function stopCamera(msg) {
    setBusy(false);
    if (window.PULSEREAD_AUTOSTOP) window.PULSEREAD_AUTOSTOP(msg);
    else if (window.stopWebcamCapture) window.stopWebcamCapture();
  }

  // launch the timer as soon as the camera is on, a face is in view and the first full window of video exists
  function tryLaunch() {
    if (!pending || pending.launched || run || !socket || !socket.connected) return;
    if (!faceOk || bufferSec < fullWindow - 0.5) return;
    pending.launched = true;
    if (pending.timer) { clearTimeout(pending.timer); pending.timer = null; }
    socket.emit('challenge_start', { protocol: sel.value, ts: nowSec() });
  }

  function showError(msg) {
    errEl.textContent = msg;
    errEl.classList.toggle('ch-hidden', !msg);
  }

  // ---------------------------------------------------------------- socket wiring
  (window.PULSEREAD_SOCKET_HOOKS = window.PULSEREAD_SOCKET_HOOKS || []).push((s) => {
    socket = s;
    bufferSec = 0; faceOk = false;

    s.on('server_info', (info) => {
      protocols = info.protocols || [];
      fullWindow = info.full_window_sec || 30;
      const keep = sel.value;
      sel.innerHTML = protocols.map(p => `<option value="${esc(p.key)}">${esc(p.label)}</option>`).join('');
      if (keep && protocols.some(p => p.key === keep)) sel.value = keep;
      showNote(); refreshStart();
    });
    s.on('live_update', (p) => {
      bufferSec = p.buffer_sec || 0;
      faceOk = !p.no_face;
      multiFace = !!p.multi_face;
      tryLaunch();
      refreshStart();
    });
    s.on('challenge_started', (proto) => { clearPending(); beginRun(proto); });
    s.on('challenge_sample', () => { sampleCount += 1; if (run) metaEl.textContent = `${sampleCount} analysis windows recorded so far`; });
    s.on('challenge_report', (rep) => {
      endRunUi(); renderReport(rep);
      setTimeout(() => stopCamera('Challenge complete · recording stopped'), 1500);   // report is already on screen
    });
    s.on('challenge_error', (m) => {
      const wasPending = !!pending;
      // "not enough video yet" while we are still warming up: just keep waiting and try again on the next update
      if (wasPending && pending.launched && /Wait until the buffer/.test((m && m.message) || '')) { pending.launched = false; return; }
      clearPending(); endRunUi(); setBusy(false);
      showError(m && m.message ? m.message : 'Challenge could not start.');
    });
    s.on('challenge_cancelled', () => { clearPending(); endRunUi(); });
    s.on('disconnect', () => { clearPending(); endRunUi(); setBusy(false); refreshStart(); });
    refreshStart();
  });

  startBtn.addEventListener('click', () => {
    if (run || pending) return;
    if (window.PULSEREAD_MODE === 'multi') return;       // one-person only (people.js also hides this panel in multi-person mode)
    showError('');
    resultEl.innerHTML = '';
    destroyCharts();
    window.PULSEREAD_CHALLENGE_REPORT = undefined;
    beep(520);                                   // also unlocks audio (this click is a user gesture)
    setBusy(true);                               // keeps auto-stop from ending the recording while the challenge runs
    pending = { opened: false, launched: false, timer: null };
    faceOk = false; bufferSec = 0;

    if (!webcamIsRunning()) {                    // open the webcam ourselves (one-person OR multi-person, whichever tab is active)
      const wb = webcamBtn();
      if (!wb) { pending = null; setBusy(false); showError('Webcam button not found on this page.'); return; }
      pending.opened = true;
      wb.click();
      pending.timer = setTimeout(() => {
        if (pending && (!socket || !socket.connected)) {
          clearPending(); setBusy(false);
          showError('Could not open the webcam or reach live_server.py (port 5001). Allow camera access and make sure live_server.py is running, then press Start again.');
          refreshStart();
        }
      }, 60000);
    }
    refreshStart();
    tryLaunch();
  });

  cancelBtn.addEventListener('click', () => {
    if (socket && run) socket.emit('challenge_cancel');
    const hadCam = webcamIsRunning();
    clearPending();
    endRunUi();
    if (hadCam) stopCamera('Challenge cancelled · recording stopped'); else setBusy(false);
  });

  // ---------------------------------------------------------------- phase timer
  function beginRun(proto) {
    const starts = [];
    let acc = 0;
    proto.phases.forEach(ph => { starts.push(acc); acc += ph.seconds; });
    run = { proto, t0: Date.now(), idx: -1, total: acc, starts, timer: null, ended: false };
    sampleCount = 0;
    runEl.classList.remove('ch-hidden');
    cancelBtn.classList.remove('ch-hidden');
    startBtn.disabled = true;
    metaEl.textContent = 'Recording…';
    run.timer = setInterval(tick, 250);
    tick();
  }

  function tick() {
    if (!run) return;
    const elapsed = (Date.now() - run.t0) / 1000;
    const phases = run.proto.phases;

    // advance through every phase boundary we have passed (a throttled tab may skip several)
    while (run.idx + 1 < phases.length && elapsed >= run.starts[run.idx + 1]) {
      run.idx += 1;
      const ph = phases[run.idx];
      socket.emit('challenge_phase', { name: ph.name, ts: run.t0 / 1000 + run.starts[run.idx] });
      beep(run.idx === 0 ? 520 : 760);
      tagEl.textContent = ph.title;
      tagEl.className = 'ch-phase-tag ' + ph.name;
      instrEl.textContent = ph.instruction;
      const paced = run.proto.key === 'paced' && ph.name === 'challenge';
      pacerEl.classList.toggle('ch-hidden', !paced);
      if (paced) { pacerDot.style.animation = 'none'; void pacerDot.offsetWidth; pacerDot.style.animation = ''; }
    }

    if (elapsed >= run.total) { finishRun(); return; }

    const ph = phases[run.idx];
    const inPhase = elapsed - run.starts[run.idx];
    countEl.textContent = mmss(Math.max(0, ph.seconds - inPhase));
    barEl.style.width = Math.min(100, elapsed / run.total * 100) + '%';
    if (!pacerEl.classList.contains('ch-hidden')) pacerText.textContent = (inPhase % 10) < 5 ? 'Breathe in' : 'Breathe out';
  }

  function finishRun() {
    clearInterval(run.timer);
    run.timer = null;
    run.ended = true;
    socket.emit('challenge_end', { ts: run.t0 / 1000 + run.total });
    beep(880);
    tagEl.textContent = 'Finishing';
    tagEl.className = 'ch-phase-tag recovery';
    countEl.textContent = '…';
    instrEl.textContent = 'Done! Analysing the last window — your report appears in a moment.';
    pacerEl.classList.add('ch-hidden');
    barEl.style.width = '100%';
    cancelBtn.classList.add('ch-hidden');
  }

  function endRunUi() {
    if (run && run.timer) clearInterval(run.timer);
    run = null;
    runEl.classList.add('ch-hidden');
    cancelBtn.classList.add('ch-hidden');
    refreshStart();
  }

  // ---------------------------------------------------------------- report
  function destroyCharts() { charts.forEach(c => c.destroy()); charts = []; }

  const phaseBands = {
    id: 'phaseBands',
    beforeDatasetsDraw(chart, args, opts) {
      const { ctx, chartArea, scales } = chart;
      if (!chartArea || !scales.x) return;
      ctx.save();
      (opts.bounds || []).forEach(b => {
        const x0 = Math.max(scales.x.getPixelForValue(b.start), chartArea.left);
        const x1 = Math.min(scales.x.getPixelForValue(b.end == null ? scales.x.max : b.end), chartArea.right);
        if (x1 <= x0) return;
        ctx.fillStyle = PHASE_COLORS[b.name] || 'rgba(0,0,0,0.04)';
        ctx.fillRect(x0, chartArea.top, x1 - x0, chartArea.bottom - chartArea.top);
        ctx.fillStyle = '#6b7280';
        ctx.font = '11px Inter, sans-serif';
        ctx.fillText(b.name, x0 + 6, chartArea.top + 14);
      });
      ctx.restore();
    },
  };

  function renderReport(rep) {
    window.PULSEREAD_CHALLENGE_REPORT = rep;
    const titles = {};
    rep.protocol.phases.forEach(p => { titles[p.name] = p.title; });
    const xMax = Math.max(...rep.phase_bounds.map(b => b.end == null ? 0 : b.end), 1);

    let html = `<div class="ch-card"><h4>📋 ${esc(rep.protocol.label)} — results</h4>`;
    if (!rep.people.length) html += '<div class="ch-read">No analysis window finished during the challenge (is the face visible and the model warmed up?).</div>';
    html += '</div>';
    resultEl.innerHTML = html;

    rep.people.forEach((p, i) => {
      const m = p.metrics || {};
      const minWin = rep.min_windows || 3;
      const rows = ['baseline', 'challenge', 'recovery'].map(n => {
        const ph = (p.phases || {})[n] || {};
        const weak = ph.reliable === false;                       // too few good windows: show the number greyed out, flagged
        const cell = (v) => weak && v != null ? `<span class="ch-dim" title="Too few good windows - not reliable">${fmt(v)} ⚠</span>` : fmt(v);
        return `<tr><td>${esc(titles[n] || n)}</td><td>${cell(ph.hr)}</td><td>${fmt(ph.rmssd)}</td><td>${cell(ph.resp)}</td>` +
               `<td>${fmt(ph.quality, 0)}</td><td>${fmt(ph.n, 0)}${weak ? ' ⚠' : ''}</td></tr>`;
      }).join('');
      const chip = (label, val, tip, dim) => `<div class="ch-chip${dim ? ' ch-dim' : ''}" ${tip ? `title="${esc(tip)}"` : ''}>${esc(label)}<b>${val}</b></div>`;
      const sign = (v, u) => v == null ? '--' : `${v > 0 ? '+' : ''}${fmt(v)}${u}`;
      const needRec = `Needs at least ${minWin} good windows in baseline, challenge and recovery, and a clear change during the challenge`;
      const chips = [
        chip('Heart-rate change', sign(m.hr_change_bpm, ' BPM') + (m.hr_change_bpm != null && m.hr_change_meaningful === false ? ' <small>(within noise)</small>' : ''),
             'Shown only when baseline and challenge both have enough good windows', m.hr_change_meaningful === false),
        chip('HRV (RMSSD) change', sign(m.rmssd_change_pct, '%') + (m.rmssd_change_pct != null && m.rmssd_change_meaningful === false ? ' <small>(within noise)</small>' : ''),
             'A change under about 25% is smaller than the normal window-to-window wobble', m.rmssd_change_meaningful === false),
        chip('Breathing change', sign(m.resp_change_bpm, ' /min'), 'Needs a reliable breathing value in baseline and challenge'),
        chip('HR recovery', m.hr_recovery_pct == null ? '--' : fmt(m.hr_recovery_pct, 0) + '%', needRec),
        chip('HRV recovery', m.rmssd_recovery_pct == null ? '--' : fmt(m.rmssd_recovery_pct, 0) + '%', needRec),
      ].join('');
      const warns = (p.notes || []).map(n => `<div class="ch-warn">⚠ ${esc(n)}</div>`).join('');
      const card = document.createElement('div');
      card.className = 'ch-card';
      card.innerHTML = `
        <h4>${esc(p.label)}</h4>
        <table class="ch-table"><tr><th>Phase</th><th>Heart rate (BPM)</th><th>RMSSD (ms)</th><th>Breathing (/min)</th><th>Quality (%)</th><th>Good windows</th></tr>${rows}</table>
        <div class="ch-chips">${chips}</div>
        <div class="ch-read">${esc(p.reading)}</div>
        ${warns}
        <div class="ch-chart"><canvas id="chChart${i}"></canvas></div>`;
      resultEl.appendChild(card);
      drawTimeline(card.querySelector('canvas'), p, rep.phase_bounds, xMax, rep.min_quality_pct == null ? 55 : rep.min_quality_pct);
    });

    const fine = document.createElement('div');
    fine.className = 'ch-fine';
    fine.textContent = rep.safety + ' Each point is a 20–30 s average, so very fast changes are smoothed. ' +
                       `Only windows with signal quality of ${rep.min_quality_pct == null ? 55 : rep.min_quality_pct}% or more are counted (grey dots on the chart were ignored); ` +
                       'lines break where no good window exists. ⚠ marks a phase with too few good windows. ' +
                       'Download PDF Report (above) includes this table.';
    resultEl.appendChild(fine);
    resultEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function drawTimeline(canvas, person, bounds, xMax, minQ) {
    if (typeof Chart === 'undefined') { canvas.replaceWith(document.createTextNode('(Chart.js not loaded)')); return; }
    const tl = person.timeline || [];
    const good = (s) => s.quality != null && s.quality >= minQ;
    // a missing / weak window becomes a null point, so the line BREAKS there instead of joining its neighbours
    const series = (key) => tl.map(s => ({ x: s.t, y: good(s) && s[key] != null ? s[key] : null }));
    const hr = series('hr');
    const hrv = series('rmssd');
    const ignored = tl.filter(s => !good(s) && s.hr != null).map(s => ({ x: s.t, y: s.hr }));
    charts.push(new Chart(canvas, {
      type: 'line',
      data: { datasets: [
        { label: 'Heart rate (BPM)', data: hr, borderColor: '#e11d48', backgroundColor: '#e11d48', borderWidth: 2, pointRadius: 3, tension: 0, spanGaps: false, yAxisID: 'y' },
        { label: 'RMSSD (ms)', data: hrv, borderColor: '#7c5cfc', backgroundColor: '#7c5cfc', borderWidth: 2, pointRadius: 3, tension: 0, spanGaps: false, yAxisID: 'y1' },
        { label: `Ignored (quality < ${minQ}%)`, data: ignored, showLine: false, borderColor: '#9ca3af', backgroundColor: 'rgba(0,0,0,0)', pointStyle: 'crossRot', pointRadius: 4, borderWidth: 1.5, yAxisID: 'y' },
      ] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
        interaction: { mode: 'nearest', intersect: false },
        plugins: {
          legend: { position: 'top', align: 'end', labels: { boxWidth: 10, boxHeight: 10, usePointStyle: true, font: { size: 11 } } },
          phaseBands: { bounds },
        },
        scales: {
          x: { type: 'linear', min: 0, max: xMax, title: { display: true, text: 'Seconds since the challenge started (end of each analysis window)' } },
          y: { position: 'left', title: { display: true, text: 'BPM' }, grid: { color: 'rgba(0,0,0,0.06)' } },
          y1: { position: 'right', title: { display: true, text: 'ms' }, grid: { drawOnChartArea: false } },
        },
      },
      plugins: [phaseBands],
    }));
  }

  // mode change (Upload / One-person / Multi-person): erase the challenge result so it never shows in another mode
  (window.PULSEREAD_CLEAR_HOOKS = window.PULSEREAD_CLEAR_HOOKS || []).push(() => {
    clearPending();
    endRunUi();
    setBusy(false);
    destroyCharts();
    resultEl.innerHTML = '';
    showError('');
    window.PULSEREAD_CHALLENGE_REPORT = undefined;
  });

  showNote();
  refreshStart();
})();
