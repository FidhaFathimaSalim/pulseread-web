/* ==========================================================================
   PulseRead — Multi-person webcam   (v3: every person gets their OWN dashboard)

   * its own top tab: "Upload Video | One-person Webcam | Multi-person Webcam"
   * a coloured box on every face, so you can see who is who
   * a short "People in view" list next to the camera (click a person to jump to their dashboard)
   * below the camera: ONE FULL DASHBOARD PER PERSON - their own Key Results, stress, breathing,
     four graphs (rPPG, beat intervals, heart rate, breathing), HRV over time, guidance and PDF report
   * the shared single-person cards (Key Results, Signal Graphs, HRV Analysis, Stress Level) are hidden
     in this mode - the one-person webcam tab is not touched at all
   * clear messages when something is not working (old server, no faces, no face detector)

   Needs the v3 live_server.py (it sends one 'person_update' per person every second).
   Loaded AFTER webcam.js and dashboard.js (see dashboard.html).
   ========================================================================== */
(function () {
  'use strict';

  const panel = document.getElementById('panelWebcam');
  const stage = document.getElementById('webcamStage');
  const video = document.getElementById('webcamVideo');
  const startBtn = document.getElementById('startWebcamBtn');
  const tabUpload = document.getElementById('tabUpload');
  const tabWebcam = document.getElementById('tabWebcam');
  const tabMulti = document.getElementById('tabMulti');
  if (!panel || !stage || !video || !tabWebcam) return;

  const COLORS = ['#7c5cfc', '#06b6d4', '#ec4899', '#f59e0b'];
  const colorOf = (id) => COLORS[(id - 1) % COLORS.length];
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const KIND = { ok: '#16a34a', good: '#0d9488', warn: '#d97706', bad: '#dc2626', low: '#2563eb', high: '#ea580c' };

  // same cut-off webcam.js uses: a number from a very weak signal is mostly noise
  const minQ = () => (window.PULSEREAD_MIN_QUALITY != null ? window.PULSEREAD_MIN_QUALITY : 30);
  const usable = (d) => !!d && d.heart_rate_bpm != null && (d.signal_quality_pct == null || d.signal_quality_pct >= minQ());
  const RR = () => window.PULSEREAD_RANGES;
  const fn = (name) => (typeof window[name] === 'function' ? window[name] : null);

  // ---------------------------------------------------------------- state
  let socket = null;
  let mode = 'single';
  let multiFrameWidth = 1280;
  let people = [];                 // latest people_update (boxes + short summary)
  let activeId = null;
  let internalClick = false;
  let gotModeSet = false;
  let gotPersonUpdate = false;
  const view = {};                 // id -> smoothed box (normalised frame coordinates)
  let rafOn = false;
  const pp = {};                   // id -> { uid, data, graph, series, lastRun, tab, chart, el, q, startMs }

  // ---------------------------------------------------------------- styles
  const style = document.createElement('style');
  style.textContent = `
    body.pp-multi .pp-hide{display:none !important;}
    body.pp-multi .insights-card, body.pp-multi #rngHelp{display:none !important;}
    body.pp-multi .ch-panel{display:none !important;}     /* Physiological Challenge is one-person only */
    .pp-status{margin:12px 0 0;padding:10px 14px;border-radius:10px;font-size:13px;line-height:1.5;background:#efeaff;color:#3b2f8f;font-weight:600;}
    .pp-status.warn{background:#fee2e2;color:#991b1b;}
    .pp-status small{display:block;font-weight:500;opacity:.85;}
    .pp-hint{text-align:center;font-size:11.5px;color:var(--muted,#8a8fa3);margin-top:8px;}
    #ppOverlay{position:absolute;inset:0;width:100%;height:100%;z-index:3;pointer-events:none;}
    #ppOverlay.on{pointer-events:auto;cursor:pointer;}
    .pp-hidden{display:none !important;}

    /* "People in view" list (right of the camera) */
    .pp-overview .pp-sub{font-size:12px;color:var(--muted,#8a8fa3);margin:-6px 0 12px;line-height:1.5;}
    .pp-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;}
    .pp-card{color:#0e0f1a;border:1px solid var(--border,#e8e9f0);border-top:5px solid var(--c,#7c5cfc);border-radius:12px;padding:12px 14px;background:#fff;cursor:pointer;transition:box-shadow .15s;}
    .pp-card:hover{box-shadow:var(--shadow,0 2px 10px rgba(0,0,0,.1));}
    .pp-card.sel{box-shadow:0 0 0 3px color-mix(in srgb,var(--c,#7c5cfc) 28%,transparent);}
    .pp-card .n{font-size:13px;font-weight:800;display:flex;align-items:center;justify-content:space-between;gap:8px;}
    .pp-card .n i{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--c);margin-right:6px;}
    .pp-chip{font-size:10px;font-weight:800;letter-spacing:.04em;text-transform:uppercase;padding:2px 8px;border-radius:999px;background:#e5e7eb;color:#4b5060;white-space:nowrap;}
    .pp-chip.ready{background:#dcfce7;color:#166534;} .pp-chip.analysing{background:#fef3c7;color:#92400e;}
    .pp-card .hr{font-size:30px;font-weight:800;margin:6px 0 2px;}
    .pp-card .hr small{font-size:11px;color:var(--muted,#8a8fa3);font-weight:600;margin-left:4px;}
    .pp-card .r{font-size:12px;color:var(--ink-soft,#4b5060);display:flex;justify-content:space-between;padding:2px 0;}
    .pp-card .r b{color:var(--ink,#0e0f1a);}
    .pp-card .wait{font-size:12px;color:var(--ink-soft,#4b5060);margin:4px 0 6px;}
    .pp-card .q{height:4px;border-radius:4px;background:#e5e7eb;margin-top:8px;overflow:hidden;}
    .pp-card .q>div{height:100%;background:var(--c);}
    .pp-card .go{font-size:11px;font-weight:700;color:var(--c);margin-top:6px;}

    /* one full dashboard per person */
    .ppb-board{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:20px;margin-top:20px;align-items:start;}
    @media (max-width:620px){.ppb-board{grid-template-columns:1fr;}}
    .ppb{border-top:6px solid var(--c,#7c5cfc);min-width:0;}
    .ppb-head{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-bottom:8px;}
    .ppb-title{display:flex;align-items:center;gap:8px;font-size:17px;}
    .ppb-title i{display:inline-block;width:12px;height:12px;border-radius:50%;background:var(--c);}
    .ppb-actions{display:flex;align-items:center;gap:10px;}
    .ppb-buf{font-size:12px;color:var(--muted,#8a8fa3);}
    .ppb-dl{padding:8px 14px;font-size:12.5px;}
    .ppb-dl:disabled{opacity:.45;cursor:not-allowed;}
    .ppb-note{font-size:12.5px;color:var(--ink-soft,#4b5060);margin:0 0 12px;min-height:18px;line-height:1.45;}
    .ppb-tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
    @media (max-width:620px){.ppb-tiles{grid-template-columns:repeat(2,1fr);}}
    .ppb-tiles .result-tile{padding:12px;min-width:0;}
    .ppb-tiles .result-tile .value{font-size:22px;}
    .ppb-tiles .result-tile .sub{font-size:11.5px;}
    .ppb-chipline{margin-top:6px;font-size:10.5px;color:var(--ink-soft,#4b5060);line-height:1.7;}
    .ppb-chip{display:inline-block;font-size:9.5px;font-weight:800;letter-spacing:.03em;text-transform:uppercase;color:#fff;padding:2px 7px;border-radius:999px;margin-right:4px;}
    .ppb-graph{margin-top:16px;}
    .ppb-tabs{display:flex;flex-wrap:wrap;background:var(--bg-light,#f6f7fb);border-radius:8px;padding:3px;gap:2px;width:fit-content;max-width:100%;}
    .ppb-tabs button{border:none;background:transparent;padding:6px 10px;border-radius:6px;font-size:12px;font-weight:600;color:var(--ink-soft,#4b5060);}
    .ppb-tabs button.active{background:#fff;box-shadow:var(--shadow,0 2px 10px rgba(0,0,0,.1));color:var(--ink,#0e0f1a);}
    .ppb-chart{position:relative;height:250px;min-width:0;margin-top:10px;}
    .ppb-chart canvas{height:100% !important;}
    .ppb-guid{margin-top:12px;font-size:12.5px;color:var(--ink-soft,#4b5060);}
    .ppb-guid summary{cursor:pointer;font-weight:700;color:var(--ink,#0e0f1a);}
    .ppb-guid p{margin:8px 0 0;line-height:1.5;}
    .ppb-fine{font-size:11px;color:var(--muted,#8a8fa3);margin-top:10px;line-height:1.5;}
  `;
  document.head.appendChild(style);

  // ---------------------------------------------------------------- page pieces
  const status = document.createElement('div');
  status.className = 'pp-status pp-hidden';
  stage.parentNode.insertBefore(status, stage);

  const hint = document.createElement('div');
  hint.className = 'pp-hint pp-hidden';
  const cta = panel.querySelector('.webcam-cta');
  if (cta) cta.parentNode.insertBefore(hint, cta.nextSibling);

  const overlay = document.createElement('canvas');
  overlay.id = 'ppOverlay';
  stage.appendChild(overlay);

  // "People in view": goes where the (hidden) Key Results card is
  const keyEl = document.getElementById('valBpm');
  const keyCard = keyEl && keyEl.closest ? keyEl.closest('.card') : null;
  const overview = document.createElement('div');
  overview.className = 'card pp-overview pp-hidden';
  overview.innerHTML = '<div class="card-head"><h3>People in view</h3></div>' +
    '<div class="pp-sub">Each person has their own full dashboard further down — click a person to jump to it.</div>' +
    '<div class="pp-cards"></div>';
  if (keyCard) keyCard.parentNode.insertBefore(overview, keyCard.nextSibling);
  else stage.parentNode.insertBefore(overview, stage.nextSibling);
  const cards = overview.querySelector('.pp-cards');

  // the board with one dashboard per person: full width, right under the camera row
  const board = document.createElement('div');
  board.className = 'ppb-board pp-hidden';
  const grid = panel.closest('.dash-grid');
  if (grid) grid.parentNode.insertBefore(board, grid.nextSibling);
  else panel.parentNode.appendChild(board);

  // hide the shared single-person cards while multi mode is on (CSS only applies with body.pp-multi)
  function tagHidden() {
    const mark = (el) => { if (el) el.classList.add('pp-hide'); };
    if (keyCard) mark(keyCard);
    const sc = document.getElementById('signalCanvas');
    const hc = document.getElementById('hrvCanvas');
    const gc = document.getElementById('gaugeCanvas');
    const row = sc && sc.closest ? sc.closest('.dash-grid-3') : null;
    if (row) mark(row);
    else [sc, hc, gc].forEach(c => { if (c && c.closest) mark(c.closest('.card')); });
  }

  function running() { return startBtn.classList.contains('hidden'); }   // Start is hidden while streaming

  // ---------------------------------------------------------------- modes + tabs
  function syncTabs() {
    if (!tabMulti) return;
    const webcamVisible = !panel.classList.contains('hidden');
    tabMulti.classList.toggle('active', webcamVisible && mode === 'multi');
    tabWebcam.classList.toggle('active', webcamVisible && mode === 'single');
  }

  function setMode(m) {
    const changed = m !== mode;
    mode = m;
    window.PULSEREAD_MODE = m;
    window.PULSEREAD_FRAME_WIDTH = m === 'multi' ? multiFrameWidth : undefined;
    window.PULSEREAD_ACTIVE_PERSON = null;
    activeId = null;
    people = [];
    Object.keys(view).forEach(k => delete view[k]);
    clearBoard();
    const multi = m === 'multi';
    tagHidden();
    document.body.classList.toggle('pp-multi', multi);
    status.classList.toggle('pp-hidden', !multi);
    overview.classList.toggle('pp-hidden', !multi);
    board.classList.toggle('pp-hidden', !multi);
    hint.classList.toggle('pp-hidden', !multi);
    overlay.classList.toggle('on', multi);
    if (multi) {
      status.className = 'pp-status';
      status.innerHTML = '👥 Multi-person mode — press <b>Start Webcam</b>. Up to 4 faces are tracked and measured separately, each with their own dashboard below.';
      hint.textContent = 'Everyone should face the camera, well lit, whole face visible, about an arm’s length away. Several people are analysed at the same time, so each person’s numbers refresh every few seconds (more people = slightly slower).';
    }
    renderCards();
    draw();
    syncTabs();
    // One-person <-> Multi-person: the other mode's results must not stay on screen
    if (changed && window.PULSEREAD_CLEAR_OUTPUT) window.PULSEREAD_CLEAR_OUTPUT();
  }

  if (tabMulti) {
    tabMulti.addEventListener('click', () => {
      if (running() && window.stopWebcamCapture) window.stopWebcamCapture();     // mode can only change while stopped
      internalClick = true;
      tabWebcam.click();                                                         // let dashboard.js open the webcam panel
      internalClick = false;
      setMode('multi');
    });
  }
  tabWebcam.addEventListener('click', () => {
    if (internalClick) return;
    if (mode !== 'single') {
      if (running() && window.stopWebcamCapture) window.stopWebcamCapture();
      setMode('single');
    }
    setTimeout(syncTabs, 0);
  });
  // Back on the Upload tab: remove everything multi-person added, so the page is the original Upload layout again
  if (tabUpload) tabUpload.addEventListener('click', () => {
    if (running() && window.stopWebcamCapture) window.stopWebcamCapture();
    if (mode !== 'single') setMode('single');          // drops body.pp-multi, hides People-in-view / board / banner
    setTimeout(syncTabs, 0);
  });

  // ---------------------------------------------------------------- socket wiring
  (window.PULSEREAD_SOCKET_HOOKS = window.PULSEREAD_SOCKET_HOOKS || []).push((s) => {
    socket = s;
    gotModeSet = false;
    gotPersonUpdate = false;
    clearBoard();                                   // a new webcam session starts with empty dashboards
    s.on('server_info', (info) => {
      multiFrameWidth = info.multi_frame_width || 1280;
      if (mode === 'multi') window.PULSEREAD_FRAME_WIDTH = multiFrameWidth;
    });
    s.on('mode_set', () => { gotModeSet = true; });
    s.on('people_update', onPeople);
    s.on('person_update', onPerson);
    s.on('disconnect', () => {                      // keep the dashboards on screen (last results, graphs, PDF) after Stop
      people = [];
      Object.keys(pp).forEach(k => { const st = pp[k]; if (st) setPill(st, 'Stopped', ''); });
      renderCards(); draw();
    });
    // An old live_server.py never answers 'config' with 'mode_set' -> tell the user clearly.
    setTimeout(() => {
      if (mode === 'multi' && socket === s && s.connected && !gotModeSet) {
        status.className = 'pp-status warn';
        status.innerHTML = '⚠ The live server is the OLD version, so multi-person cannot work. ' +
          '<small>Close the running <b>live_server.py</b>, copy the new live_server.py + face_tracker.py into backend/, and start it again. ' +
          'The console should say “PulseRead live_server v2 starting”.</small>';
      }
    }, 4500);
    // A v2 server answers 'mode_set' but does not send person_update: dashboards would stay empty.
    setTimeout(() => {
      if (mode === 'multi' && socket === s && s.connected && gotModeSet && people.length && !gotPersonUpdate) {
        status.className = 'pp-status warn';
        status.innerHTML = '⚠ The live server does not send per-person results yet. ' +
          '<small>Copy the newest <b>live_server.py</b> into backend/ and restart it (it must send “person_update”). ' +
          'Until then each person only shows the short summary.</small>';
      }
    }, 9000);
  });

  function selectPerson(id) { if (socket && id !== activeId) socket.emit('select_person', { id }); }

  function onPeople(msg) {
    people = msg.people || [];
    const sel = people.find(p => p.selected);
    activeId = sel ? sel.id : null;
    people.forEach(p => { if (p.box && !view[p.id]) view[p.id] = { ...p.box }; });

    // a face that left takes its dashboard with it; a reused "Person N" number for a NEW face starts from scratch
    Object.keys(pp).forEach(k => {
      const id = +k, st = pp[id];
      const p = people.find(x => x.id === id);
      if (!p || (p.uid != null && st.uid != null && p.uid !== st.uid)) removeSection(id);
    });
    let created = false;
    people.forEach(p => { if (!pp[p.id]) { createSection(p.id, p.uid); created = true; } });
    if (created) sortBoard();
    people.forEach(p => { const st = pp[p.id]; if (st) { updateHeader(st, p); if (!st.data) updateSection(p.id); } });

    updateStatus(msg);
    renderCards();
    startLoop();
  }

  function updateStatus(msg) {
    if (mode !== 'multi') return;
    if (msg.want_width) window.PULSEREAD_FRAME_WIDTH = msg.want_width;   // server asks for a sharper picture when 2+ people share the frame
    if (msg.detector_error) {
      status.className = 'pp-status warn';
      status.innerHTML = '⚠ No face detector works in the live server’s Python environment. <small>' + esc(msg.detector_error_text || 'See the live_server console.') +
        '<br>Fix: run <b>python check_face_detector.py</b> in the same environment as live_server.py — it prints the exact repair commands (usually: reinstall OpenCV cleanly, or install mediapipe). Then restart live_server.py.</small>';
      return;
    }
    status.className = 'pp-status';
    if (!people.length) {
      status.innerHTML = '🔎 Looking for faces… <small>Face the camera in good light with your whole face visible.' +
        (msg.model_ready === false ? ' The model is still warming up on the server (first start only).' : '') + '</small>';
      return;
    }
    const names = people.map(p => esc(p.label)).join(', ');
    const busy = people.find(p => p.state === 'analysing');
    const minStop = window.PULSEREAD_AUTOSTOP_QUALITY || 50;
    const weakList = people.filter(p => !(p.signal_quality_pct > minStop)).map(p => esc(p.label) + (p.signal_quality_pct != null ? ' (' + p.signal_quality_pct + '%)' : ''));
    status.innerHTML = `👥 <b>${people.length}</b> ${people.length === 1 ? 'face' : 'faces'} detected: ${names}${msg.detector ? ` <span style="opacity:.6;font-weight:500">(finder: ${esc(msg.detector)})</span>` : ''}` +
      `<small>${busy ? esc(busy.label) + ' is being analysed now. ' : ''}Up to two people are analysed at the same time. ` +
      (weakList.length ? `Still below ${minStop}%: <b>${weakList.join(', ')}</b> — they are re-checked first; face the light and keep still. ` : '') +
      `Recording stops by itself only when everyone is above ${minStop}%.</small>`;
  }

  // ---------------------------------------------------------------- "People in view" list
  const STATE_TEXT = { collecting: 'Collecting video', analysing: 'Analysing…', waiting: 'Waiting for a free slot', ready: 'Live' };

  function renderCards() {
    if (mode !== 'multi') return;
    if (!people.length) { cards.innerHTML = ''; return; }
    cards.innerHTML = people.map(p => {
      const q = p.signal_quality_pct != null ? Math.max(0, Math.min(100, p.signal_quality_pct)) : 0;
      const hasHr = usable(p);
      const weak = p.heart_rate_bpm != null && !hasHr;
      const st = p.state || (hasHr ? 'ready' : 'collecting');
      const chipCls = st === 'ready' ? 'ready' : (st === 'analysing' ? 'analysing' : '');
      let hrTag = '';
      if (hasHr && RR()) {
        const a = RR().assess('bpm', p.heart_rate_bpm);
        if (a) hrTag = `<span class="pp-chip" style="background:${KIND[a.kind] || '#9ca3af'};color:#fff;margin-left:6px;vertical-align:middle">${esc(a.status)}</span>`;
      }
      const body = hasHr
        ? `<div class="r"><span>HRV (RMSSD)</span><b>${p.hrv_rmssd_ms != null ? p.hrv_rmssd_ms + ' ms' : '--'}</b></div>
           <div class="r"><span>Signal quality</span><b>${p.signal_quality_pct != null ? p.signal_quality_pct + '%' : '--'}</b></div>`
        : weak ? `<div class="wait">Signal too weak (${p.signal_quality_pct}%) — face the light and hold still.</div>`
        : `<div class="wait">${st === 'collecting' ? `Collecting video: ${(p.buffer_sec || 0).toFixed(0)} s / ${(window.PULSEREAD_FULL_WINDOW || 20)} s` :
            st === 'analysing' ? 'The model is analysing this person now…' : 'Enough video — waiting for a free analysis slot.'}</div>`;
      return `<div class="pp-card ${p.selected ? 'sel' : ''}" style="--c:${colorOf(p.id)}" data-id="${p.id}">
        <div class="n"><span><i></i>${esc(p.label)}</span><span class="pp-chip ${chipCls}">${STATE_TEXT[st] || st}</span></div>
        <div class="hr">${hasHr ? p.heart_rate_bpm : '--'}<small>BPM</small>${hrTag}</div>
        ${body}
        <div class="q"><div style="width:${q}%"></div></div>
        <div class="go">Open ${esc(p.label)}’s dashboard ↓</div></div>`;
    }).join('');
    cards.querySelectorAll('.pp-card').forEach(el => el.addEventListener('click', () => focusPerson(+el.dataset.id)));
  }

  function focusPerson(id) {
    selectPerson(id);
    const st = pp[id];
    if (!st) return;
    st.el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    st.el.classList.remove('v2-flash'); void st.el.offsetWidth; st.el.classList.add('v2-flash');
  }

  // ---------------------------------------------------------------- per-person dashboards
  const TILES = [
    ['bpm', 'Heart Rate', 'tile-red', '❤', 'BPM'],
    ['rmssd', 'HRV (RMSSD)', 'tile-teal', '〰', 'ms'],
    ['sdnn', 'HRV (SDNN)', 'tile-blue', '📈', 'ms'],
    ['stress', 'Stress Level', 'tile-amber', '🙂', ''],
    ['quality', 'Signal Quality', 'tile-blue', '📶', '%'],
    ['resp', 'Breathing', 'tile-teal', '🫁', '/min'],
  ];
  const GRAPH_TABS = [['rppg', 'rPPG Signal'], ['hrv', 'Beat intervals'], ['hr', 'Heart Rate'], ['resp', 'Breathing'], ['trend', 'HRV over time']];

  function createSection(id, uid) {
    const el = document.createElement('section');
    el.className = 'card ppb';
    el.id = 'ppb-' + id;
    el.style.setProperty('--c', colorOf(id));
    el.innerHTML = `
      <div class="ppb-head">
        <div class="ppb-title"><i></i><b data-k="label">Person ${id}</b><span class="pp-chip" data-k="pill">Collecting video</span></div>
        <div class="ppb-actions"><span class="ppb-buf" data-k="buf"></span>
          <button class="btn btn-green ppb-dl" data-k="dl" disabled>⬇ PDF report</button></div>
      </div>
      <div class="ppb-note" data-k="note"></div>
      <div class="ppb-tiles">
        ${TILES.map(([k, label, cls, icon, unit]) => `
          <div class="result-tile ${cls}">
            <div class="top"><span class="label">${label}</span><span class="icon-badge">${icon}</span></div>
            <div class="value"><span data-k="${k}">--</span>${unit ? `<span class="unit">${unit}</span>` : ''}</div>
            <div class="sub" data-k="${k}Sub">Awaiting data</div>
            <div class="ppb-chipline" data-k="${k}Chip"></div>
          </div>`).join('')}
      </div>
      <div class="ppb-graph">
        <div class="ppb-tabs">${GRAPH_TABS.map(([g, label], i) => `<button data-g="${g}" class="${i === 0 ? 'active' : ''}">${label}</button>`).join('')}</div>
        <div class="ppb-chart"><canvas></canvas></div>
      </div>
      <details class="ppb-guid"><summary>Wellness guidance for this person</summary><div data-k="guid"><p>Appears once a reliable reading is available.</p></div></details>
      <div class="ppb-fine">General wellness information from a video estimate — not a medical diagnosis.</div>`;
    board.appendChild(el);

    const st = {
      uid, el, data: null, graph: null, series: [], lastRun: null, tab: 'rppg', chart: null, startMs: Date.now(),
      q: (k) => el.querySelector('[data-k="' + k + '"]'),
    };
    pp[id] = st;
    el.querySelector('.ppb-tabs').addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      el.querySelectorAll('.ppb-tabs button').forEach(x => x.classList.toggle('active', x === b));
      st.tab = b.dataset.g;
      drawGraph(id);
    });
    st.q('dl').addEventListener('click', () => downloadPdf(id));
    updateSection(id);
    drawGraph(id);
    return st;
  }

  function sortBoard() {
    Object.keys(pp).map(Number).sort((a, b) => a - b).forEach(id => board.appendChild(pp[id].el));
  }

  function removeSection(id) {
    const st = pp[id];
    if (!st) return;
    if (st.chart) { st.chart.destroy(); st.chart = null; }
    st.el.remove();
    delete pp[id];
  }

  function clearBoard() { Object.keys(pp).forEach(k => removeSection(+k)); }

  function setPill(st, text, cls) {
    const e = st.q('pill');
    if (!e) return;
    e.textContent = text;
    e.className = 'pp-chip ' + (cls || '');
  }

  function updateHeader(st, p) {
    const stt = p.state || 'collecting';
    setPill(st, STATE_TEXT[stt] || stt, stt === 'ready' ? 'ready' : (stt === 'analysing' ? 'analysing' : ''));
    const lab = st.q('label');
    if (lab && lab.textContent !== p.label) lab.textContent = p.label;
  }

  function onPerson(msg) {
    const id = msg && msg.person_id;
    if (id == null) return;
    gotPersonUpdate = true;
    let st = pp[id];
    if (st && msg.person_uid != null && st.uid != null && st.uid !== msg.person_uid) { removeSection(id); st = null; }
    if (!st) { st = createSection(id, msg.person_uid); sortBoard(); }
    if (st.uid == null) st.uid = msg.person_uid;

    const g = msg.graph;
    const d = { ...msg };
    delete d.graph;
    st.data = d;

    let redraw = false;
    if (msg.run_id != null && msg.run_id !== st.lastRun) {      // a NEW model run finished for this person
      st.lastRun = msg.run_id;
      if (g && Array.isArray(g.bvp) && g.bvp.length) st.graph = g;
      st.series.push({
        t: Math.round((Date.now() - st.startMs) / 1000),
        rmssd: d.hrv_available ? d.hrv_rmssd_ms : null,
        sdnn: d.hrv_available ? d.hrv_sdnn_ms : null,
      });
      if (st.series.length > 120) st.series.shift();
      redraw = true;
    }
    const lab = st.q('label');
    if (lab && d.label && lab.textContent !== d.label) lab.textContent = d.label;
    updateSection(id);
    if (redraw) drawGraph(id);
    checkAutoStop(st);
  }

  // Auto-stop: recording stops ONLY when EVERY person in view is, right now, above the quality limit with all values present.
  // (No time cap and no "once good, always good": a person who was good for one run and then dropped keeps the camera on.)
  function checkAutoStop(st) {
    if (mode !== 'multi' || !socket || !socket.connected || window.PULSEREAD_CHALLENGE_BUSY) return;
    if (!window.PULSEREAD_IS_COMPLETE || !window.PULSEREAD_AUTOSTOP) return;
    if (!people.length) return;
    const doneN = people.filter(p => pp[p.id] && window.PULSEREAD_IS_COMPLETE(pp[p.id].data)).length;
    const all = doneN === people.length;
    if (!all) return;
    const n = people.length;
    const msg = `Done · all values captured for ${n} ${n === 1 ? 'person' : 'people'}`;
    if (window.PULSEREAD_AUTOSTOP(msg)) {
      status.className = 'pp-status';
      status.innerHTML = `✅ ${esc(msg.replace(/^Done · /, ''))}. Recording stopped — each person’s results stay on screen below.` +
        '<small>Press <b>Start Webcam</b> to measure again.</small>';
    }
  }

  // what the tiles show: the full result if we have it, else the short summary from people_update
  function dataOf(id) {
    const st = pp[id];
    if (st && st.data) return st.data;
    return people.find(p => p.id === id) || {};
  }

  function setText(st, k, v) { const e = st.q(k); if (e && e.textContent !== String(v)) e.textContent = v; }
  function setHtml(st, k, h) { const e = st.q(k); if (e && e.innerHTML !== h) e.innerHTML = h; }

  function chipHtml(key, value) {
    const r = RR();
    if (!r || value == null) return '';
    const a = r.assess(key, value);
    if (!a) return '';
    return `<span class="ppb-chip" style="background:${KIND[a.kind] || '#9ca3af'}">${esc(a.status)}</span>Typical: ${esc(r.RANGES[key].typical)}`;
  }

  function updateSection(id) {
    const st = pp[id];
    if (!st) return;
    const d = dataOf(id);
    const ok = usable(d);
    const q = d.signal_quality_pct;
    const weak = d.heart_rate_bpm != null && !ok;
    const r = RR();

    // heart rate
    setText(st, 'bpm', ok ? d.heart_rate_bpm : '--');
    const a = ok && r ? r.assess('bpm', d.heart_rate_bpm) : null;
    setText(st, 'bpmSub', ok ? ((a ? a.status : '') + (d.bpm_reliability ? ' · ' + d.bpm_reliability + ' confidence' : ''))
                             : (weak ? 'Signal too weak' : 'Awaiting data'));
    setHtml(st, 'bpmChip', ok ? chipHtml('bpm', d.heart_rate_bpm) : '');

    // HRV
    const hrvOk = ok && d.hrv_available;
    setText(st, 'rmssd', hrvOk && d.hrv_rmssd_ms != null ? d.hrv_rmssd_ms : '--');
    setText(st, 'rmssdSub', hrvOk ? (d.hrv_confidence === 'Low' ? 'Rough estimate · signal only fair' : 'Good') : (d.hrv_note || 'Awaiting data'));
    setHtml(st, 'rmssdChip', hrvOk ? chipHtml('rmssd', d.hrv_rmssd_ms) : '');
    setText(st, 'sdnn', hrvOk && d.hrv_sdnn_ms != null ? d.hrv_sdnn_ms : '--');
    setText(st, 'sdnnSub', hrvOk ? 'Overall spread of beat gaps' : 'Awaiting data');
    setHtml(st, 'sdnnChip', hrvOk ? chipHtml('sdnn', d.hrv_sdnn_ms) : '');

    // stress
    const level = hrvOk ? d.stress_level : null;
    setText(st, 'stress', level || '--');
    const sv = st.q('stress');
    if (sv) sv.style.color = level && fn('stressColor') ? window.stressColor(level) : '';
    setText(st, 'stressSub', level && fn('stressDescriptor') ? window.stressDescriptor(level) +
      (d.hrv_confidence === 'Low' ? ' · rough estimate' : '') : 'Awaiting data');
    setHtml(st, 'stressChip', 'Very Low · Low · Moderate · High');

    // signal quality (always shown, even when too weak to trust the numbers)
    setText(st, 'quality', q != null ? q : '--');
    setText(st, 'qualitySub', q == null ? 'Awaiting data' : (q >= 75 ? 'Good' : (q >= 50 ? 'Fair' : 'Poor')));
    setHtml(st, 'qualityChip', q != null ? chipHtml('quality', q) : '');

    // breathing
    const rs = d.resp || null;
    setText(st, 'resp', ok && d.breathing_rate_per_min != null ? d.breathing_rate_per_min : '--');
    setText(st, 'respSub', ok && d.breathing_rate_per_min != null
      ? (rs && rs.confidence ? rs.confidence + ' confidence' : (d.breathing_source || ''))
      : (rs && rs.reason ? rs.reason : 'Needs about 30 s of clean signal'));
    setHtml(st, 'respChip', ok && d.breathing_rate_per_min != null ? chipHtml('resp', d.breathing_rate_per_min) : '');

    // note + buffer + PDF button + guidance
    setText(st, 'note', weak ? `Signal too weak (${q}%) — face the light and hold still.` : (d.hrv_note || ''));
    setText(st, 'buf', d.buffer_sec != null ? `buffer ${Number(d.buffer_sec).toFixed(0)}s / ${d.hrv_min_buffer_sec || 20}s` : '');
    const dl = st.q('dl');
    if (dl) dl.disabled = !ok;
    const rec = ok ? d.recommendation : null;
    setHtml(st, 'guid', rec
      ? `<p><b>${esc(rec.state_label)}:</b> ${esc(rec.nervous_system_note)}</p>
         <p><b>Breathing.</b> ${esc(rec.breathing)}</p><p><b>Exercise.</b> ${esc(rec.exercise)}</p>
         <p><b>Sleep.</b> ${esc(rec.sleep)}</p><p><b>Nutrition.</b> ${esc(rec.nutrition)}</p>
         <p style="color:var(--muted,#8a8fa3)">${esc(rec.disclaimer)}</p>`
      : '<p>Appears once a reliable reading is available.</p>');
  }

  // ---------------------------------------------------------------- per-person graphs
  function drawGraph(id) {
    const st = pp[id];
    if (!st) return;
    const canvas = st.el.querySelector('canvas');
    if (st.chart) { st.chart.destroy(); st.chart = null; }
    const empty = (t) => { const f = fn('drawEmptyMessage'); if (f) f(canvas, t); };
    if (typeof Chart === 'undefined') { empty('Chart library did not load. Check the Chart.js <script> tag in dashboard.html.'); return; }
    if (!fn('baseOptions') || !fn('drawRppgChart') || !fn('drawBeatHrChart')) { empty('dashboard.js is out of date — copy the newest js/dashboard.js and hard-refresh (Ctrl+F5).'); return; }
    const d = dataOf(id);
    const tab = st.tab;

    if (tab === 'resp') { st.chart = respChart(canvas, d, empty); return; }
    if (tab === 'trend') { st.chart = trendChart(canvas, st, empty); return; }

    const gr = st.graph;
    if (!gr || !gr.bvp || !gr.bvp.length) { empty('Waiting for the first model result (about 20 seconds of video)…'); return; }
    if (tab !== 'rppg' && (gr.n_valid_ibi || 0) < 3) { empty('Not enough clean heartbeats were detected to draw this graph. Face the light and hold still.'); return; }
    if (tab === 'rppg') st.chart = window.drawRppgChart(canvas, gr, `Latest window · ${gr.n_beats} beats detected`);
    else if (tab === 'hr') st.chart = window.drawBeatHrChart(canvas, gr, usable(d) ? d.heart_rate_bpm : null);
    else st.chart = tachogram(canvas, gr, d);
  }

  function tachogram(canvas, gr, d) {
    const pts = gr.beat_t.map((t, i) => ({ x: t, y: gr.ibi_ms[i] }));
    const valid = window.validNums(gr.ibi_ms);
    const m = window.meanOf(valid);
    const x0 = gr.beat_t[0], x1 = gr.beat_t[gr.beat_t.length - 1];
    const r = window.niceRange(valid, 300, 300, 1600, 50);
    const hrvText = d && d.hrv_available && d.hrv_rmssd_ms != null
      ? `RMSSD ${d.hrv_rmssd_ms} ms · SDNN ${d.hrv_sdnn_ms} ms` : 'HRV not reliable for this window';
    return new Chart(canvas, {
      type: 'line',
      data: { datasets: [
        { label: 'Beat-to-beat interval', data: pts, borderColor: '#7c5cfc', backgroundColor: '#7c5cfc', borderWidth: 2, pointRadius: 3, tension: 0, spanGaps: false, order: 1 },
        { label: `Mean ${Math.round(m)} ms`, data: [{ x: x0, y: m }, { x: x1, y: m }], borderColor: '#6b7280', borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0, order: 2 },
      ] },
      options: window.baseOptions({ xTitle: 'Time (s)', yTitle: 'Interval between beats (ms)', legend: true,
        title: `Beat-to-beat intervals · ${hrvText}`, y: { min: r.min, max: r.max } }),
    });
  }

  function respChart(canvas, d, empty) {
    const r = d && d.resp;
    if (!r || !r.wave || !r.wave.t || !r.wave.t.length) {
      empty(r && r.reason ? `Breathing: ${r.reason}.` : 'Breathing needs at least 25-30 seconds of clean signal. Keep the camera running.');
      return null;
    }
    const t0 = r.wave.t[0];
    const pts = r.wave.t.map((t, i) => ({ x: t - t0, y: r.wave.y[i] }));
    const m = r.methods || {};
    const used = ['FM', 'AM', 'BW'].filter(k => m[k] != null).map(k => `${k} ${m[k]}`).join(' · ');
    return new Chart(canvas, {
      type: 'line',
      data: { datasets: [{ label: 'Breathing waveform', data: pts, borderColor: '#06b6d4', borderWidth: 2, pointRadius: 0, tension: 0.35 }] },
      options: window.baseOptions({ xTitle: 'Time (s)', yTitle: 'Breathing (normalised)', legend: false,
        title: `Breathing about ${r.rate_bpm != null ? r.rate_bpm : '--'}/min (${r.confidence}) · per method: ${used || 'n/a'}`,
        y: { suggestedMin: -3, suggestedMax: 3 } }),
    });
  }

  function trendChart(canvas, st, empty) {
    const pts = st.series.filter(p => p.rmssd != null || p.sdnn != null);
    if (!pts.length) { empty('HRV points appear here as each analysis finishes (needs signal quality of at least 50% and about 20 s of video).'); return null; }
    return new Chart(canvas, {
      type: 'line',
      data: { labels: pts.map(p => `${p.t}s`), datasets: [
        { label: 'RMSSD', data: pts.map(p => p.rmssd), borderColor: '#7c5cfc', backgroundColor: '#7c5cfc', borderWidth: 2, pointRadius: 4, cubicInterpolationMode: 'monotone', spanGaps: true },
        { label: 'SDNN', data: pts.map(p => p.sdnn), borderColor: '#06b6d4', backgroundColor: '#06b6d4', borderWidth: 2, pointRadius: 4, cubicInterpolationMode: 'monotone', spanGaps: true },
      ] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { position: 'top', align: 'end', labels: { boxWidth: 10, boxHeight: 10, usePointStyle: true, font: { size: 11 } } } },
        scales: {
          x: { grid: { display: false }, ticks: { maxRotation: 40, autoSkip: true, maxTicksLimit: 6, font: { size: 10 } } },
          y: { beginAtZero: true, grace: '10%', title: { display: true, text: 'ms' }, grid: { color: 'rgba(0,0,0,0.06)' } },
        },
      },
    });
  }

  // ---------------------------------------------------------------- per-person PDF
  async function downloadPdf(id) {
    const st = pp[id];
    const d = st && st.data;
    if (!usable(d)) return;
    const btn = st.q('dl');
    const old = btn.textContent;
    btn.disabled = true; btn.textContent = 'Generating…';
    try {
      const rep = window.PULSEREAD_CHALLENGE_REPORT;
      const mine = rep && Array.isArray(rep.people) ? rep.people.filter(p => p.id === id) : [];
      const body = {
        status: 'ok', heart_rate_bpm: d.heart_rate_bpm, bpm_reliability: d.bpm_reliability,
        hrv_rmssd_ms: d.hrv_rmssd_ms, hrv_sdnn_ms: d.hrv_sdnn_ms, mean_ibi_ms: d.mean_ibi_ms,
        hrv_available: d.hrv_available, hrv_confidence: d.hrv_confidence, signal_quality_pct: d.signal_quality_pct,
        stress_level: d.stress_level, breathing_rate_per_min: d.breathing_rate_per_min, breathing_source: d.breathing_source,
        recommendation: d.recommendation,
        challenge: mine.length ? { ...rep, people: mine } : undefined,
      };
      const base = (window.PULSEREAD_CONFIG && window.PULSEREAD_CONFIG.API_BASE) || '';
      const res = await fetch(`${base}/api/generate-report`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      if (!res.ok) throw new Error('report generation failed');
      const url = URL.createObjectURL(await res.blob());
      const a = document.createElement('a');
      a.href = url; a.download = `pulseread_report_person_${id}.pdf`; a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert(`Could not generate the report: ${err.message}`);
    } finally {
      btn.textContent = old;
      btn.disabled = !usable(st.data);
    }
  }

  // ---------------------------------------------------------------- overlay drawing
  // The <video> is CSS-mirrored (scaleX(-1)) and uses object-fit: cover INSIDE its own element box,
  // so: take the video element's real rectangle, apply "cover" scaling inside it, then flip left-right.
  function geometry() {
    const sr = stage.getBoundingClientRect(), vr = video.getBoundingClientRect();
    const ew = vr.width, eh = vr.height;
    const vw = video.videoWidth || 640, vh = video.videoHeight || 480;
    const scale = Math.max(ew / vw, eh / vh) || 1;
    const dw = vw * scale, dh = vh * scale;
    return { ex: vr.left - sr.left, ey: vr.top - sr.top, ew, eh, dw, dh, ox: (ew - dw) / 2, oy: (eh - dh) / 2 };
  }

  function toScreen(b, g) {
    const x = g.ox + b.x * g.dw, y = g.oy + b.y * g.dh, w = b.w * g.dw, h = b.h * g.dh;
    return { x: g.ex + (g.ew - x - w), y: g.ey + y, w, h };      // mirror inside the video box
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const w = stage.clientWidth, h = stage.clientHeight;
    if (overlay.width !== Math.round(w * dpr) || overlay.height !== Math.round(h * dpr)) {
      overlay.width = Math.round(w * dpr); overlay.height = Math.round(h * dpr);
    }
    const ctx = overlay.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (mode !== 'multi' || !people.length) return;
    const g = geometry();
    people.forEach(p => {
      const v = view[p.id];
      if (!v) return;
      const r = toScreen(v, g);
      const color = colorOf(p.id);
      ctx.lineWidth = p.selected ? 4 : 2.5;
      ctx.strokeStyle = color;
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      const tail = usable(p) ? ` · ${p.heart_rate_bpm} BPM` : (p.heart_rate_bpm != null ? ' · weak signal' :
        (p.state === 'analysing' ? ' · analysing…' : ' · collecting…'));
      const label = `${String(p.label)}${tail}`;
      ctx.font = '700 12px Inter, sans-serif';
      const tw = ctx.measureText(label).width + 14;
      const ly = r.y > 24 ? r.y - 22 : r.y + r.h + 2;
      ctx.fillStyle = color;
      ctx.fillRect(r.x, ly, tw, 20);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, r.x + 7, ly + 14);
    });
  }

  function loop() {
    let moving = false;
    people.forEach(p => {
      if (!p.box) return;
      const v = view[p.id] || (view[p.id] = { ...p.box });
      ['x', 'y', 'w', 'h'].forEach(k => {
        const d = p.box[k] - v[k];
        if (Math.abs(d) > 0.0005) { v[k] += d * 0.35; moving = true; } else { v[k] = p.box[k]; }
      });
    });
    Object.keys(view).forEach(id => { if (!people.some(p => String(p.id) === id)) delete view[id]; });
    draw();
    if (moving && mode === 'multi') requestAnimationFrame(loop); else rafOn = false;
  }
  function startLoop() { if (!rafOn) { rafOn = true; requestAnimationFrame(loop); } }

  overlay.addEventListener('click', (e) => {
    if (mode !== 'multi') return;
    const rect = overlay.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const g = geometry();
    const hit = people.find(p => {
      const v = view[p.id]; if (!v) return false;
      const r = toScreen(v, g);
      return px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h;
    });
    if (hit) focusPerson(hit.id);
  });
  window.addEventListener('resize', draw);

  window.PULSEREAD_PEOPLE_VERSION = 3;
  setMode('single');
})();
