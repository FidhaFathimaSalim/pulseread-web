/* ==========================================================================
   PulseRead — "is my number high or low?"
   Adds, under every result:  a coloured range bar with a dot for YOUR value,
   a status label (Typical / Low / High ...) and the typical range in plain words.
   Also adds a "What do these numbers mean?" help card.
   The numbers are a copy of backend/reference_ranges.py — change both together.
   HRV bands are this app's own simple guide, NOT clinical limits.
   ========================================================================== */
(function () {
  'use strict';

  const COLORS = { ok: '#16a34a', good: '#0d9488', warn: '#d97706', bad: '#dc2626', low: '#2563eb', high: '#ea580c' };

  const RANGES = {
    bpm: {
      label: 'Heart rate', unit: 'BPM', typical: '60–100 BPM', scale: [40, 140],
      meaning: 'How many times your heart beats in one minute, at rest.',
      bands: [[null, 60, 'Below typical', 'low'], [60, 100, 'Typical', 'ok'], [100, null, 'Above typical', 'high']],
      note: 'Very fit people and athletes can normally be below 60.',
    },
    rmssd: {
      label: 'HRV (RMSSD)', unit: 'ms', typical: '35–70 ms', scale: [0, 120],
      meaning: 'How much the time between heartbeats changes from beat to beat. Higher usually means a more relaxed, recovered body.',
      bands: [[null, 15, 'Very low', 'bad'], [15, 35, 'Low', 'warn'], [35, 70, 'Typical', 'ok'], [70, null, 'High', 'good']],
      note: 'Depends a lot on age and fitness — compare with your own usual values. Video-based HRV is rough.',
    },
    sdnn: {
      label: 'HRV (SDNN)', unit: 'ms', typical: '40–80 ms', scale: [0, 150],
      meaning: 'The overall spread of the gaps between heartbeats during the recording.',
      bands: [[null, 15, 'Very low', 'bad'], [15, 40, 'Low', 'warn'], [40, 80, 'Typical', 'ok'], [80, null, 'High', 'good']],
      note: 'Same caution as RMSSD.',
    },
    meanrr: {
      label: 'Mean RR', unit: 'ms', typical: '600–1000 ms', scale: [400, 1300],
      meaning: 'The average time between two heartbeats (800 ms is about 75 BPM).',
      bands: [[null, 600, 'Short (fast heart)', 'high'], [600, 1000, 'Typical', 'ok'], [1000, null, 'Long (slow heart)', 'low']],
      note: '',
    },
    resp: {
      label: 'Breathing rate', unit: '/min', typical: '12–20 /min', scale: [4, 30],
      meaning: 'How many breaths you take in one minute, at rest.',
      bands: [[null, 12, 'Slower than typical', 'low'], [12, 20, 'Typical', 'ok'], [20, null, 'Faster than typical', 'high']],
      note: 'Slow paced breathing (about 6 per minute) is normal during relaxation exercises.',
    },
    quality: {
      label: 'Signal quality', unit: '%', typical: 'above 75 % is good', scale: [0, 100],
      meaning: 'How clean the pulse signal in the video was. Low quality makes every other number less reliable.',
      bands: [[null, 50, 'Poor', 'bad'], [50, 75, 'Fair', 'warn'], [75, null, 'Good', 'ok']],
      note: 'Face the light, keep your face still and fully in view.',
    },
  };

  function assess(key, value) {
    const r = RANGES[key];
    const v = Number(value);
    if (value == null || value === '' || !isFinite(v)) return null;
    for (const [lo, hi, status, kind] of r.bands) {
      if ((lo == null || v >= lo) && (hi == null || v < hi)) return { status, kind };
    }
    return null;
  }

  // ---------------------------------------------------------------- styles
  const style = document.createElement('style');
  style.textContent = `
    .rng{margin-top:10px;}
    .rng-bar{position:relative;display:flex;height:7px;border-radius:6px;overflow:visible;background:#e5e7eb;}
    .rng-bar b{display:block;height:100%;opacity:.55;}
    .rng-bar b:first-child{border-radius:6px 0 0 6px;} .rng-bar b:last-of-type{border-radius:0 6px 6px 0;}
    .rng-bar i{position:absolute;top:-4px;width:15px;height:15px;margin-left:-7px;border-radius:50%;background:#fff;border:3px solid #0e0f1a;box-shadow:0 1px 3px rgba(0,0,0,.25);transition:left .5s ease;}
    .rng-bar i.off{display:none;}
    .rng-text{margin-top:9px;font-size:11.5px;line-height:1.45;color:var(--ink-soft,#4b5060);font-weight:500;}
    .rng-chip{display:inline-block;font-size:10.5px;font-weight:800;letter-spacing:.03em;text-transform:uppercase;color:#fff;padding:2px 8px;border-radius:999px;margin-right:6px;}
    .rng-chip.none{background:#9ca3af;}
    .rng-mini{margin-top:6px;font-size:10.5px;line-height:1.35;color:var(--ink-soft,#4b5060);font-weight:600;}
    .rng-mini .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px;vertical-align:middle;}
    .rng-card summary{cursor:pointer;font-weight:800;font-size:15px;list-style:none;display:flex;align-items:center;gap:10px;}
    .rng-card summary::-webkit-details-marker{display:none;}
    .rng-card summary:after{content:'+';margin-left:auto;font-size:20px;color:var(--purple,#7c5cfc);}
    .rng-card[open] summary:after{content:'–';}
    .rng-table{width:100%;border-collapse:collapse;font-size:13px;margin-top:14px;}
    .rng-table th{text-align:left;background:var(--bg-light,#f6f7fb);padding:8px 10px;font-weight:700;font-size:12px;color:var(--ink-soft,#4b5060);}
    .rng-table td{padding:9px 10px;border-bottom:1px solid var(--border,#e8e9f0);vertical-align:top;}
    .rng-table td:first-child{font-weight:700;white-space:nowrap;}
    .rng-legend{display:flex;flex-wrap:wrap;gap:6px;margin-top:3px;}
    .rng-legend span{font-size:11px;font-weight:700;color:#fff;padding:2px 8px;border-radius:999px;}
    .rng-fine{font-size:12px;color:var(--muted,#8a8fa3);margin-top:12px;line-height:1.55;}
    .rng-scroll{overflow-x:auto;}
  `;
  document.head.appendChild(style);

  // ---------------------------------------------------------------- tile hints
  function barHtml(r) {
    const [min, max] = r.scale;
    const span = max - min;
    const segs = r.bands.map(([lo, hi, , kind]) => {
      const a = Math.max(lo == null ? min : lo, min);
      const b = Math.min(hi == null ? max : hi, max);
      return `<b style="width:${Math.max(0, (b - a) / span * 100)}%;background:${COLORS[kind]}"></b>`;
    }).join('');
    return `<div class="rng-bar">${segs}<i class="off"></i></div>`;
  }

  function addHint(afterEl, key) {
    if (!afterEl || document.getElementById('rng-' + key)) return;
    const r = RANGES[key];
    const wrap = document.createElement('div');
    wrap.className = 'rng';
    wrap.id = 'rng-' + key;
    wrap.innerHTML = `${barHtml(r)}<div class="rng-text"><span class="rng-chip none">no data</span><span class="rng-typ">Typical: ${r.typical}</span></div>`;
    afterEl.parentNode.insertBefore(wrap, afterEl.nextSibling);
  }

  function addMini(valueEl, key) {
    if (!valueEl || document.getElementById('rngm-' + key)) return;
    const d = document.createElement('div');
    d.className = 'rng-mini';
    d.id = 'rngm-' + key;
    d.innerHTML = `<span style="font-weight:500;color:var(--muted,#8a8fa3)">typical ${RANGES[key].typical}</span>`;
    valueEl.parentNode.appendChild(d);
  }

  function setHint(key, value, extra) {
    const wrap = document.getElementById('rng-' + key);
    if (!wrap) return;
    const r = RANGES[key];
    const a = assess(key, value);
    const chip = wrap.querySelector('.rng-chip');
    const dot = wrap.querySelector('.rng-bar i');
    if (!a) {
      chip.textContent = 'no data'; chip.className = 'rng-chip none'; chip.style.background = '';
      dot.classList.add('off');
    } else {
      chip.textContent = a.status; chip.className = 'rng-chip'; chip.style.background = COLORS[a.kind];
      const [min, max] = r.scale;
      const pct = Math.max(0, Math.min(100, (Number(value) - min) / (max - min) * 100));
      dot.style.left = pct + '%';
      dot.classList.remove('off');
    }
    wrap.querySelector('.rng-typ').textContent = 'Typical: ' + r.typical + (extra ? ' · ' + extra : '');
  }

  function setMini(key, value) {
    const el = document.getElementById('rngm-' + key);
    if (!el) return;
    const a = assess(key, value);
    const typ = `<span style="font-weight:500;color:var(--muted,#8a8fa3)">typical ${RANGES[key].typical}</span>`;
    el.innerHTML = a
      ? `<span class="dot" style="background:${COLORS[a.kind]}"></span>${a.status.split(' (')[0]}<br>${typ}`
      : typ;
  }

  // ---------------------------------------------------------------- help card
  function buildHelpCard() {
    if (document.getElementById('rngHelp')) return;
    const anchor = document.querySelector('.insights-card');
    if (!anchor) return;
    const rows = ['bpm', 'rmssd', 'sdnn', 'meanrr', 'resp', 'quality'].map(k => {
      const r = RANGES[k];
      const legend = r.bands.map(([lo, hi, status, kind]) => {
        const rng = lo == null ? `below ${hi}` : (hi == null ? `${lo}+` : `${lo}–${hi}`);
        return `<span style="background:${COLORS[kind]}">${status} (${rng})</span>`;
      }).join('');
      return `<tr><td>${r.label}</td><td>${r.meaning}${r.note ? `<br><span style="color:var(--muted,#8a8fa3)">${r.note}</span>` : ''}</td>
              <td><div class="rng-legend">${legend}</div></td></tr>`;
    }).join('');
    const card = document.createElement('details');
    card.className = 'card rng-card';
    card.id = 'rngHelp';
    card.style.marginTop = '20px';
    card.innerHTML = `
      <summary>📖 What do these numbers mean? Typical ranges</summary>
      <div class="rng-scroll"><table class="rng-table">
        <tr><th>Measure</th><th>What it tells you</th><th>How to read it</th></tr>
        ${rows}
        <tr><td>Stress level</td><td>An estimate worked out from HRV: lower HRV gives a higher stress level.</td>
            <td><div class="rng-legend"><span style="background:#0f8a4b">Very Low</span><span style="background:#7cc152">Low</span><span style="background:#d97706">Moderate</span><span style="background:#ee1f4d">High</span></div></td></tr>
      </table></div>
      <div class="rng-fine">
        Heart rate (60–100) and breathing (12–20) are the usual textbook resting ranges for healthy adults.
        HRV has no single “normal” — it changes with age, fitness and sleep, and the HRV bands here are this app’s own simple guide,
        not clinical limits. The best comparison is <b>you today versus your own usual numbers</b>.
        This is general wellness information, not a medical diagnosis.
      </div>`;
    anchor.parentNode.insertBefore(card, anchor);
  }

  // ---------------------------------------------------------------- public update
  function update(data) {
    data = data || {};
    const lowHrv = data.hrv_available && data.hrv_confidence === 'Low';
    setHint('bpm', data.heart_rate_bpm);
    setHint('rmssd', data.hrv_available ? data.hrv_rmssd_ms : null, lowHrv ? 'rough estimate' : '');
    setHint('quality', data.signal_quality_pct);
    setMini('rmssd', data.hrv_available ? data.hrv_rmssd_ms : null);
    setMini('sdnn', data.hrv_available ? data.hrv_sdnn_ms : null);
    setMini('meanrr', data.mean_ibi_ms);
    setMini('resp', data.breathing_rate_per_min);
  }

  function init() {
    addHint(document.getElementById('subBpm'), 'bpm');
    addHint(document.getElementById('subHrv'), 'rmssd');
    addHint(document.getElementById('subQuality'), 'quality');
    addMini(document.getElementById('numRmssd'), 'rmssd');
    addMini(document.getElementById('numSdnn'), 'sdnn');
    addMini(document.getElementById('numMeanrr'), 'meanrr');
    addMini(document.getElementById('numResp'), 'resp');
    const stress = document.getElementById('subStress');
    if (stress && !document.getElementById('rng-stress')) {
      const d = document.createElement('div');
      d.id = 'rng-stress';
      d.className = 'rng-text';
      d.style.marginTop = '10px';
      d.textContent = 'Scale: Very Low · Low · Moderate · High (lower HRV → higher stress)';
      stress.parentNode.insertBefore(d, stress.nextSibling);
    }
    buildHelpCard();
    update({});
  }

  window.PULSEREAD_RANGES = { update, assess, RANGES };
  window.PULSEREAD_RANGES_VERSION = 1;
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
