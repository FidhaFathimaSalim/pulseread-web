/* ==========================================================================
   PulseRead — Animated Signal Waves
   Renders beautiful rPPG / ECG waveforms on canvas elements.
   ========================================================================== */

(function () {
  'use strict';

  /* ---- Utility ---- */
  function resizeCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    if (canvas.width !== Math.round(rect.width) || canvas.height !== Math.round(rect.height)) {
      canvas.width = Math.round(rect.width);
      canvas.height = Math.round(rect.height);
    }
  }

  /* ---- Realistic PPG pulse shape (one beat) ----
     Uses a sum-of-Gaussians to approximate systolic peak + dicrotic notch */
  function ppgBeat(t) {
    // t in [0,1], returns amplitude in [-0.2, 1]
    const sys  = Math.exp(-0.5 * ((t - 0.18) / 0.07) ** 2);          // systolic peak
    const dic  = 0.28 * Math.exp(-0.5 * ((t - 0.40) / 0.06) ** 2);   // dicrotic notch shoulder
    const dia  = 0.18 * Math.exp(-0.5 * ((t - 0.55) / 0.09) ** 2);   // diastolic peak
    return sys + dic + dia - 0.04;
  }

  /* ---- HRV waveform: irregular PPG intervals ---- */
  function hervBeat(t) {
    const beat = Math.exp(-0.5 * ((t - 0.20) / 0.09) ** 2);
    const tail = 0.22 * Math.exp(-0.5 * ((t - 0.55) / 0.12) ** 2);
    return beat + tail - 0.05;
  }

  /* ---- Stress (low-freq rolling wave) ---- */
  function stressBeat(t) {
    return 0.5 + 0.45 * Math.sin(t * Math.PI * 2);
  }

  /* ================================================================
     GENERIC SCROLLING WAVEFORM RENDERER
     ================================================================ */
  class WaveRenderer {
    constructor({ canvas, color, glowColor, beatFn, bpm = 72, amplitude, gridColor }) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.color = color;
      this.glowColor = glowColor || color;
      this.beatFn = beatFn;
      this.bpm = bpm;
      this.amplitude = amplitude || 0.38;
      this.gridColor = gridColor || 'rgba(255,255,255,0.04)';

      this.t = 0;          // global time in seconds
      this.data = [];      // rolling buffer of y-values (pixels)
      this.raf = null;
      this.lastTs = null;

      this._resize();
      this._startLoop();
    }

    _resize() {
      resizeCanvas(this.canvas);
      this.data = new Array(this.canvas.width).fill(this.canvas.height / 2);
    }

    /* Generate the next y-value based on time */
    _sample(t) {
      const beatPeriod = 60 / this.bpm;          // seconds per beat
      const phase = (t % beatPeriod) / beatPeriod; // 0..1 within beat
      const raw = this.beatFn(phase);              // -0.05 .. 1
      const h = this.canvas.height;
      const mid = h * 0.5;
      // Map raw [0..1] to pixels; flip so peaks go up
      return mid - raw * h * this.amplitude;
    }

    _drawGrid() {
      const ctx = this.ctx;
      const { width: w, height: h } = this.canvas;
      ctx.strokeStyle = this.gridColor;
      ctx.lineWidth = 1;

      // Horizontal lines
      const hLines = 4;
      for (let i = 0; i <= hLines; i++) {
        const y = (h / hLines) * i;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }
      // Vertical lines
      const vLines = 12;
      for (let i = 0; i <= vLines; i++) {
        const x = (w / vLines) * i;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
    }

    _drawWave() {
      const ctx = this.ctx;
      const { width: w, height: h } = this.canvas;
      const data = this.data;

      // Glow pass
      ctx.save();
      ctx.shadowBlur = 14;
      ctx.shadowColor = this.glowColor;
      ctx.strokeStyle = this.glowColor;
      ctx.lineWidth = 3;
      ctx.globalAlpha = 0.35;
      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {
        if (i === 0) ctx.moveTo(i, data[i]);
        else ctx.lineTo(i, data[i]);
      }
      ctx.stroke();
      ctx.restore();

      // Crisp line pass
      ctx.save();
      ctx.strokeStyle = this.color;
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {
        if (i === 0) ctx.moveTo(i, data[i]);
        else ctx.lineTo(i, data[i]);
      }
      ctx.stroke();

      // Gradient fill under the wave
      const grad = ctx.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, this.glowColor.replace(')', ', 0.18)').replace('rgb', 'rgba').replace('#', 'rgba(').replace('rgba(', 'rgba('));
      // Simpler approach:
      grad.addColorStop(0, 'rgba(255,255,255,0.06)');
      grad.addColorStop(1, 'rgba(255,255,255,0.00)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(0, h);
      for (let i = 0; i < data.length; i++) ctx.lineTo(i, data[i]);
      ctx.lineTo(w, h);
      ctx.closePath();
      ctx.fill();
      ctx.restore();

      // Moving dot at the leading edge
      const last = data[data.length - 1];
      ctx.save();
      ctx.fillStyle = this.color;
      ctx.shadowBlur = 18;
      ctx.shadowColor = this.glowColor;
      ctx.beginPath();
      ctx.arc(data.length - 1, last, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    _tick(ts) {
      if (this.lastTs === null) this.lastTs = ts;
      const dt = Math.min((ts - this.lastTs) / 1000, 0.05); // seconds elapsed, capped
      this.lastTs = ts;
      this.t += dt;

      resizeCanvas(this.canvas);
      if (this.data.length !== this.canvas.width) {
        this.data = new Array(this.canvas.width).fill(this.canvas.height / 2);
      }

      // Push new sample, shift buffer
      const y = this._sample(this.t);
      this.data.push(y);
      if (this.data.length > this.canvas.width) this.data.shift();

      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this._drawGrid();
      this._drawWave();

      this.raf = requestAnimationFrame(this.raf ? this._tick.bind(this) : this._tick.bind(this));
    }

    _startLoop() {
      this.raf = requestAnimationFrame((ts) => {
        this.lastTs = ts;
        const loop = (ts2) => {
          this._tick(ts2);
          this.raf = requestAnimationFrame(loop);
        };
        this.raf = requestAnimationFrame(loop);
      });
    }

    destroy() {
      if (this.raf) cancelAnimationFrame(this.raf);
    }
  }

  /* ================================================================
     HERO MINI WAVE (bottom of the hero card)
     ================================================================ */
  function initHeroWave() {
    const canvas = document.getElementById('heroWaveCanvas');
    if (!canvas) return;

    let t = 0;
    let lastTs = null;
    const bpm = 72;

    function draw(ts) {
      if (lastTs === null) lastTs = ts;
      const dt = Math.min((ts - lastTs) / 1000, 0.05);
      lastTs = ts;
      t += dt;

      resizeCanvas(canvas);
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const beatPeriod = 60 / bpm;
      const points = [];
      const steps = w;

      // Build one full visible sweep of ~3 heartbeats
      const totalDuration = 3; // seconds of waveform shown
      for (let i = 0; i < steps; i++) {
        const tSample = t - totalDuration + (i / steps) * totalDuration;
        const phase = ((tSample % beatPeriod) + beatPeriod) % beatPeriod / beatPeriod;
        const raw = ppgBeat(phase);
        const y = h * 0.5 - raw * h * 0.42;
        points.push({ x: i, y });
      }

      // Subtle grid
      ctx.strokeStyle = 'rgba(126,247,193,0.1)';
      ctx.lineWidth = 1;
      for (let i = 0; i < 5; i++) {
        const y = (h / 4) * i;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }

      // Glow
      ctx.save();
      ctx.shadowBlur = 16;
      ctx.shadowColor = '#22d3ee';
      ctx.strokeStyle = 'rgba(34,211,238,0.5)';
      ctx.lineWidth = 3;
      ctx.beginPath();
      points.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
      ctx.stroke();
      ctx.restore();

      // Main line
      ctx.save();
      ctx.strokeStyle = '#22d3ee';
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      points.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
      ctx.stroke();

      // Fill
      const grad = ctx.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, 'rgba(34,211,238,0.18)');
      grad.addColorStop(1, 'rgba(34,211,238,0.00)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(0, h);
      points.forEach(p => ctx.lineTo(p.x, p.y));
      ctx.lineTo(w, h);
      ctx.closePath();
      ctx.fill();
      ctx.restore();

      // Moving dot
      const last = points[points.length - 1];
      ctx.save();
      ctx.fillStyle = '#7ef7c1';
      ctx.shadowBlur = 20;
      ctx.shadowColor = '#22d3ee';
      ctx.beginPath();
      ctx.arc(last.x, last.y, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);

    // Animate the BPM display with slight variation
    let displayBpm = 72;
    setInterval(() => {
      displayBpm = 70 + Math.round(Math.random() * 6);
      const el = document.getElementById('heroHr');
      if (el) el.textContent = displayBpm;
    }, 2800);
  }

  /* ================================================================
     MAIN SIGNAL SECTION (3 channels)
     ================================================================ */
  function initSignalSection() {
    const configs = [
      {
        id: 'signalCanvas1',
        color: '#ec4899',
        glowColor: '#f472b6',
        beatFn: ppgBeat,
        bpm: 72,
        amplitude: 0.40,
        gridColor: 'rgba(236,72,153,0.07)'
      },
      {
        id: 'signalCanvas2',
        color: '#22d3ee',
        glowColor: '#67e8f9',
        beatFn: hervBeat,
        bpm: 68,
        amplitude: 0.35,
        gridColor: 'rgba(34,211,238,0.07)'
      },
      {
        id: 'signalCanvas3',
        color: '#a78bfa',
        glowColor: '#c4b5fd',
        beatFn: stressBeat,
        bpm: 18,
        amplitude: 0.30,
        gridColor: 'rgba(167,139,250,0.07)'
      }
    ];

    configs.forEach(cfg => {
      const canvas = document.getElementById(cfg.id);
      if (!canvas) return;
      initChannelWave(canvas, cfg);
    });
  }

  function initChannelWave(canvas, cfg) {
    let t = 0;
    let lastTs = null;
    const data = [];

    function sample(ts2) {
      const beatPeriod = 60 / cfg.bpm;
      const phase = ((ts2 % beatPeriod) + beatPeriod) % beatPeriod / beatPeriod;
      const raw = cfg.beatFn(phase);
      const h = canvas.height;
      return h / 2 - raw * h * cfg.amplitude;
    }

    function draw(ts) {
      if (lastTs === null) lastTs = ts;
      const dt = Math.min((ts - lastTs) / 1000, 0.05);
      lastTs = ts;
      t += dt;

      resizeCanvas(canvas);
      const w = canvas.width, h = canvas.height;
      const ctx = canvas.getContext('2d');

      if (data.length !== w) {
        data.length = 0;
        for (let i = 0; i < w; i++) data.push(h / 2);
      }

      data.push(sample(t));
      if (data.length > w) data.shift();

      ctx.clearRect(0, 0, w, h);

      // Grid
      ctx.strokeStyle = cfg.gridColor;
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = (h / 4) * i;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }
      for (let i = 0; i <= 10; i++) {
        const x = (w / 10) * i;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }

      // Glow pass
      ctx.save();
      ctx.shadowBlur = 16;
      ctx.shadowColor = cfg.glowColor;
      ctx.strokeStyle = cfg.color + '88';
      ctx.lineWidth = 4;
      ctx.globalAlpha = 0.4;
      ctx.beginPath();
      data.forEach((y, i) => i === 0 ? ctx.moveTo(i, y) : ctx.lineTo(i, y));
      ctx.stroke();
      ctx.restore();

      // Sharp wave
      ctx.save();
      ctx.strokeStyle = cfg.color;
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      data.forEach((y, i) => i === 0 ? ctx.moveTo(i, y) : ctx.lineTo(i, y));
      ctx.stroke();

      // Gradient fill
      const grad = ctx.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, cfg.color + '22');
      grad.addColorStop(1, cfg.color + '00');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(0, h);
      data.forEach((y, i) => ctx.lineTo(i, y));
      ctx.lineTo(w, h);
      ctx.closePath();
      ctx.fill();
      ctx.restore();

      // Leading dot
      const lastY = data[data.length - 1];
      ctx.save();
      ctx.fillStyle = cfg.color;
      ctx.shadowBlur = 20;
      ctx.shadowColor = cfg.glowColor;
      ctx.beginPath();
      ctx.arc(data.length - 1, lastY, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      requestAnimationFrame(draw);
    }

    requestAnimationFrame(draw);
  }

  /* ================================================================
     ANIMATED STATS COUNTER (slight live variation)
     ================================================================ */
  function initStatsAnimation() {
    const targets = {
      demoHr:     { base: 72, delta: 3, dec: 0 },
      demoSdnn:   { base: 48, delta: 5, dec: 0 },
      demoStress: { base: null, labels: ['Low', 'Low', 'Moderate', 'Low'], idx: 0 },
      demoSpo2:   { base: 98, delta: 1, dec: 0, suffix: '%' }
    };

    setInterval(() => {
      Object.entries(targets).forEach(([id, cfg]) => {
        const el = document.getElementById(id);
        if (!el) return;
        if (cfg.labels) {
          cfg.idx = (cfg.idx + 1) % cfg.labels.length;
          el.textContent = cfg.labels[cfg.idx];
        } else {
          const val = cfg.base + Math.round((Math.random() * 2 - 1) * cfg.delta);
          el.textContent = val.toFixed(cfg.dec) + (cfg.suffix || '');
        }
      });
    }, 2400);
  }

  /* ================================================================
     BOOT
     ================================================================ */
  function boot() {
    initHeroWave();
    initSignalSection();
    initStatsAnimation();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

})();
