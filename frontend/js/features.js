/* PulseRead v2 — makes the three new features discoverable.
   * the three chips at the top of the dashboard jump straight to the feature
   * dashboard.html?feature=breathing | challenge | multi  does the same (used by index.html)
   Loaded LAST (after challenge.js and people.js). */
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);

  // Loud warning if the browser is running an OLD dashboard.js (the Breathing tab would then
  // show the wrong graph and the Breathing number would stay "--").
  if (!window.PULSEREAD_JS_VERSION || !window.PULSEREAD_WEBCAM_VERSION) {
    const bar = document.createElement('div');
    bar.style.cssText = 'background:#fee2e2;color:#991b1b;padding:10px 16px;font:600 13px Inter,sans-serif;text-align:center;';
    bar.textContent = 'Old dashboard.js or webcam.js detected — copy BOTH new files (js/dashboard.js and js/webcam.js), then hard-refresh (Ctrl+F5).';
    document.body.insertBefore(bar, document.body.firstChild);
  }

  function flash(el) {
    if (!el) return;
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    el.classList.remove('v2-flash'); void el.offsetWidth; el.classList.add('v2-flash');
  }

  function open(feature) {
    if (feature === 'breathing') {
      if ($('tabUpload')) $('tabUpload').click();               // breathing works for both; upload is the quickest to try
      const btn = document.querySelector('.tabs-mini button[data-graph="resp"]');
      if (btn) btn.click();
      flash(document.getElementById('signalCanvas'));
    } else if (feature === 'challenge') {
      if ($('tabWebcam')) $('tabWebcam').click();
      flash(document.querySelector('.ch-panel'));
    } else if (feature === 'multi') {
      if ($('tabMulti')) $('tabMulti').click();
      flash(document.getElementById('webcamStage'));
    }
  }

  document.querySelectorAll('.v2-chip').forEach(c => c.addEventListener('click', () => open(c.dataset.feature)));

  const wanted = new URLSearchParams(location.search).get('feature');
  if (wanted) setTimeout(() => open(wanted), 300);
})();
