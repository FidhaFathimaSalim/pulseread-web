"""
report_builder.py — the downloadable PulseRead wellness report (PDF).

Layout (inspired by a "patient card" product sheet):
  logo + brand  ->  big headline  ->  one white "profile card" (header, tabs row,
  Key Results grid, Nervous System State)  ->  four guidance cards (Breathing, Exercise,
  Sleep, Nutrition)  ->  footer with the disclaimer.
Optional extra pages: Personal Baseline and Physiological Challenge.

LOGO: uses the website's PulseRead_logo.png (looked up in ../frontend, next to this file, ./static ...).
If none is found a built-in vector logo is drawn and the console says where it looked.

General wellness information only — not a medical report.
"""

import os
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import Paragraph, Table, TableStyle

W, H = letter                      # 612 x 792 pt
MARGIN = 36

PURPLE = colors.HexColor('#7c5cfc')
PINK = colors.HexColor('#ec4899')
INK = colors.HexColor('#0e0f1a')
SOFT = colors.HexColor('#4b5060')
MUTED = colors.HexColor('#8a8fa3')
LINE = colors.HexColor('#e6e8f2')
PAGE_BG = colors.HexColor('#f4f5fc')
TINT = colors.HexColor('#efeaff')

try:
    import reference_ranges as RR
except ImportError:          # report still builds without the ranges file
    RR = None

KIND_COLORS = {'ok': '#16a34a', 'good': '#0d9488', 'warn': '#d97706', 'bad': '#dc2626', 'low': '#2563eb', 'high': '#ea580c'}

HERE = os.path.dirname(os.path.abspath(__file__))
LOGO_NAMES = ['PulseRead_logo.png', 'pulseread_logo.png', 'PulseRead_logo.jpg', 'PulseRead_logo.jpeg']
LOGO_DIRS = [
    os.path.join(HERE, '..', 'frontend'),      # <- where your website keeps it (app.py serves ../frontend)
    HERE,
    os.path.join(HERE, 'frontend'),
    os.path.join(HERE, 'static'),
    os.path.join(HERE, '..', 'frontend', 'images'),
    os.path.join(HERE, '..', 'frontend', 'img'),
    os.getcwd(),
    os.path.join(os.getcwd(), '..', 'frontend'),
]
_logo_cache = {}

DISCLAIMER = ('This report is generated from an estimated remote-PPG reading and general wellness information. '
              'It is not a diagnosis and does not replace professional medical advice.')


# ------------------------------------------------------------------ small helpers

def _txt(v, unit='', dash='--'):
    if v is None or v == '' or v == '--':
        return dash
    return f'{v}{unit}'


def _para(text, size=9, color=INK, leading=None, bold=False, align=0):
    st = ParagraphStyle('p', fontName='Helvetica-Bold' if bold else 'Helvetica', fontSize=size,
                        leading=leading or size * 1.35, textColor=color, alignment=align)
    return Paragraph(text, st)


def _draw_para(c, text, x, y_top, width, **kw):
    """Draw a wrapped paragraph with its top-left at (x, y_top). Returns its height."""
    p = _para(text, **kw)
    _, h = p.wrap(width, 10000)
    p.drawOn(c, x, y_top - h)
    return h


CUR_ALPHA = 1.0


def _alpha(c, a):
    """Set the current transparency. NOTE: reportlab colours carry their own alpha (1.0) which would
    cancel setFillAlpha, so every colour is set through _fc/_sc, which re-apply CUR_ALPHA."""
    global CUR_ALPHA
    CUR_ALPHA = a
    c.setFillAlpha(a)          # also affects images (drawImage), which have no colour of their own
    c.setStrokeAlpha(a)


def _fc(c, color):
    c.setFillColor(color, CUR_ALPHA)


def _sc(c, color):
    c.setStrokeColor(color, CUR_ALPHA)


def _round_rect(c, x, y, w, h, r, fill=None, stroke=None, lw=0.8):
    if fill is not None:
        _fc(c, fill)
    if stroke is not None:
        _sc(c, stroke)
        c.setLineWidth(lw)
    c.roundRect(x, y, w, h, r, stroke=1 if stroke is not None else 0, fill=1 if fill is not None else 0)


def _card(c, x, y, w, h, r=14):
    """White rounded card with a soft layered shadow."""
    for i, a in enumerate((0.025, 0.03, 0.035, 0.04)):
        _alpha(c, a)
        grow = 6 - i * 1.5
        _round_rect(c, x - grow, y - grow - 2, w + 2 * grow, h + 2 * grow, r + grow, fill=colors.HexColor('#3b3f6b'))
    _alpha(c, 1)
    _round_rect(c, x, y, w, h, r, fill=colors.white, stroke=LINE, lw=0.7)


# ------------------------------------------------------------------ logo + icons

def _logo_path():
    """Find the website's logo. Exact names first, then any *logo*.png/jpg in the usual folders."""
    if 'path' in _logo_cache:
        return _logo_cache['path']
    found = None
    for d in LOGO_DIRS:
        for n in LOGO_NAMES:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                found = p
                break
        if found:
            break
    if not found:
        for d in LOGO_DIRS:
            try:
                for f in sorted(os.listdir(d)):
                    if 'logo' in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        found = os.path.join(d, f)
                        break
            except OSError:
                continue
            if found:
                break
    if found:
        print(f'[report] using logo: {os.path.abspath(found)}')
    else:
        print('[report] PulseRead_logo.png not found - drawing the built-in logo. Looked in: ' +
              ', '.join(os.path.abspath(d) for d in LOGO_DIRS))
    _logo_cache['path'] = found
    return found


def _prepare_logo():
    """
    Load the website logo once and cut it into two pieces (both cropped tight, transparent background kept):
      'full' = emblem + the "PulseRead" wordmark      'mark' = the emblem only (for slim headers)
    Returns {'full': PIL.Image, 'mark': PIL.Image} or None (-> built-in vector logo is used).
    """
    if 'img' in _logo_cache:
        return _logo_cache['img']
    result = None
    path = _logo_path()
    if path:
        try:
            from PIL import Image
            import numpy as np
            im = Image.open(path).convert('RGBA')
            a = np.array(im)
            # "content" = visible pixels; a fully opaque white background counts as empty too
            opaque = a[..., 3] > 10
            nonwhite = (255 - a[..., :3].astype(int)).max(axis=2) > 14
            content = opaque & nonwhite
            rows = np.where(content.any(axis=1))[0]
            cols = np.where(content.any(axis=0))[0]
            if len(rows) and len(cols):
                pad = 6
                y0, y1 = max(rows.min() - pad, 0), min(rows.max() + pad, im.height - 1)
                x0, x1 = max(cols.min() - pad, 0), min(cols.max() + pad, im.width - 1)

                # first band of content rows = the emblem (a gap separates it from the wordmark)
                r = content.any(axis=1)
                bands, start = [], None
                for i, v in enumerate(r):
                    if v and start is None:
                        start = i
                    if not v and start is not None:
                        bands.append((start, i - 1))
                        start = None
                if start is not None:
                    bands.append((start, len(r) - 1))
                big_bands = [b for b in bands if b[1] - b[0] > 8]

                def crop(top, bottom):
                    img = im.crop((x0, top, x1 + 1, bottom + 1))
                    if img.width > 800:
                        img = img.resize((800, int(img.height * 800 / img.width)), Image.LANCZOS)
                    return img

                full = crop(y0, y1)
                mark = crop(max(big_bands[0][0] - pad, 0), big_bands[0][1] + pad) if len(big_bands) > 1 else full
                result = {'full': full, 'mark': mark}
        except Exception as e:      # noqa: BLE001 - never break the report because of the logo
            print('[report] could not process the logo, using the built-in one:', repr(e))
    _logo_cache['img'] = result
    return result


def _draw_pil(c, img, x, y_top, height):
    """Draw a PIL image with its top-left at (x, y_top), scaled to `height`. Returns the drawn width."""
    w = img.width * height / img.height
    _alpha(c, 1)
    c.drawImage(ImageReader(img), x, y_top - height, w, height, mask='auto')
    return w


def _vector_logo(c, cx, top, size):
    x, y = cx - size / 2, top - size
    _round_rect(c, x, y, size, size, size * 0.26, fill=PURPLE)
    _alpha(c, 0.35)
    _round_rect(c, x + size * 0.5, y + size * 0.5, size * 0.5, size * 0.5, size * 0.26, fill=PINK)
    _alpha(c, 1)
    _fc(c, colors.white)
    hx, hy, s = cx, y + size * 0.52, size * 0.30
    p = c.beginPath()
    p.moveTo(hx, hy - s * 0.95)
    p.curveTo(hx - s * 1.35, hy - s * 0.1, hx - s * 0.95, hy + s * 0.95, hx, hy + s * 0.35)
    p.curveTo(hx + s * 0.95, hy + s * 0.95, hx + s * 1.35, hy - s * 0.1, hx, hy - s * 0.95)
    c.drawPath(p, fill=1, stroke=0)
    _sc(c, PURPLE)
    c.setLineWidth(size * 0.045)
    c.setLineJoin(1)
    c.setLineCap(1)
    ly = hy - s * 0.05
    q = c.beginPath()
    pts = [(-0.95, 0), (-0.45, 0), (-0.25, 0.55), (0.0, -0.6), (0.25, 0.35), (0.45, 0), (0.95, 0)]
    q.moveTo(hx + pts[0][0] * s, ly + pts[0][1] * s)
    for px, py in pts[1:]:
        q.lineTo(hx + px * s, ly + py * s)
    c.drawPath(q, fill=0, stroke=1)


def _draw_logo_hero(c, cx, top, height):
    """Big centred logo (emblem + wordmark). Returns the y of its bottom edge."""
    logos = _prepare_logo()
    if logos:
        img = logos['full']
        w = img.width * height / img.height
        _draw_pil(c, img, cx - w / 2, top, height)
        return top - height
    _vector_logo(c, cx, top, 52)
    _fc(c, PURPLE)
    c.setFont('Helvetica-Bold', 17)
    c.drawCentredString(cx, top - 70, 'PulseRead')
    return top - 78


def _draw_logo_small(c, x, top, height):
    """Emblem only, left aligned (slim page headers). Returns its width."""
    logos = _prepare_logo()
    if logos:
        return _draw_pil(c, logos['mark'], x, top, height)
    _vector_logo(c, x + height / 2, top, height)
    return height


def _icon_bg(c, cx, cy, r, tint):
    _fc(c, tint)
    c.circle(cx, cy, r, stroke=0, fill=1)


def _icon_breathing(c, cx, cy):
    _sc(c, colors.HexColor('#0891b2'))
    c.setLineWidth(1.8)
    c.setLineCap(1)
    for dy, ln in ((6, 14), (0, 18), (-6, 12)):
        c.line(cx - 9, cy + dy, cx - 9 + ln, cy + dy)
    _fc(c, colors.HexColor('#0891b2'))
    c.circle(cx + 9, cy + 6, 1.6, stroke=0, fill=1)


def _icon_exercise(c, cx, cy):
    col = colors.HexColor('#dc2626')
    _sc(c, col)
    _fc(c, col)
    c.setLineWidth(2.2)
    c.line(cx - 8, cy, cx + 8, cy)
    for sx in (-1, 1):
        c.roundRect(cx + sx * 8 - 2.2, cy - 6, 4.4, 12, 1.5, stroke=0, fill=1)
        c.roundRect(cx + sx * 12.5 - 1.6, cy - 4, 3.2, 8, 1.2, stroke=0, fill=1)


def _icon_sleep(c, cx, cy):
    _fc(c, colors.HexColor('#6366f1'))
    c.circle(cx, cy, 9, stroke=0, fill=1)
    _fc(c, colors.HexColor('#e0e7ff'))
    c.circle(cx + 5, cy + 4, 8, stroke=0, fill=1)


def _icon_nutrition(c, cx, cy):
    _fc(c, colors.HexColor('#16a34a'))
    c.circle(cx - 3.5, cy - 1.5, 6.5, stroke=0, fill=1)
    c.circle(cx + 3.5, cy - 1.5, 6.5, stroke=0, fill=1)
    _fc(c, colors.HexColor('#15803d'))
    p = c.beginPath()
    p.moveTo(cx, cy + 5)
    p.curveTo(cx + 2, cy + 12, cx + 8, cy + 12, cx + 9, cy + 9)
    p.curveTo(cx + 6, cy + 5, cx + 3, cy + 5, cx, cy + 5)
    c.drawPath(p, fill=1, stroke=0)


def _icon_heart(c, cx, cy, s, color):
    _fc(c, color)
    p = c.beginPath()
    p.moveTo(cx, cy - s * 0.95)
    p.curveTo(cx - s * 1.35, cy - s * 0.1, cx - s * 0.95, cy + s * 0.95, cx, cy + s * 0.35)
    p.curveTo(cx + s * 0.95, cy + s * 0.95, cx + s * 1.35, cy - s * 0.1, cx, cy - s * 0.95)
    c.drawPath(p, fill=1, stroke=0)


def _icon_lock(c, cx, cy, color=PURPLE):
    _fc(c, color)
    c.roundRect(cx - 5, cy - 5, 10, 8, 1.5, stroke=0, fill=1)
    _sc(c, color)
    c.setLineWidth(1.4)
    c.arc(cx - 3.2, cy - 1, cx + 3.2, cy + 7, 0, 180)


# ------------------------------------------------------------------ page background + header

def _background(c):
    _fc(c, PAGE_BG)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    _alpha(c, 0.07)
    _fc(c, PURPLE)
    c.circle(W - 40, H - 60, 150, stroke=0, fill=1)
    _fc(c, PINK)
    c.circle(20, 330, 120, stroke=0, fill=1)
    _alpha(c, 0.10)
    _fc(c, PURPLE)
    for px, py, s in ((60, H - 80, 14), (W - 90, H - 250, 12), (50, 470, 10)):     # little "+" marks
        c.rect(px - s / 2, py - s / 6, s, s / 3, stroke=0, fill=1)
        c.rect(px - s / 6, py - s / 2, s / 3, s, stroke=0, fill=1)
    _alpha(c, 1)


def _small_header(c, title):
    """Slim header used on the extra pages. Returns the y where content may start."""
    _background(c)
    lw = _draw_logo_small(c, MARGIN, H - 20, 34)
    _fc(c, PURPLE)
    c.setFont('Helvetica-Bold', 15)
    c.drawString(MARGIN + lw + 8, H - 43, 'PulseRead')
    _fc(c, MUTED)
    c.setFont('Helvetica', 9)
    c.drawRightString(W - MARGIN, H - 40, title)
    _sc(c, LINE)
    c.setLineWidth(0.7)
    c.line(MARGIN, H - 58, W - MARGIN, H - 58)
    return H - 82


def _footer(c, page_no):
    _icon_lock(c, MARGIN + 6, 52)
    _fc(c, SOFT)
    c.setFont('Helvetica-Bold', 8.5)
    c.drawString(MARGIN + 18, 49, 'A complete wellness snapshot, in one private report.')
    _draw_para(c, DISCLAIMER, MARGIN, 40, W - 2 * MARGIN, size=7, color=MUTED, leading=9.5)
    _fc(c, MUTED)
    c.setFont('Helvetica', 7)
    c.drawRightString(W - MARGIN, 18, f'PulseRead  |  page {page_no}')


# ------------------------------------------------------------------ page 1

def _page_one(c, data):
    _background(c)
    cx = W / 2

    _draw_logo_hero(c, cx, H - 20, 104)

    _fc(c, INK)
    c.setFont('Helvetica-Bold', 27)
    c.drawCentredString(cx, H - 164, 'YOUR PULSE.')
    _fc(c, PURPLE)
    c.drawCentredString(cx, H - 194, 'ONE COMPLETE PICTURE.')

    _sc(c, PURPLE)
    c.setLineWidth(1.2)
    c.line(cx - 112, H - 213, cx - 62, H - 213)
    c.line(cx + 62, H - 213, cx + 112, H - 213)
    _fc(c, INK)
    c.setFont('Helvetica-Bold', 11)
    c.drawCentredString(cx, H - 217, 'Wellness Report')
    _fc(c, SOFT)
    c.setFont('Helvetica', 9)
    c.drawCentredString(cx, H - 232, 'Heart rate, HRV, breathing, stress and personal guidance - all in one place.')

    # ---------------- the profile card ----------------
    card_x, card_w = MARGIN, W - 2 * MARGIN
    card_top, card_h = H - 250, 296
    card_y = card_top - card_h
    _card(c, card_x, card_y, card_w, card_h)

    ax, ay = card_x + 38, card_top - 40
    _icon_bg(c, ax, ay, 22, TINT)
    _icon_heart(c, ax, ay + 1, 9, PINK)
    _fc(c, INK)
    c.setFont('Helvetica-Bold', 14)
    c.drawString(card_x + 70, card_top - 36, 'Your Results')
    generated = datetime.now(timezone.utc).strftime('%B %d, %Y  %H:%M UTC')
    _fc(c, MUTED)
    c.setFont('Helvetica', 8.5)
    c.drawString(card_x + 70, card_top - 50, f'Generated {generated}')

    def chip(right_x, label, filled):
        c.setFont('Helvetica-Bold', 8)
        tw = c.stringWidth(label, 'Helvetica-Bold', 8) + 22
        x = right_x - tw
        y = card_top - 46
        if filled:
            _round_rect(c, x, y, tw, 20, 10, fill=PURPLE)
            _fc(c, colors.white)
        else:
            _round_rect(c, x, y, tw, 20, 10, fill=colors.white, stroke=LINE)
            _fc(c, SOFT)
        c.drawString(x + 11, y + 6.5, label)
        return x - 8

    nx = chip(card_x + card_w - 18, f'Signal Quality  {_txt(data.get("signal_quality_pct"), "%")}', True)
    chip(nx, 'Contactless rPPG', False)

    tab_y = card_top - 82
    labels = ['Overview', 'Heart Rate', 'HRV', 'Stress', 'Breathing', 'Guidance']
    x = card_x + 22
    c.setFont('Helvetica-Bold', 8.5)
    for i, lab in enumerate(labels):
        tw = c.stringWidth(lab, 'Helvetica-Bold', 8.5)
        _fc(c, PURPLE if i == 0 else MUTED)
        c.drawString(x, tab_y, lab)
        if i == 0:
            _sc(c, PURPLE)
            c.setLineWidth(2)
            c.line(x, tab_y - 7, x + tw, tab_y - 7)
        x += tw + 26
    _sc(c, LINE)
    c.setLineWidth(0.7)
    c.line(card_x + 14, tab_y - 9, card_x + card_w - 14, tab_y - 9)

    px, pw = card_x + 18, card_w - 36
    py_top = tab_y - 20
    ph = 122
    _round_rect(c, px, py_top - ph, pw, ph, 10, fill=colors.white, stroke=LINE)
    _fc(c, INK)
    c.setFont('Helvetica-Bold', 10)
    c.drawString(px + 14, py_top - 18, 'Key Results')
    _sc(c, LINE)
    c.line(px, py_top - 26, px + pw, py_top - 26)

    rel = data.get('bpm_reliability')
    cells = [
        ('HEART RATE' + (f' - {rel.upper()} RELIABILITY' if rel else ''), _txt(data.get('heart_rate_bpm'), ' BPM'),
         'heart_rate', data.get('heart_rate_bpm'), 'Typical: 60-100 BPM'),
        ('HRV (RMSSD)', _txt(data.get('hrv_rmssd_ms'), ' ms'), 'rmssd',
         data.get('hrv_rmssd_ms') if data.get('hrv_available') else None, 'Typical: 35-70 ms'),
        ('HRV (SDNN)', _txt(data.get('hrv_sdnn_ms'), ' ms'), 'sdnn',
         data.get('hrv_sdnn_ms') if data.get('hrv_available') else None, 'Typical: 40-80 ms'),
        ('STRESS LEVEL', _txt(data.get('stress_level')), None, None, 'Very Low / Low / Moderate / High'),
        ('SIGNAL QUALITY', _txt(data.get('signal_quality_pct'), '%'), 'quality', data.get('signal_quality_pct'),
         'Good 75+ / Fair 50-75 / Poor <50'),
        ('BREATHING RATE', _txt(data.get('breathing_rate_per_min'), ' /min'), 'breathing',
         data.get('breathing_rate_per_min'), 'Typical: 12-20 /min'),
    ]
    col_w = pw / 3
    for i, (lab, val, metric, raw, note) in enumerate(cells):
        r, k = divmod(i, 3)
        cx0 = px + 14 + k * col_w
        cy0 = py_top - 44 - r * 46
        _fc(c, MUTED)
        c.setFont('Helvetica-Bold', 7)
        c.drawString(cx0, cy0, lab)
        _fc(c, INK)
        c.setFont('Helvetica-Bold', 14)
        c.drawString(cx0, cy0 - 17, val)
        if metric and RR and raw is not None:                       # status chip next to the value
            a = RR.assess(metric, raw)
            if a['status']:
                tx = cx0 + c.stringWidth(val, 'Helvetica-Bold', 14) + 7
                label = a['status'].upper()
                c.setFont('Helvetica-Bold', 6.5)
                tw = c.stringWidth(label, 'Helvetica-Bold', 6.5) + 9
                _round_rect(c, tx, cy0 - 16, tw, 11, 5.5, fill=colors.HexColor(KIND_COLORS[a['kind']]))
                _fc(c, colors.white)
                c.drawString(tx + 4.5, cy0 - 13, label)
        _fc(c, SOFT)
        c.setFont('Helvetica', 7.2)
        src = data.get('breathing_source') if metric == 'breathing' else None
        c.drawString(cx0, cy0 - 28, note + (f' ({src})' if src else ''))
        if k > 0:
            _sc(c, LINE)
            c.line(cx0 - 8, cy0 + 8, cx0 - 8, cy0 - 30)

    rec = data.get('recommendation')
    ny = py_top - ph - 16
    _fc(c, INK)
    c.setFont('Helvetica-Bold', 10)
    c.drawString(px + 2, ny - 4, 'Nervous System State')
    if rec:
        label = f"{rec.get('state_label', '')}"
        _fc(c, PURPLE)
        c.setFont('Helvetica-Bold', 9.5)
        c.drawString(px + 2, ny - 20, label)
        lw = c.stringWidth(label, 'Helvetica-Bold', 9.5)
        _fc(c, MUTED)
        c.setFont('Helvetica', 8)
        c.drawString(px + 2 + lw + 8, ny - 20, f"match confidence: {rec.get('match_confidence', '--')}")
        _draw_para(c, rec.get('nervous_system_note', ''), px + 2, ny - 26, pw - 4, size=8.5, color=SOFT, leading=11.5)
    else:
        _draw_para(c, 'No state could be matched (heart rate was not available).', px + 2, ny - 12, pw - 4,
                   size=8.5, color=SOFT)

    # ---------------- four guidance cards ----------------
    gap = 10
    cw = (W - 2 * MARGIN - 3 * gap) / 4
    top = card_y - 26
    bodies = [(rec or {}).get(k) or 'Not available for this reading.'
              for k in ('breathing', 'exercise', 'sleep', 'nutrition')]
    max_h = top - 80                       # keep clear of the footer
    for body_size in (8.6, 8.0, 7.5, 7.0, 6.6):          # shrink the text until the longest one fits
        body_lead = body_size * 1.33
        need = max(_para(b, size=body_size, leading=body_lead).wrap(cw - 18, 10000)[1] for b in bodies)
        if need + 78 <= max_h:
            break
    ch = min(max_h, need + 88)
    items = [
        ('Breathing', 'breathing', _icon_breathing, '#cffafe'),
        ('Exercise', 'exercise', _icon_exercise, '#fee2e2'),
        ('Sleep', 'sleep', _icon_sleep, '#e0e7ff'),
        ('Nutrition', 'nutrition', _icon_nutrition, '#dcfce7'),
    ]
    for i, (title, key, icon, tint) in enumerate(items):
        x0 = MARGIN + i * (cw + gap)
        _card(c, x0, top - ch, cw, ch, r=12)
        icx, icy = x0 + cw / 2, top - 26
        _icon_bg(c, icx, icy, 17, colors.HexColor(tint))
        icon(c, icx, icy)
        _fc(c, INK)
        c.setFont('Helvetica-Bold', 10)
        c.drawCentredString(x0 + cw / 2, top - 58, title)
        _draw_para(c, bodies[i], x0 + 9, top - 68, cw - 18, size=body_size, color=SOFT, leading=body_lead)


# ------------------------------------------------------------------ extra pages

class _Flow:
    """Tiny top-to-bottom layout cursor that starts a new styled page when space runs out."""

    def __init__(self, c, title, page_no):
        self.c = c
        self.page_no = page_no
        self.title = title
        self.y = _small_header(c, title)
        self.floor = 110

    def need(self, h):
        if self.y - h < self.floor:
            _footer(self.c, self.page_no)
            self.c.showPage()
            self.page_no += 1
            self.y = _small_header(self.c, self.title)

    def heading(self, text):
        self.y -= 14                      # breathing room above a section heading
        self.need(46)
        _fc(self.c, INK)
        self.c.setFont('Helvetica-Bold', 13)
        self.c.drawString(MARGIN, self.y, text)
        _sc(self.c, PURPLE)
        self.c.setLineWidth(2)
        self.c.line(MARGIN, self.y - 6, MARGIN + 34, self.y - 6)
        self.y -= 26

    def paragraph(self, text, size=9.5, color=SOFT, bold=False, gap=8):
        p = _para(text, size=size, color=color, leading=size * 1.45, bold=bold)
        _, h = p.wrap(W - 2 * MARGIN, 10000)
        self.need(h + gap)
        p.drawOn(self.c, MARGIN, self.y - h)
        self.y -= h + gap

    def table(self, rows, col_widths, gap=10):
        t = Table(rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PURPLE), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f8fd')]),
            ('GRID', (0, 0), (-1, -1), 0.5, LINE), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        _, h = t.wrap(sum(col_widths), 10000)
        self.need(h + gap)
        t.drawOn(self.c, MARGIN, self.y - h)
        self.y -= h + gap


def _how_to_read(flow, data):
    """Plain-language table: what each number means, the typical range, and where YOU are."""
    if not RR:
        return
    flow.heading('How to read your results')
    flow.paragraph('A "typical" range is where most healthy adults at rest fall. Being outside it is not a diagnosis - '
                   'the best comparison is you today versus your own usual numbers.', size=9, gap=8)
    st = ParagraphStyle('c', fontName='Helvetica', fontSize=8.2, leading=10.6, textColor=SOFT)
    stb = ParagraphStyle('cb', parent=st, fontName='Helvetica-Bold', textColor=INK)
    head = ParagraphStyle('h', parent=st, fontName='Helvetica-Bold', textColor=colors.white)
    values = {
        'heart_rate': (data.get('heart_rate_bpm'), ' BPM'),
        'rmssd': (data.get('hrv_rmssd_ms') if data.get('hrv_available') else None, ' ms'),
        'sdnn': (data.get('hrv_sdnn_ms') if data.get('hrv_available') else None, ' ms'),
        'mean_rr': (data.get('mean_ibi_ms'), ' ms'),
        'breathing': (data.get('breathing_rate_per_min'), ' /min'),
        'quality': (data.get('signal_quality_pct'), '%'),
    }
    rows = [[Paragraph(t, head) for t in ('Measure', 'What it tells you', 'Typical range', 'Your result')]]
    for key in ('heart_rate', 'rmssd', 'sdnn', 'mean_rr', 'breathing', 'quality'):
        info = RR.RANGES[key]
        v, unit = values[key]
        a = RR.assess(key, v)
        if a['status']:
            col = KIND_COLORS[a['kind']]
            mine = Paragraph(f'<b>{v}{unit}</b><br/><font color="{col}"><b>{a["status"]}</b></font>', st)
        else:
            mine = Paragraph('--', st)
        bands = ' / '.join(f"{b[2]} {('<' + str(b[1])) if b[0] is None else (str(b[0]) + '+' if b[1] is None else str(b[0]) + '-' + str(b[1]))}"
                           for b in info['bands'])
        extra = f'<br/><font color="#8a8fa3">{info["note"]}</font>' if info['note'] else ''
        rows.append([Paragraph(info['label'], stb), Paragraph(info['meaning'] + extra, st),
                     Paragraph(f'<b>{info["typical"]}</b><br/><font color="#8a8fa3">{bands}</font>', st), mine])
    rows.append([Paragraph('Stress level', stb),
                 Paragraph(RR.STRESS_NOTE, st), Paragraph(RR.STRESS_LEVELS.replace(', ', ' / '), st),
                 Paragraph(f'<b>{_txt(data.get("stress_level"))}</b>', st)])
    t = Table(rows, colWidths=[78, 205, 150, 80], repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PURPLE),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f8fd')]),
        ('GRID', (0, 0), (-1, -1), 0.5, LINE), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    _, h = t.wrap(sum([78, 205, 150, 80]), 10000)
    flow.need(h + 10)
    t.drawOn(flow.c, MARGIN, flow.y - h)
    flow.y -= h + 10
    flow.paragraph('Heart rate (60-100) and breathing (12-20) are the usual textbook resting ranges. HRV has no single '
                   '"normal" - it depends on age, fitness, sleep and how it is measured, and the HRV bands here are '
                   "this app's own simple guide, not clinical limits.", size=8, color=MUTED, gap=6)


def _challenge_pages(flow, challenge):
    flow.need(230)                     # keep the heading, its table and its reading together on one page
    flow.heading('Physiological Challenge')
    proto = challenge.get('protocol', {})
    flow.paragraph(f"<b>{proto.get('label', 'Challenge')}</b> - {proto.get('note', '')}")
    titles = {p['name']: p['title'] for p in proto.get('phases', [])}
    for person in challenge.get('people', []):
        flow.need(170)
        flow.paragraph(f"<b>{person.get('label', 'Person')}</b>", size=10.5, color=INK, gap=4)
        rows = [['Phase', 'Heart rate (BPM)', 'RMSSD (ms)', 'Breathing (/min)', 'Windows']]
        for name in ('baseline', 'challenge', 'recovery'):
            ph = person.get('phases', {}).get(name, {})
            rows.append([titles.get(name, name.capitalize()), _txt(ph.get('hr')), _txt(ph.get('rmssd')),
                         _txt(ph.get('resp')), _txt(ph.get('n'))])
        flow.table(rows, [150, 115, 100, 115, 60], gap=8)
        flow.paragraph(person.get('reading', ''), gap=4)
        for note in person.get('notes', []):
            flow.paragraph(f'Note: {note}', size=8, color=MUTED, gap=3)
        flow.y -= 6
    flow.paragraph(challenge.get('safety', ''), size=8, color=MUTED)


# ------------------------------------------------------------------ public API

def build_report_pdf(data, out_path):
    c = rl_canvas.Canvas(out_path, pagesize=letter)
    c.setTitle('PulseRead - Wellness Report')
    c.setAuthor('PulseRead')

    _page_one(c, data)
    _footer(c, 1)

    if True:      # page 2 always: how to read the numbers (+ baseline / challenge when present)
        c.showPage()
        flow = _Flow(c, 'Wellness Report - details', 2)
        _how_to_read(flow, data)
        if data.get('baseline_note'):
            flow.heading('Personal Baseline')
            flow.paragraph(data['baseline_note'])
        if data.get('challenge'):
            _challenge_pages(flow, data['challenge'])
        _footer(c, flow.page_no)

    c.save()
    return out_path
