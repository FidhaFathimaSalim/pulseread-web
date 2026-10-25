"""
face_tracker.py — finds several faces in a webcam frame, keeps a stable
identity ("Person 1", "Person 2" ...) for each, and cuts a steady square
crop around every face.

Why this exists
---------------
open-rppg measures ONE face per video. To measure several people from one
webcam we give the model one cropped "video" per person. Two things matter:

  1. STABLE IDS   - a person must keep the same number from frame to frame,
                    otherwise their pulse history gets mixed with someone else's.
  2. STEADY CROPS - if the crop box jitters, the background moves in the clip and
                    that motion drowns the tiny pulse colour change. So box
                    positions are smoothed and only move when the face really moves.

Face detectors (chosen automatically, first one that works wins):
  1. MediaPipe FaceLandmarker (needs the `mediapipe` package + face_landmarker.task, the file
     your project already has). Handles tilted / partly turned faces better.
  2. OpenCV Haar cascade (needs a healthy OpenCV; front-facing faces only).
If neither works the dashboard shows a red message with the exact reason.
You can also pass your own `detector` function: frame_rgb -> list of (x, y, w, h).
"""

import math
import os

import numpy as np
import cv2


CASCADE_NAME = 'haarcascade_frontalface_default.xml'
HERE = os.path.dirname(os.path.abspath(__file__))


def cascade_candidates():
    """Every place the Haar face file might be. The copy shipped next to this file comes FIRST,
    so face detection works even when OpenCV was installed without its data folder
    (a common problem: cv2.data is missing in some pip / conda / headless installs)."""
    c = [os.path.join(HERE, CASCADE_NAME),
         os.path.join(HERE, 'models', CASCADE_NAME),
         os.path.join(HERE, '..', 'models', CASCADE_NAME)]
    try:
        c.append(os.path.join(cv2.data.haarcascades, CASCADE_NAME))
    except Exception:                     # noqa: BLE001 - cv2.data missing
        pass
    c.append(os.path.join(os.path.dirname(cv2.__file__), 'data', CASCADE_NAME))
    for root in ('/usr/share/opencv4/haarcascades', '/usr/share/opencv/haarcascades',
                 '/usr/local/share/opencv4/haarcascades', '/opt/homebrew/share/opencv4/haarcascades'):
        c.append(os.path.join(root, CASCADE_NAME))
    return c


def find_cascade():
    """Path of a usable Haar cascade file, or None."""
    for p in cascade_candidates():
        if os.path.isfile(p):
            return p
    return None


def load_cascade():
    """Load the cascade or raise an error that says exactly what is wrong."""
    if not hasattr(cv2, 'CascadeClassifier'):
        raise RuntimeError('this OpenCV build has no CascadeClassifier (reinstall: pip install opencv-python)')
    path = find_cascade()
    if path is None:
        raise RuntimeError(f'{CASCADE_NAME} not found. Copy it next to face_tracker.py. Looked in: '
                           + ', '.join(os.path.abspath(p) for p in cascade_candidates()[:6]))
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        raise RuntimeError(f'OpenCV could not read the cascade file {os.path.abspath(path)} (file damaged?)')
    return clf, path


def haar_detector_factory(detect_width=640):
    cascade = {'c': None}

    def detect(frame_rgb):
        if cascade['c'] is None:
            cascade['c'], path = load_cascade()
            print(f'[face_tracker] face detector ready (cascade: {os.path.abspath(path)})')
        h, w = frame_rgb.shape[:2]
        scale = detect_width / w if w > detect_width else 1.0
        img = cv2.resize(frame_rgb, (int(w * scale), int(h * scale))) if scale != 1.0 else frame_rgb
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        faces = cascade['c'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
        return [(x / scale, y / scale, fw / scale, fh / scale) for (x, y, fw, fh) in faces]

    return detect


LANDMARKER_NAME = 'face_landmarker.task'


def find_landmarker():
    for p in (os.path.join(HERE, LANDMARKER_NAME), os.path.join(HERE, 'models', LANDMARKER_NAME),
              os.path.join(HERE, '..', LANDMARKER_NAME), os.path.join(HERE, '..', 'models', LANDMARKER_NAME),
              os.path.join(os.getcwd(), LANDMARKER_NAME), os.path.join(os.getcwd(), 'models', LANDMARKER_NAME)):
        if os.path.isfile(p):
            return p
    return None


def mediapipe_detector_factory(max_faces=4):
    """Detector based on MediaPipe FaceLandmarker: the box is the outline of the 478 face landmarks."""
    state = {'lm': None, 'mp': None}

    def detect(frame_rgb):
        if state['lm'] is None:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            path = find_landmarker()
            if path is None:
                raise RuntimeError(f'{LANDMARKER_NAME} not found (put it in the backend folder)')
            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=path),
                running_mode=vision.RunningMode.IMAGE, num_faces=max_faces,
                min_face_detection_confidence=0.5, min_face_presence_confidence=0.5)
            state['lm'] = vision.FaceLandmarker.create_from_options(opts)
            state['mp'] = mp
            print(f'[face_tracker] MediaPipe face detector ready ({os.path.abspath(path)})')
        mp = state['mp']
        h, w = frame_rgb.shape[:2]
        # Large frames (multi-person sends up to 1280 px so every face keeps many pixels) are only
        # DETECTED on a smaller copy - landmarks are 0..1 fractions, so the boxes still map to the full frame.
        det = frame_rgb
        if w > 640:
            det = cv2.resize(frame_rgb, (640, int(round(h * 640 / w))), interpolation=cv2.INTER_AREA)
        res = state['lm'].detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(det)))
        boxes = []
        for lms in res.face_landmarks:
            xs = [p.x for p in lms]
            ys = [p.y for p in lms]
            x0, x1 = max(min(xs), 0) * w, min(max(xs), 1) * w
            y0, y1 = max(min(ys), 0) * h, min(max(ys), 1) * h
            if x1 - x0 > 8 and y1 - y0 > 8:
                boxes.append((x0, y0, x1 - x0, y1 - y0))
        return boxes

    return detect


def auto_detector_factory(prefer='auto'):
    """
    prefer: 'auto' (MediaPipe, then Haar), 'mediapipe', or 'haar'.
    The returned function has .info = {'name': 'mediapipe'|'haar'|None, 'errors': [...]}.
    """
    info = {'name': None, 'errors': []}
    order = {'auto': ['mediapipe', 'haar'], 'mediapipe': ['mediapipe'], 'haar': ['haar']}.get(prefer, ['mediapipe', 'haar'])
    state = {'fn': None}

    def detect(frame_rgb):
        if state['fn'] is None:
            info['errors'] = []
            for name in order:
                try:
                    fn = mediapipe_detector_factory() if name == 'mediapipe' else haar_detector_factory()
                    boxes = fn(frame_rgb)           # forces the model / cascade to load, raises if it is broken
                    state['fn'], info['name'] = fn, name
                    return boxes
                except Exception as e:              # noqa: BLE001
                    info['errors'].append(f'{name}: {str(e)[:160]}')
                    print(f'[face_tracker] {name} detector unavailable: {str(e)[:200]}')
            raise RuntimeError(' | '.join(info['errors']) +
                               '  ->  repair OpenCV (see check_face_detector.py) or install mediapipe')
        return state['fn'](frame_rgb)

    detect.info = info
    return detect


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _inter(a, b):
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return iw * ih


def _nms(boxes, thr=0.4, contain_thr=0.6):
    """Keep the biggest boxes first. A box is dropped when it overlaps a kept box (IoU >= thr)
    OR when most of it (>= contain_thr of its own area) lies inside a kept, bigger box.
    The second rule removes the classic false 'second face' on glasses / nose / chin that sits
    INSIDE a real face and has a low IoU with it."""
    keep = []
    for b in sorted(boxes, key=lambda b: -b[2] * b[3]):
        area = b[2] * b[3]
        dup = False
        for k in keep:
            if _iou(b, k) >= thr or (area > 0 and _inter(b, k) / area >= contain_thr):
                dup = True
                break
        if not dup:
            keep.append(b)
    return keep


class Track:
    def __init__(self, tid, box, ts):
        self.id = tid
        self.box = [float(v) for v in box]     # smoothed x, y, w, h in source-frame pixels
        self.first_seen = ts
        self.last_seen = ts                    # last time a detector actually saw this face
        self.hits = 1

    @property
    def center(self):
        return (self.box[0] + self.box[2] / 2, self.box[1] + self.box[3] / 2)

    def update(self, det, ts, pos_alpha=0.35, size_alpha=0.15, dead_zone=0.06):
        """Move toward the new detection, but ignore tiny moves (dead zone) to stop jitter."""
        x, y, w, h = det
        cx, cy = x + w / 2, y + h / 2
        ox, oy = self.center
        size = max(self.box[2], self.box[3])
        if math.hypot(cx - ox, cy - oy) > dead_zone * size:
            ncx = ox + pos_alpha * (cx - ox)
            ncy = oy + pos_alpha * (cy - oy)
        else:
            ncx, ncy = ox, oy
        nw = self.box[2] + size_alpha * (w - self.box[2])
        nh = self.box[3] + size_alpha * (h - self.box[3])
        self.box = [ncx - nw / 2, ncy - nh / 2, nw, nh]
        self.last_seen = ts
        self.hits += 1


class FaceTracker:
    def __init__(self, max_people=4, detect_every=2, max_missing_sec=3.0, detector=None,
                 match_dist=0.8, min_hits=2, detector_kind='auto'):
        self.max_people = max_people
        self.detect_every = max(1, int(detect_every))
        self.max_missing_sec = max_missing_sec
        self.match_dist = match_dist           # max centre distance, in units of the face size
        self.min_hits = min_hits               # a face must be detected this many times before it counts as a person
        self.detector = detector or auto_detector_factory(detector_kind)
        self.tracks = {}                       # id -> Track
        self._frame_no = 0
        self.detector_broken = False
        self.detector_error_text = ''

    @property
    def detector_name(self):
        info = getattr(self.detector, 'info', None)
        return (info or {}).get('name')

    # ---------------------------------------------------------------
    def update(self, frame_rgb, ts):
        """
        Feed one frame. Returns (live_tracks, removed_ids).
        live_tracks: list of Track currently followed (sorted by id).
        removed_ids: ids whose face has been missing too long -> drop their buffers.
        """
        self._frame_no += 1
        if self._frame_no % self.detect_every == 0 or not self.tracks:
            try:
                dets = _nms(list(self.detector(frame_rgb)))
            except Exception as e:  # noqa: BLE001
                if not self.detector_broken:
                    print('[face_tracker] detector failed:', repr(e))
                self.detector_broken = True
                self.detector_error_text = str(e)[:300]
                dets = []
            self._match(dets, ts)

        # a one-off false detection (a poster, a shadow) never becomes a person: it is dropped after 1 s
        removed = [tid for tid, t in self.tracks.items()
                   if ts - t.last_seen > (self.max_missing_sec if t.hits >= self.min_hits else 1.0)]
        for tid in removed:
            del self.tracks[tid]
        return [self.tracks[k] for k in sorted(self.tracks) if self.tracks[k].hits >= self.min_hits], removed

    def _match(self, dets, ts):
        """Greedy nearest-centre matching of detections to existing tracks."""
        pairs = []
        for tid, t in self.tracks.items():
            tcx, tcy = t.center
            size = max(t.box[2], t.box[3])
            for di, d in enumerate(dets):
                dcx, dcy = d[0] + d[2] / 2, d[1] + d[3] / 2
                dist = math.hypot(dcx - tcx, dcy - tcy) / size
                if dist <= self.match_dist:
                    pairs.append((dist, tid, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for dist, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            self.tracks[tid].update(dets[di], ts)
            used_t.add(tid)
            used_d.add(di)

        # unmatched detections become new people (largest faces first), up to max_people
        free_ids = [i for i in range(1, self.max_people + 1) if i not in self.tracks]
        for di in sorted((i for i in range(len(dets)) if i not in used_d), key=lambda i: -dets[i][2] * dets[i][3]):
            if not free_ids:
                break
            tid = free_ids.pop(0)
            self.tracks[tid] = Track(tid, dets[di], ts)


# ------------------------------------------------------------------ cropping

def crop_face(frame_rgb, box, out_size=176, context=1.6):
    """
    Square crop centred on the (smoothed) face box, `context` times the face width,
    shifted (not padded) to stay inside the frame, resized to out_size x out_size.
    Returns (crop, (x, y, side)) so callers can also draw the box.
    """
    h, w = frame_rgb.shape[:2]
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2
    side = int(round(min(max(bw, bh) * context, h, w)))
    side = max(side, 32)
    x0 = int(round(min(max(cx - side / 2, 0), w - side)))
    y0 = int(round(min(max(cy - side / 2, 0), h - side)))
    crop = frame_rgb[y0:y0 + side, x0:x0 + side]
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop, (x0, y0, side)


def normalised_box(box, frame_w, frame_h):
    """Face box as fractions of the frame (0..1) so the browser can draw it at any size."""
    x, y, bw, bh = box
    return {'x': round(x / frame_w, 4), 'y': round(y / frame_h, 4),
            'w': round(bw / frame_w, 4), 'h': round(bh / frame_h, 4)}
