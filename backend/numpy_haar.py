"""
numpy_haar.py — a face detector that needs ONLY numpy and the Haar cascade file
(haarcascade_frontalface_default.xml, shipped next to this file).

Why it exists: some Python environments have a broken OpenCV (no CascadeClassifier) and no
MediaPipe. This reads the same cascade file OpenCV uses and evaluates it itself, so
multi-person mode still finds faces. It is slower than OpenCV's C++ version (roughly
50-200 ms for a 320 px wide picture), so the tracker only calls it a few times per second.

How a Haar cascade works (beginner version):
  A face is checked by a chain of 25 small tests ("stages"). Each test adds up a few
  bright/dark rectangle comparisons (e.g. "the eye band is darker than the cheeks").
  Most non-face windows fail the first test and are dropped immediately, which is what makes
  it fast. A window that passes all 25 tests is a face. The same test is run for windows of
  many sizes and positions, then overlapping hits are merged.

Frontal faces only (like OpenCV's version).
"""

import xml.etree.ElementTree as ET

import numpy as np


class NumpyHaar:
    def __init__(self, xml_path, min_neighbors=4, scale_factor=1.2, min_size=24, max_size=None, detect_width=320):
        root = ET.parse(xml_path).getroot()
        c = root.find('cascade')
        self.base_w = int(c.find('width').text)
        self.base_h = int(c.find('height').text)
        self.min_neighbors = min_neighbors
        self.scale_factor = scale_factor
        self.min_size = max(min_size, self.base_w)
        self.max_size = max_size
        self.detect_width = detect_width

        # --- features: up to 3 weighted rectangles each
        feats = c.find('features')
        nf = len(feats)
        self.rx = np.zeros((nf, 3), np.int32)
        self.ry = np.zeros((nf, 3), np.int32)
        self.rw = np.zeros((nf, 3), np.int32)
        self.rh = np.zeros((nf, 3), np.int32)
        self.rwt = np.zeros((nf, 3), np.float64)
        for i, f in enumerate(feats):
            for k, r in enumerate(f.find('rects')):
                t = r.text.split()
                self.rx[i, k], self.ry[i, k], self.rw[i, k], self.rh[i, k] = (int(t[0]), int(t[1]), int(t[2]), int(t[3]))
                self.rwt[i, k] = float(t[4])

        # --- stages: stump classifiers (feature, threshold, left value, right value)
        self.stages = []
        for st in c.find('stages'):
            thr = float(st.find('stageThreshold').text)
            fi, th, lv, rv = [], [], [], []
            for w in st.find('weakClassifiers'):
                nodes = w.find('internalNodes').text.split()
                leaf = w.find('leafValues').text.split()
                fi.append(int(nodes[2]))
                th.append(float(nodes[3]))
                lv.append(float(leaf[0]))
                rv.append(float(leaf[1]))
            self.stages.append((thr, np.array(fi), np.array(th), np.array(lv), np.array(rv)))

    # ------------------------------------------------------------------ image helpers
    @staticmethod
    def _to_gray(frame_rgb):
        f = frame_rgb.astype(np.float32)
        return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]

    @staticmethod
    def _equalize(gray):
        g = np.clip(gray, 0, 255).astype(np.uint8)
        hist = np.bincount(g.ravel(), minlength=256)
        cdf = hist.cumsum()
        nz = cdf[cdf > 0]
        if nz.size == 0 or cdf[-1] == nz[0]:
            return g.astype(np.float64)
        lut = np.round((cdf - nz[0]) / float(cdf[-1] - nz[0]) * 255.0).clip(0, 255)
        return lut[g]

    # ------------------------------------------------------------------ detection
    def detect(self, frame_rgb):
        """frame_rgb: uint8 (H, W, 3). Returns [(x, y, w, h), ...] in the ORIGINAL frame's pixels."""
        h0, w0 = frame_rgb.shape[:2]
        k = max(1, int(round(w0 / float(self.detect_width))))            # integer down-sampling factor
        gray = self._to_gray(frame_rgb)
        if k > 1:
            hh, ww = (h0 // k) * k, (w0 // k) * k
            gray = gray[:hh, :ww].reshape(hh // k, k, ww // k, k).mean(axis=(1, 3))
        gray = self._equalize(gray)
        H, W = gray.shape

        ii = np.zeros((H + 1, W + 1), np.float64)
        ii[1:, 1:] = gray.cumsum(0).cumsum(1)
        sq = np.zeros((H + 1, W + 1), np.float64)
        sq[1:, 1:] = (gray * gray).cumsum(0).cumsum(1)

        hits = []                                                        # (x, y, size) in down-sampled pixels
        size = float(self.min_size / k if self.min_size / k >= self.base_w else self.base_w)
        max_size = min(H, W) if self.max_size is None else min(self.max_size / k, min(H, W))
        while size <= max_size:
            hits.extend(self._scan(ii, sq, H, W, int(round(size))))
            size *= self.scale_factor
        boxes = self._group(hits)
        return [(x * k, y * k, s * k, s * k) for (x, y, s) in boxes]

    def _scan(self, ii, sq, H, W, size):
        scale = size / float(self.base_w)
        step = max(2, int(round(size * 0.08)))
        xs = np.arange(0, W - size, step)
        ys = np.arange(0, H - size, step)
        if len(xs) == 0 or len(ys) == 0:
            return []
        wx, wy = np.meshgrid(xs, ys)
        wx, wy = wx.ravel(), wy.ravel()

        def box_sum(table, x, y, w, h):
            return table[y + h, x + w] - table[y, x + w] - table[y + h, x] + table[y, x]

        area = float(size * size)
        s1 = box_sum(ii, wx, wy, size, size)
        s2 = box_sum(sq, wx, wy, size, size)
        var = s2 / area - (s1 / area) ** 2
        keep = var > 4.0                                                 # flat windows (blank wall, black frame) can't be faces
        wx, wy = wx[keep], wy[keep]
        if len(wx) == 0:
            return []
        sigma = np.sqrt(var[keep])
        norm = area * sigma

        # scaled integer rectangles for every feature
        sx = np.rint(self.rx * scale).astype(np.int64)
        sy = np.rint(self.ry * scale).astype(np.int64)
        sw = np.maximum(1, np.rint(self.rw * scale)).astype(np.int64)
        sh = np.maximum(1, np.rint(self.rh * scale)).astype(np.int64)

        alive = np.arange(len(wx))
        for thr, fi, th, lv, rv in self.stages:
            if len(alive) == 0:
                return []
            x, y, nrm = wx[alive], wy[alive], norm[alive]
            total = np.zeros(len(alive), np.float64)
            for j in range(len(fi)):
                f = fi[j]
                raw = np.zeros(len(alive), np.float64)
                for r in range(3):
                    wt = self.rwt[f, r]
                    if wt == 0.0:
                        continue
                    raw += wt * box_sum(ii, x + sx[f, r], y + sy[f, r], sw[f, r], sh[f, r])
                total += np.where(raw / nrm < th[j], lv[j], rv[j])
            alive = alive[total >= thr]
        return [(int(wx[i]), int(wy[i]), size) for i in alive]

    def _group(self, hits):
        """Merge overlapping hits; keep groups with at least `min_neighbors` members."""
        n = len(hits)
        if n == 0:
            return []
        arr = np.array(hits, dtype=np.float64)                           # x, y, size
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i in range(n):
            xi, yi, si = arr[i]
            close = (np.abs(arr[:, 0] - xi) <= 0.25 * si) & (np.abs(arr[:, 1] - yi) <= 0.25 * si) & \
                    (arr[:, 2] <= 1.3 * si) & (si <= 1.3 * arr[:, 2])
            for j in np.nonzero(close)[0]:
                a, b = find(i), find(int(j))
                if a != b:
                    parent[a] = b
        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        out = []
        for idx in groups.values():
            if len(idx) >= self.min_neighbors:
                m = arr[idx].mean(axis=0)
                out.append((m[0], m[1], m[2], len(idx)))
        # drop a box that sits inside a bigger, better-supported box
        out.sort(key=lambda b: -b[3])
        kept = []
        for x, y, s, cnt in out:
            cx, cy = x + s / 2, y + s / 2
            if any(abs(cx - (kx + ks / 2)) < 0.4 * ks and abs(cy - (ky + ks / 2)) < 0.4 * ks and s < 1.5 * ks
                   for kx, ky, ks, _ in kept):
                continue
            kept.append((x, y, s, cnt))
        return [(x, y, s) for x, y, s, _ in kept]
