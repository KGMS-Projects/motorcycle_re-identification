"""Appearance features: embeddings, colour, shape, image quality and plate OCR."""
from __future__ import annotations

import logging
import os
import re
from difflib import SequenceMatcher

import cv2
import numpy as np

log = logging.getLogger("reid.features")
EPS = 1e-9
CHROMATIC = {"red", "orange", "yellow", "green", "blue", "purple"}


def l2(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), EPS)


def ramp(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / max(hi - lo, EPS), 0.0, 1.0))


# ---------------------------------------------------------------- quality
def sharpness_score(img) -> float:
    g = cv2.cvtColor(cv2.resize(img, (96, 96)), cv2.COLOR_BGR2GRAY)
    return float(min(1.0, cv2.Laplacian(g, cv2.CV_64F).var() / 150.0))


# ---------------------------------------------------------------- colour
NAMES = ["black", "white", "grey", "red", "orange", "yellow", "green", "blue", "purple"]


def _centre(img, margin: float = 0.1):
    h, w = img.shape[:2]
    my, mx = int(h * margin), int(w * margin)
    c = img[my:h - my, mx:w - mx]
    return c if c.size else img


def apply_mask(img, mask):
    """Replace everything outside the vehicle mask by neutral grey (no-op without a mask)."""
    if mask is None:
        return img
    m = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return np.where(m[..., None] > 0, img, 114).astype(np.uint8)


def _pixels(img, mask=None, size: int = 48):
    """BGR pixels belonging to the vehicle: the segmentation mask if usable, else the central crop."""
    if mask is not None:
        m = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST) > 0
        if m.sum() >= 40:
            return cv2.resize(img, (size, size))[m]
    return cv2.resize(_centre(img), (size, size)).reshape(-1, 3)


def hsv_hist(img, margin: float = 0.1, mask=None) -> np.ndarray:
    px = np.ascontiguousarray(_pixels(img, mask).reshape(-1, 1, 3), dtype=np.uint8)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    return (hist / max(hist.sum(), EPS)).astype(np.float32)


def bhattacharyya(p, q) -> float:
    return float(np.sum(np.sqrt(np.asarray(p) * np.asarray(q))))


def colour_name(h: int, s: int, v: int) -> str:
    if v < 60:
        return "black"
    if s < 45:
        return "white" if v > 180 else "grey"
    d = h * 2
    if d < 12 or d >= 340:
        return "red"
    if d < 40:
        return "orange"
    if d < 70:
        return "yellow"
    if d < 170:
        return "green"
    if d < 260:
        return "blue"
    return "purple"


def _classify(hsv):
    h, s, v = hsv[:, 0].astype(int) * 2, hsv[:, 1].astype(int), hsv[:, 2].astype(int)
    code = np.where(v < 60, 0, np.where(s < 60, np.where(v > 180, 1, 2), -1))
    hue = np.select([h < 12, h < 40, h < 70, h < 170, h < 260, h < 340], [3, 4, 5, 6, 7, 8], default=3)
    return np.where(code == -1, hue, code)


def order_primary(cols: list) -> list:
    """Body colour first. Tyres, seat, engine, shadows and rider clothing make nearly every bike partly black, so a
    clear chromatic colour (>= 12% of the vehicle) is the primary colour even when black covers more pixels."""
    cols = sorted(cols, key=lambda d: -d["frac"])
    chrom = [c for c in cols if c["name"] in CHROMATIC and c["frac"] >= 0.12]
    if chrom:
        p0 = chrom[0]
        cols = [p0] + [c for c in cols if c is not p0]
    return cols


def chromatic_share(cols: list) -> float:
    return float(sum(c["frac"] for c in cols if c["name"] in CHROMATIC))


def dominant_colours(img, k: int = 3, mask=None) -> list[dict]:
    px = np.ascontiguousarray(_pixels(img, mask, 32).reshape(-1, 1, 3), dtype=np.uint8)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    code = _classify(hsv)
    out = []
    for i, name in enumerate(NAMES):
        sel = code == i
        f = float(sel.mean())
        if f < 0.02:
            continue
        hh = hsv[sel, 0].astype(float) * 2
        if name == "red":
            hh = np.where(hh >= 340, hh - 360, hh)
        out.append({"name": name, "frac": f, "hue": float(hh.mean() % 360),
                    "sat": float(hsv[sel, 1].mean()), "val": float(hsv[sel, 2].mean())})
    return order_primary(out) or [{"name": "grey", "frac": 1.0, "hue": 0.0, "sat": 0.0, "val": 120.0}]


def hue_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


# ---------------------------------------------------------------- shape
def shape_descriptor(img) -> np.ndarray:
    """Grid of gradient-orientation histograms: a structure cue independent of the DNN embedding."""
    g = cv2.cvtColor(cv2.resize(img, (96, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag, ang = cv2.cartToPolar(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    bins = (ang % np.pi / np.pi * 8).astype(int) % 8
    feats = []
    for r in np.array_split(np.arange(64), 3):
        for c in np.array_split(np.arange(96), 4):
            m, b = mag[np.ix_(r, c)], bins[np.ix_(r, c)]
            feats.append(l2(np.bincount(b.ravel(), weights=m.ravel(), minlength=8)))
    return l2(np.concatenate(feats))


# ---------------------------------------------------------------- embedders
def pad_square(img):
    h, w = img.shape[:2]
    s = max(h, w)
    out = np.full((s, s, 3), 114, np.uint8)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    return out


class HandcraftedEmbedder:
    """Dependency-free fallback: spatial HSV layout + gradient structure."""
    name = "handcrafted-hsv-grad"
    calib = (0.60, 0.95)

    def __call__(self, crops):
        return np.stack([self._one(c) for c in crops])

    @staticmethod
    def _one(c):
        h, w = c.shape[:2]
        parts = []
        for ys in (slice(0, h // 2), slice(h // 2, h)):
            for xs in (slice(0, w // 2), slice(w // 2, w)):
                hsv = cv2.cvtColor(cv2.resize(c[ys, xs], (24, 24)), cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 3, 3], [0, 180, 0, 256, 0, 256]).flatten()
                parts.append(np.sqrt(hist / max(hist.sum(), EPS)))
        parts.append(shape_descriptor(c))
        return l2(np.concatenate(parts))


class DinoEmbedder:
    """DINOv2-small (CPU): CLS token + mean patch token."""
    name = "dinov2-small"
    calib = (0.30, 0.80)

    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.model = AutoModel.from_pretrained("facebook/dinov2-small").eval()

    def __call__(self, crops):
        rgb = [cv2.cvtColor(pad_square(c), cv2.COLOR_BGR2RGB) for c in crops]
        out = []
        with self.torch.no_grad():
            for i in range(0, len(rgb), 16):
                inp = self.proc(images=rgb[i:i + 16], return_tensors="pt")
                hs = self.model(**inp).last_hidden_state
                out.append(self.torch.cat([hs[:, 0], hs[:, 1:].mean(1)], 1).numpy())
        return l2(np.concatenate(out))


_EMBEDDER = None


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        choice = os.environ.get("EMBEDDER", "auto")
        if choice in ("auto", "dino"):
            try:
                _EMBEDDER = DinoEmbedder()
            except Exception as e:  # offline / not installed
                log.warning("DINOv2 unavailable (%s); using handcrafted embedder", e)
        if _EMBEDDER is None:
            _EMBEDDER = HandcraftedEmbedder()
    return _EMBEDDER


# ---------------------------------------------------------------- plates
_OCR = None
_CONFUSE = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"})


def ocr_state() -> str:
    return "not started" if _OCR is None else ("unavailable" if _OCR is False else "easyocr")


def _get_ocr():
    global _OCR
    if _OCR is None:
        try:
            import easyocr
            _OCR = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception as e:
            log.warning("OCR unavailable (%s); plates will be reported as missing", e)
            _OCR = False
    return _OCR


def _plate_boxes(img, max_n: int = 2) -> list:
    """[(score, region, width_in_original_pixels)] for plate-like rectangles (white or yellow, wide, textured)."""
    h = img.shape[0]
    roi = img[int(h * 0.3):]
    if roi.shape[0] < 16 or roi.shape[1] < 24:
        return []
    s = max(1.0, 320.0 / roi.shape[1])
    big = cv2.resize(roi, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(big, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    light = ((vv > 130) & (ss < 90)) | ((hh > 12) & (hh < 40) & (ss > 70) & (vv > 100))
    m = cv2.morphologyEx(light.astype(np.uint8) * 255, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    total = big.shape[0] * big.shape[1]
    found = []
    for c in cnts:
        x, y, w, hgt = cv2.boundingRect(c)
        ar, area = w / max(hgt, 1), w * hgt
        if not (1.3 <= ar <= 6.5) or w < 40 or area < 0.008 * total or area > 0.6 * total:
            continue
        edges = cv2.Canny(gray[y:y + hgt, x:x + w], 60, 160).mean() / 255.0
        if edges < 0.03:
            continue
        pad = 4
        found.append((edges * np.sqrt(area), big[max(0, y - pad):y + hgt + pad, max(0, x - pad):x + w + pad], w / s))
    return sorted(found, key=lambda t: -t[0])[:max_n]


def plate_candidates(img, max_n: int = 2) -> list:
    return [r for _, r, _ in _plate_boxes(img, max_n)]


def plate_width(img) -> float:
    """Width in original pixels of the most plate-like region (0 if none): bigger plates are more readable."""
    b = _plate_boxes(img, 1)
    return float(b[0][2]) if b else 0.0


def _ocr_region(ocr, region) -> list[tuple[str, float]]:
    f = max(1.0, 260.0 / max(1, region.shape[1]))
    g = cv2.cvtColor(cv2.resize(region, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)
    binary = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    results = []
    for variant in (g, binary):  # small plates read differently with and without binarisation; keep both reads
        results += ocr.readtext(variant, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
    out, pieces = [], []
    for box, text, conf in results:
        t = re.sub(r"[^A-Z0-9]", "", text.upper())
        if conf < 0.3 or not t:
            continue
        pieces.append((float(np.mean([p[1] for p in box])), float(np.mean([p[0] for p in box])), t, float(conf)))
        if 4 <= len(t) <= 10 and any(ch.isdigit() for ch in t):
            out.append((t, float(conf)))
    if len(pieces) >= 2:  # two-line plates: join the lines top-to-bottom, left-to-right
        joined = "".join(p[2] for p in sorted(pieces))
        if 5 <= len(joined) <= 12 and any(ch.isdigit() for ch in joined):
            out.append((joined, float(np.mean([p[3] for p in pieces]))))
    return out


def read_plates(crops) -> list[tuple[str, float]]:
    ocr = _get_ocr()
    if not ocr:
        return []
    reads = []
    for c in crops:
        got = False
        for region in plate_candidates(c):
            r = _ocr_region(ocr, region)
            reads += r
            got = got or bool(r)
        if not got:  # fall back to the lower part of the whole crop
            reads += _ocr_region(ocr, c[int(c.shape[0] * 0.35):])
    return reads


def plate_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.upper().translate(_CONFUSE), b.upper().translate(_CONFUSE)).ratio()


def plate_contained(a: str, b: str) -> bool:
    """True when the shorter plate read (>= 4 chars) sits inside the longer one: OCR often adds or drops an edge character."""
    ra, rb = a.upper(), b.upper()
    na, nb = ra.translate(_CONFUSE), rb.translate(_CONFUSE)
    short, long_ = sorted((na, nb), key=len)
    if len(short) >= 4 and short in long_:
        return True
    # letters are misread far more often than digits on small plates, so also compare the digit runs
    da, db = re.sub(r"\D", "", ra), re.sub(r"\D", "", rb)
    ds, dl = sorted((da, db), key=len)
    return len(ds) >= 4 and ds in dl


def aggregate_plate(reads) -> dict | None:
    """Vote across noisy reads: the text most supported by similar reads wins."""
    if not reads:
        return None
    best, best_s, members = None, -1.0, []
    for t, _ in reads:
        group = [(u, c) for u, c in reads if plate_ratio(t, u) >= 0.75]
        sc = sum(c for _, c in group)
        if sc > best_s:
            best, best_s, members = t, sc, group
    return {"text": best, "conf": float(np.mean([c for _, c in members])), "n_reads": len(reads)}
