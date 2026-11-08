"""Track-level representation: many frames -> one robust descriptor + quality statistics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import (aggregate_plate, apply_mask, dominant_colours, hsv_hist, l2, order_primary, read_plates,
                       plate_width, shape_descriptor, sharpness_score)


@dataclass
class Track:
    id: int
    cam: str
    fps: float
    frame_w: int
    frame_h: int
    obs: list            # [t, x1, y1, x2, y2, conf, riders, occlusion, truncated]
    quality: dict
    emb: np.ndarray      # aggregated, L2-normalised
    frame_embs: np.ndarray
    shape: np.ndarray
    hist: np.ndarray
    colours: list        # [{name, frac, hue, sat, val}]
    riders: int
    aspect: float
    plate: dict | None
    entry: str
    exit: str
    direction: tuple
    moving: bool
    trend: float
    thumb: str | None = None
    merged: list | None = None  # ids of the raw tracker fragments merged into this track

    @property
    def t0(self) -> float:
        return self.obs[0][0]

    @property
    def t1(self) -> float:
        return self.obs[-1][0]

    @property
    def q(self) -> float:
        return self.quality["q"]

    def summary(self, with_obs: bool = False) -> dict:
        d = {"id": self.id, "t0": round(self.t0, 2), "t1": round(self.t1, 2), "quality": self.quality,
             "colours": [{"name": c["name"], "frac": round(c["frac"], 2)} for c in self.colours[:3]],
             "riders": self.riders, "aspect": round(self.aspect, 2), "plate": self.plate,
             "entry": self.entry, "exit": self.exit, "direction": [round(float(x), 2) for x in self.direction],
             "moving": self.moving, "thumb": self.thumb, "merged": self.merged or [self.id]}
        if with_obs:
            d["obs"] = [[round(o[0], 2), *[round(x, 1) for x in o[1:5]], round(o[5], 2)] for o in self.obs]
        return d


def _side(pt) -> str:
    d = {"left": pt[0], "right": 1 - pt[0], "top": pt[1], "bottom": 1 - pt[1]}
    k = min(d, key=d.get)
    return k if d[k] < 0.25 else "inside"


def motion(obs, W, H):
    c = np.array([[(o[1] + o[3]) / 2 / W, (o[2] + o[4]) / 2 / H] for o in obs])
    n = max(1, min(3, len(c) // 3))
    s, e = c[:n].mean(0), c[-n:].mean(0)
    v = e - s
    mag = float(np.linalg.norm(v * np.array([W, H])) / W)
    direction = tuple(v / (np.linalg.norm(v) + 1e-9))
    area = lambda o: max(1.0, (o[3] - o[1]) * (o[4] - o[2]))
    trend = float(np.log(np.mean([area(o) for o in obs[-n:]]) / np.mean([area(o) for o in obs[:n]])))
    return _side(s), _side(e), direction, mag >= 0.08, trend


def track_quality(obs, sharp_vals, stride, fps) -> dict:
    n = len(obs)
    step = stride / fps
    t0, t1 = obs[0][0], obs[-1][0]
    expected = max(1, round((t1 - t0) / step) + 1)
    det_ratio = min(1.0, n / expected)
    conf = float(np.mean([o[5] for o in obs]))
    areas = np.maximum([(o[3] - o[1]) * (o[4] - o[2]) for o in obs], 1.0)
    dl = np.abs(np.diff(np.log(areas)))
    stability = float(1 - min(1.0, np.median(dl) * 3)) if len(dl) else 0.5
    vis = float(np.mean([1 - max(o[7], 0.5 * o[8]) for o in obs]))
    ts = np.array([o[0] for o in obs])
    gaps = int(np.sum(np.diff(ts) > max(1.0, 5 * step)))
    min_side = float(np.median([min(o[3] - o[1], o[4] - o[2]) for o in obs]))
    sharp = float(np.mean(sharp_vals)) if len(sharp_vals) else 0.0
    q = min(1.0, n / 10) * (0.2 * det_ratio + 0.15 * conf + 0.2 * sharp + 0.15 * min(1.0, min_side / 90)
                            + 0.2 * vis + 0.1 * stability)
    return {"n_obs": n, "duration": round(t1 - t0, 2), "det_ratio": round(det_ratio, 2),
            "mean_conf": round(conf, 2), "bbox_stability": round(stability, 2),
            "visibility": round(vis, 2), "fragments": gaps, "min_side_px": round(min_side),
            "sharpness": round(sharp, 2), "q": round(float(q), 3)}


def aggregate_colours(crops, weights, masks=None) -> list:
    acc: dict[str, dict] = {}
    masks = masks or [None] * len(crops)
    for c, w, m in zip(crops, weights, masks):
        for d in dominant_colours(c, mask=m):
            a = acc.setdefault(d["name"], {"name": d["name"], "frac": 0.0, "hue": d["hue"],
                                           "sat": d["sat"], "val": d["val"]})
            a["frac"] += d["frac"] * w
    tot = sum(a["frac"] for a in acc.values()) or 1.0
    for a in acc.values():
        a["frac"] /= tot
    return order_primary(list(acc.values()))


def build_track(tid, cam, obs, samples, meta, embedder, enable_ocr) -> Track:
    """samples: list of (score, crop, mask|None) - the best frames of this track. The mask (from the segmentation
    model) removes road and background so colour, embedding and structure describe the motorcycle itself."""
    samples = sorted(samples, key=lambda s: -s[0])
    scores = np.array([s[0] for s in samples], dtype=float) + 1e-3
    crops = [s[1] for s in samples]
    masks = [s[2] if len(s) > 2 else None for s in samples]
    vis = [apply_mask(c, m) for c, m in zip(crops, masks)]
    embs = embedder(vis)
    emb = l2((embs * scores[:, None]).sum(0))
    hist = sum(hsv_hist(c, mask=m) * w for c, m, w in zip(crops, masks, scores)) / scores.sum()
    shape = l2(sum(shape_descriptor(v) * w for v, w in zip(vis, scores)) / scores.sum())
    colours = aggregate_colours(crops[:6], scores[:6], masks[:6])
    sharp = [sharpness_score(c) for c in crops]
    # read plates from the frames where a plate is largest and sharpest
    ocr_idx = sorted(range(len(crops)), key=lambda i: -(plate_width(crops[i]) * (0.5 + sharp[i])))[:6]
    plate = aggregate_plate(read_plates([crops[i] for i in ocr_idx])) if enable_ocr else None
    entry, exit_, direction, moving, trend = motion(obs, meta["width"], meta["height"])
    aspect = float(np.median([(o[3] - o[1]) / max(1.0, o[4] - o[2]) for o in obs]))
    return Track(tid, cam, meta["fps"], meta["width"], meta["height"], obs,
                 track_quality(obs, sharp, meta["stride"], meta["fps"]),
                 emb, embs.astype(np.float32), shape, hist.astype(np.float32), colours,
                 int(round(np.median([o[6] for o in obs]))), aspect, plate,
                 entry, exit_, direction, moving, trend)
