"""Within-camera tracklet stitching.

ByteTrack often gives one motorcycle several IDs (occlusion, perspective change, a missed detection). Left alone, those
fragments look like 'lookalike' competitors of each other and block every cross-camera match, so fragments of the same
physical vehicle are merged before features are built.

Two tracks are the same vehicle when either
  * they overlap in time and their boxes overlap strongly (duplicate IDs on one bike), or
  * one ends where the other begins (short gap, nearby position, similar size and colour).
Two tracks visible at the same time in different places are never merged: one bike cannot be in two places.
"""
from __future__ import annotations

import heapq

import numpy as np

from .features import bhattacharyya, hsv_hist


def _iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, w) * max(0.0, h)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _best_hist(t):
    if "_hist" not in t:
        item = max(t["heap"], key=lambda i: i[0]) if t["heap"] else None
        t["_hist"] = hsv_hist(item[2], mask=item[3]) if item is not None else None
    return t["_hist"]


def relation(x: dict, y: dict, step: float, max_gap: float = 1.5) -> str:
    """'duplicate' | 'sequel' | 'concurrent' (different vehicles) | 'apart'."""
    ox, oy = x["obs"], y["obs"]
    x0, x1, y0, y1 = ox[0][0], ox[-1][0], oy[0][0], oy[-1][0]
    if min(x1, y1) - max(x0, y0) > -0.5 * step:  # they coexist in time
        yt = np.array([o[0] for o in oy])
        ious = []
        for o in ox:
            j = int(np.argmin(np.abs(yt - o[0])))
            if abs(yt[j] - o[0]) <= 0.6 * step:
                ious.append(_iou(o[1:5], oy[j][1:5]))
        return "duplicate" if len(ious) >= 2 and np.mean(ious) >= 0.5 else "concurrent"
    first, second = (ox, oy) if x1 < y0 else (oy, ox)
    gap = second[0][0] - first[-1][0]
    if gap > max_gap:
        return "apart"
    e, s = first[-1], second[0]
    ce = ((e[1] + e[3]) / 2, (e[2] + e[4]) / 2)
    cs = ((s[1] + s[3]) / 2, (s[2] + s[4]) / 2)
    diag = max(np.hypot(e[3] - e[1], e[4] - e[2]), np.hypot(s[3] - s[1], s[4] - s[2]), 1.0)
    dist = np.hypot(ce[0] - cs[0], ce[1] - cs[1]) / diag
    area = lambda o: max(1.0, (o[3] - o[1]) * (o[4] - o[2]))
    ratio = area(s) / area(e)
    ha, hb = _best_hist(x), _best_hist(y)
    colour_ok = ha is None or hb is None or bhattacharyya(ha, hb) >= 0.6
    if dist <= 0.8 + 1.0 * gap and 0.4 <= ratio <= 2.5 and colour_ok:
        return "sequel"
    return "apart"


def stitch_raw(raw: dict, fps: float, stride: int, top_k: int) -> dict:
    """raw: {tid: {"obs": [...], "heap": [...]}}  ->  same structure with fragments merged ("merged": [ids])."""
    ids = sorted(raw, key=lambda i: min(o[0] for o in raw[i]["obs"]))
    for i in ids:
        raw[i]["obs"].sort(key=lambda o: o[0])
    step = stride / fps
    rel = {}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            rel[(ids[a], ids[b])] = relation(raw[ids[a]], raw[ids[b]], step)
    r = lambda i, j: rel.get((i, j)) or rel.get((j, i))
    clusters = [[i] for i in ids]
    changed = True
    while changed:
        changed = False
        for ci in range(len(clusters)):
            for cj in range(ci + 1, len(clusters)):
                pairs = [r(i, j) for i in clusters[ci] for j in clusters[cj]]
                if any(p == "concurrent" for p in pairs):
                    continue                       # two bikes at once: never the same vehicle
                if any(p in ("duplicate", "sequel") for p in pairs):
                    clusters[ci] += clusters.pop(cj)
                    changed = True
                    break
            if changed:
                break
    out = {}
    for cl in clusters:
        cl = sorted(cl, key=lambda i: raw[i]["obs"][0][0])
        obs = sorted((o for i in cl for o in raw[i]["obs"]), key=lambda o: (o[0], -o[5]))
        merged_obs = []
        for o in obs:                              # keep the stronger of two boxes at the same time (duplicates)
            if merged_obs and abs(o[0] - merged_obs[-1][0]) < 0.5 * step:
                continue
            merged_obs.append(o)
        heap = heapq.nlargest(top_k, [it for i in cl for it in raw[i]["heap"]], key=lambda it: it[0])
        out[cl[0]] = {"obs": merged_obs, "heap": heap, "merged": sorted(cl)}
    return out
