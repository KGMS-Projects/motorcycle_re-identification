"""Independent evidence reviewer (stage 5).

Deliberately does NOT reuse the matcher's aggregated scores. It re-derives its own checks from
the raw per-frame features (frame-pair consensus, dominant-colour hue, plate text, timing state)
and applies structured rules, so a high matcher score can still be rejected or flagged uncertain.
"""
from __future__ import annotations

import numpy as np

from .features import CHROMATIC, chromatic_share, hue_diff, plate_contained, plate_ratio, ramp

W = {"strong": 3, "moderate": 2, "weak": 1}


def _i(kind, pol, strength, text):
    return {"type": kind, "polarity": pol, "strength": strength, "text": text}


def _visual(A, B, calib, items):
    lo, hi = calib
    S = A.frame_embs @ B.frame_embs.T
    thr = lo + 0.6 * (hi - lo)
    consensus = float((np.mean(S.max(1) >= thr) + np.mean(S.max(0) >= thr)) / 2)
    best = float(S.max())
    if consensus >= 0.7:
        items.append(_i("visual", "support", "strong", f"{consensus:.0%} of frames in each track have a close visual match in the other"))
    elif consensus >= 0.4:
        items.append(_i("visual", "support", "moderate", f"Partial frame-level visual agreement ({consensus:.0%} of frames)"))
    elif best < lo + 0.3 * (hi - lo):
        items.append(_i("visual", "contradict", "moderate", f"No frame pair looks alike (best cosine {best:.2f})"))
    else:
        items.append(_i("visual", "contradict", "weak", f"Weak frame-level visual agreement ({consensus:.0%})"))


def _colour(A, B, qp, items):
    ca, cb = A.colours[0], B.colours[0]
    ha, hb = ca["name"] in CHROMATIC, cb["name"] in CHROMATIC
    if ha and hb:
        dh = hue_diff(ca["hue"], cb["hue"])
        if ca["name"] == cb["name"] or dh < 30:
            items.append(_i("colour", "support", "moderate", f"Body colour matches ({ca['name']})"))
        elif dh > 60:
            items.append(_i("colour", "contradict", "moderate",
                            f"Body colours differ: {ca['name']} vs {cb['name']} (lighting can shift colour)"))
    elif not ha and not hb:
        if ca["name"] == cb["name"]:
            items.append(_i("colour", "support", "weak", f"Both mostly {ca['name']}; achromatic colours are weak evidence"))
    else:
        # tyres, seat and shadows make most bikes partly black, so only call it a conflict when the dark one
        # has no body colour at all
        dark = B if ha else A
        coloured = ca if ha else cb
        if chromatic_share(dark.colours) < 0.05 and coloured["frac"] >= 0.25:
            items.append(_i("colour", "contradict", "moderate" if qp >= 0.45 else "weak",
                            f"One body is clearly {coloured['name']}, the other has no body colour"))


def _plate(A, B, items):
    pa, pb = A.plate, B.plate
    if not (pa and pb):
        side = "both tracks" if not (pa or pb) else ("camera A" if not pa else "camera B")
        items.append(_i("plate", "missing", "moderate", f"Plate not readable in {side}"))
        return
    r = plate_ratio(pa["text"], pb["text"])
    contained = plate_contained(pa["text"], pb["text"])
    txt = f"Plates read {pa['text']} / {pb['text']} (similarity {r:.0%}, OCR confidence {pa['conf']:.2f} / {pb['conf']:.2f})"
    if r >= 0.85 or contained:
        items.append(_i("plate", "support", "strong", txt + (" (one read is contained in the other)" if contained and r < 1.0 else "")))
        shortest = min(len(pa["text"]), len(pb["text"]))
        if min(pa["conf"], pb["conf"]) >= 0.30 and (r == 1.0 or contained or (shortest >= 5 and r >= 0.85)):
            items[-1]["confirmed"] = True
    elif r <= 0.55 and min(pa["conf"], pb["conf"]) >= 0.6 and min(len(pa["text"]), len(pb["text"])) >= 5:
        items.append(_i("plate", "contradict", "strong", txt + " - clearly different readable plates"))
    else:
        items.append(_i("plate", "missing", "weak", txt + " - low-confidence or partial read"))


def review_pair(A, B, cand, p, calib) -> dict:
    items: list[dict] = []
    ctx, comps = cand["ctx"], cand["components"]
    _plate(A, B, items)
    _visual(A, B, calib, items)
    _colour(A, B, cand["q_pair"], items)
    shp = float(A.shape @ B.shape)
    slo, shi = cand.get("shape_calib", (0.5, 0.95))
    if shp >= slo + 0.7 * (shi - slo):
        items.append(_i("structure", "support", "weak", f"Edge/structure layout is similar ({shp:.2f})"))
    elif shp < slo + 0.2 * (shi - slo):
        items.append(_i("structure", "contradict", "weak", f"Edge/structure layout differs ({shp:.2f})"))
    if A.riders == B.riders:
        items.append(_i("riders", "support", "weak", f"Same rider count ({A.riders}) - weak, riders change"))
    else:
        items.append(_i("riders", "contradict", "weak", f"Rider count differs ({A.riders} vs {B.riders}) - weak, riders change"))

    t = comps["temporal"]["info"]
    st = t["state"]
    if st == "unknown":
        items.append(_i("timing", "missing", "moderate", t["text"]))
    elif st in ("consistent", "tolerated"):
        items.append(_i("timing", "support", "moderate" if st == "consistent" else "weak", t["text"]))
    elif st == "impossible":
        items.append(_i("timing", "contradict", "strong", t["text"]))
    else:
        items.append(_i("timing", "contradict", "weak", t["text"]))

    m = comps["movement"]["info"]
    if m["state"] == "unknown":
        items.append(_i("movement", "missing", "weak", m["text"]))
    elif m["state"] == "consistent":
        items.append(_i("movement", "support", "weak", m["text"]))
    elif m["state"] == "inconsistent":
        items.append(_i("movement", "contradict", "moderate", m["text"]))

    qp = cand["q_pair"]
    if qp < 0.35 or min(A.quality["n_obs"], B.quality["n_obs"]) < 8:
        items.append(_i("tracking", "missing", "moderate",
                        f"Low tracking quality (pair q={qp:.2f}; {A.quality['n_obs']}/{B.quality['n_obs']} observations)"))
    if ctx["lookalikes_a"] or ctx["lookalikes_b"]:
        who = [f"B#{b}" for b in ctx["lookalikes_a"]] + [f"A#{a}" for a in ctx["lookalikes_b"]]
        items.append(_i("ambiguity", "contradict", "moderate",
                        f"Lookalike candidates score within {p.margin:.2f}: {', '.join(who)}"))
    if not ctx["mutual_best"]:
        items.append(_i("ranking", "contradict", "weak",
                        f"Not the mutual best match (rank {ctx['rank_a']} for A#{A.id}, rank {ctx['rank_b']} for B#{B.id})"))

    sup = [i for i in items if i["polarity"] == "support"]
    con = [i for i in items if i["polarity"] == "contradict"]
    mis = [i for i in items if i["polarity"] == "missing"]
    S, C = sum(W[i["strength"]] for i in sup), sum(W[i["strength"]] for i in con)
    rev_conf = 0.5 + 0.5 * (S - C) / (S + C + 2)
    mc = cand["confidence"]
    strong_c = [i for i in con if i["strength"] == "strong"]
    n_mod = sum(1 for i in con if i["strength"] == "moderate" and i["type"] != "ambiguity")
    indep = {i["type"] for i in sup if i["type"] in ("visual", "colour", "plate", "structure", "timing")}

    flags, reasons = [], []
    plate_ok = any(i.get("confirmed") for i in sup)  # same readable plate: decisive unless something contradicts it
    if strong_c:
        status, reasons = "rejected", [i["text"] for i in strong_c]
    elif plate_ok:
        if qp < 0.35:
            status, reasons = "uncertain", ["Plates match but tracking quality is low, so the reads may be unreliable"]
        else:
            status, reasons = "accepted", ["The same plate is readable in both tracks and nothing contradicts it"]
    elif mc < p.reject_thr:
        status, reasons = "rejected", ["Combined matching score is below the rejection threshold"]
    elif n_mod >= 2:
        status, reasons = "rejected", ["Several independent cues disagree"]
    else:
        if ctx["lookalikes_a"] or ctx["lookalikes_b"]:
            flags.append("lookalike candidates within margin")
        if not ctx["mutual_best"]:
            flags.append("not the mutual best match")
        if qp < 0.35:
            flags.append("low track quality")
        if len(indep) < 2:
            flags.append("fewer than two independent supporting cues")
        if n_mod >= 1:
            flags.append("a moderate conflicting cue")
        # a clear winner on both sides (far ahead of every other candidate) needs slightly less absolute score
        distinct = ctx["margin_a"] >= 0.08 and ctx["margin_b"] >= 0.08
        # several independent cues agreeing, as judged by the reviewer on its own, also earns the same allowance
        corroborated = rev_conf >= 0.8 and len(indep) >= 3
        if mc < p.accept_thr - (0.10 if (distinct or corroborated) else 0.0):
            flags.append("score below acceptance threshold")
        if rev_conf < 0.6:
            flags.append("independent review confidence is low")
        status = "uncertain" if flags else "accepted"
        reasons = flags or ["Strong, consistent evidence with no significant contradiction"]

    conf = 0.5 * mc + 0.5 * rev_conf
    if strong_c:
        conf = min(conf * 0.5, 0.3)
    amb = 1.0 if (ctx["lookalikes_a"] or ctx["lookalikes_b"]) else 1 - ramp(ctx["margin_a"], 0.0, 0.15)
    unc = float(np.clip(0.4 * (1 - qp) + 0.3 * min(1.0, len(mis) / 4) + 0.3 * amb, 0, 1))
    return {"status": status, "confidence": round(float(conf), 3), "uncertainty": round(unc, 3),
            "matcher_confidence": round(float(mc), 3), "reviewer_confidence": round(float(rev_conf), 3),
            "supporting": sup, "contradicting": con, "missing": mis, "reasons": reasons}
