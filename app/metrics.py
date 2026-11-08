"""Evaluation against optional ground truth: [{"a": 3, "b": 7}, {"a": 4, "b": null}]."""
from __future__ import annotations


def evaluate(summary_a: list[dict], candidates: list[dict], gt: list[dict]) -> dict:
    by_a = {s["a"]: s for s in summary_a}
    tp = fp = fn = tn = top1 = n_pos = 0
    rr, rows = [], []
    for g in gt:
        a, b = g["a"], g.get("b")
        s = by_a.get(a)
        if s is None:
            continue
        sel = s["selected_b"]
        if b is None:
            if sel is None:
                tn += 1
            else:
                fp += 1
        else:
            n_pos += 1
            ranked = sorted((c for c in candidates if c["a"] == a), key=lambda c: c["ctx"]["rank_a"])
            ids = [c["b"] for c in ranked]
            top1 += int(s["top_b"] == b)
            rr.append(1.0 / (ids.index(b) + 1) if b in ids else 0.0)
            if sel == b:
                tp += 1
            elif sel is None:
                fn += 1
            else:
                fp += 1
        rows.append({"a": a, "truth_b": b, "top_b": s["top_b"], "selected_b": sel, "status": s["status"]})
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / n_pos if n_pos else None
    return {"n_labelled": len(rows), "true_matches": tp, "false_matches": fp, "missed_matches": fn,
            "correct_rejections": tn, "precision": prec, "recall": rec,
            "f1": (2 * prec * rec / (prec + rec)) if prec and rec else None,
            "top1_accuracy": top1 / n_pos if n_pos else None,
            "mrr": sum(rr) / len(rr) if rr else None, "rows": rows}
