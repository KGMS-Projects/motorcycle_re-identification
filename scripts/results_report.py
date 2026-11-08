"""Turn an exported JSON into the technical-results section for the demo / report.

Usage:  python scripts/results_report.py reid_<job>.json > results.md
"""
import json
import sys


def pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def main(path):
    d = json.load(open(path))
    pairs = d["candidate_pairs"]
    by_status = {}
    for p in pairs:
        by_status.setdefault(p["verification"]["outcome"], []).append(p)
    t = d.get("timing", {})
    print("# Technical results\n")
    print(f"* Models: {d['versions']['detector']}, {d['versions']['tracker']}, {d['versions']['embedder']}, OCR {d['versions']['ocr']}")
    print(f"* Camera A tracks: {len(d['tracks']['a'])}, camera B tracks: {len(d['tracks']['b'])}")
    print(f"* Processing speed: {t.get('pipeline_fps')} frames/s ({t.get('frames_processed')} frames in {t.get('total_s')} s); "
          f"matching and review took {t.get('match_review_s')} s")
    print(f"* Candidate pairs reviewed: {len(pairs)} "
          f"(accepted {len(by_status.get('accepted', []))}, uncertain {len(by_status.get('uncertain', []))}, "
          f"rejected {len(by_status.get('rejected', []))})\n")

    print("## Result per camera-A track\n")
    print("| A track | Selected B | Top candidate | Outcome | No reliable match |\n|---|---|---|---|---|")
    for s in d["summary_a"]:
        print(f"| A#{s['a']} | {'B#%s' % s['selected_b'] if s['selected_b'] is not None else '-'} | "
              f"{'B#%s' % s['top_b'] if s['top_b'] is not None else '-'} | {s['status']} | {'yes' if s['no_reliable_match'] else 'no'} |")

    for name, title in (("accepted", "Accepted matches"), ("uncertain", "Uncertain cases"), ("rejected", "Rejected (top examples)")):
        rows = sorted(by_status.get(name, []), key=lambda p: -p["verification"]["confidence"])[:5]
        if not rows:
            continue
        print(f"\n## {title}\n")
        for p in rows:
            v = p["verification"]
            print(f"### A#{p['track_a']} and B#{p['track_b']}: {name}, confidence {pct(v['confidence'])}, uncertainty {pct(v['uncertainty'])}")
            print(f"* Reason: {'; '.join(v['reasons'])}")
            for label, key in (("Supports", "supporting_evidence"), ("Contradicts", "contradictory_evidence"), ("Missing", "missing_evidence")):
                if v[key]:
                    print(f"* {label}: " + "; ".join(i["text"] for i in v[key]))
            print()

    m = d.get("metrics")
    print("## Matching metrics\n")
    if m:
        print(f"* Labelled tracks: {m['n_labelled']}; true matches accepted: {m['true_matches']}; "
              f"**false matches accepted: {m['false_matches']}**; missed: {m['missed_matches']}; correct rejections: {m['correct_rejections']}")
        print(f"* Precision {pct(m['precision'])}, recall {pct(m['recall'])}, F1 {pct(m['f1'])}, top-1 accuracy {pct(m['top1_accuracy'])}, MRR {m['mrr']}")
    else:
        print("No ground truth supplied. Paste it into the UI (e.g. `[{\"a\": 78, \"b\": 5}]`) and re-export.")


if __name__ == "__main__":
    main(sys.argv[1])
