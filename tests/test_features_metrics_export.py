import numpy as np

from app.features import (HandcraftedEmbedder, colour_name, dominant_colours, plate_ratio, sharpness_score)
from app.metrics import evaluate
from app.pipeline import export_csv, export_json
from tests.helpers import make_track, params
from app.matcher import rank_all
from app.reviewer import review_pair


def solid(bgr):
    return np.full((80, 120, 3), bgr, np.uint8)


def test_colour_names():
    assert colour_name(0, 200, 200) == "red"
    assert colour_name(110, 200, 200) == "blue"
    assert colour_name(0, 0, 20) == "black"
    assert dominant_colours(solid((0, 0, 220)))[0]["name"] == "red"


def test_handcrafted_embedding_separates_colours():
    emb = HandcraftedEmbedder()
    v = emb([solid((0, 0, 220)), solid((0, 0, 225)), solid((220, 40, 20))])
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


def test_plate_similarity_tolerates_ocr_confusions():
    assert plate_ratio("AB0123", "ABO123") == 1.0
    assert plate_ratio("ABC1234", "XYZ5678") < 0.5


def test_sharp_beats_blurred():
    rng = np.random.default_rng(0)
    img = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    import cv2
    assert sharpness_score(img) > sharpness_score(cv2.GaussianBlur(img, (21, 21), 8))


def _summary():
    summary = [{"a": 1, "top_b": 7, "selected_b": 7, "status": "accepted"},
               {"a": 2, "top_b": 8, "selected_b": 8, "status": "accepted"},   # false match
               {"a": 3, "top_b": 9, "selected_b": None, "status": "uncertain"}]
    cands = [{"a": a, "b": b, "ctx": {"rank_a": 1}} for a, b in ((1, 7), (2, 8), (3, 9))]
    return summary, cands


def test_metrics_counts_false_matches_and_misses():
    s, c = _summary()
    m = evaluate(s, c, [{"a": 1, "b": 7}, {"a": 2, "b": None}, {"a": 3, "b": 9}])
    assert (m["true_matches"], m["false_matches"], m["missed_matches"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_export_contains_required_fields():
    A, B = [make_track(1, "A", ident=1)], [make_track(7, "B", ident=1, t0=14)]
    p = params()
    cands = rank_all(A, B, p, (0.3, 0.8))
    for c in cands:
        c["review"] = review_pair(A[0], B[0], c, p, (0.3, 0.8))
        c["components"] = {k: {"score": v["score"], "detail": v["detail"]} for k, v in c["components"].items()}
    result = {"job_id": "t", "params": {}, "versions": {}, "timing": {}, "summary_a": [], "candidates": cands,
              "videos": {"a": {"tracks": [A[0].summary(True)]}, "b": {"tracks": [B[0].summary(True)]}}}
    j = export_json(result)["candidate_pairs"][0]
    assert {"track_a", "track_b", "similarity_score", "verification"} <= set(j)
    assert {"outcome", "confidence", "uncertainty", "supporting_evidence", "contradictory_evidence"} <= set(j["verification"])
    assert export_csv(result).splitlines()[0].startswith("track_a,track_b")


def test_result_with_numpy_scalars_is_json_safe():
    """Regression: float32 from detection overlap values broke json.dumps after analysis."""
    import json
    from app.pipeline import to_builtin
    data = {"frac": np.float32(0.25), "n": np.int64(3), "flag": np.bool_(True), "v": np.arange(2, dtype=np.float32)}
    out = json.loads(json.dumps(data, default=to_builtin))
    assert out == {"frac": 0.25, "n": 3, "flag": True, "v": [0.0, 1.0]}


def test_start_time_parsed_from_file_names():
    from app.pipeline import parse_start_ts
    a, b = parse_start_ts("20261106_141615.mp4"), parse_start_ts("20261106_141817.mp4")
    assert b - a == 122
    assert parse_start_ts("clip.mp4") is None


def test_plate_localiser_finds_a_plate_like_rectangle():
    import cv2
    from app.features import plate_candidates
    img = np.full((220, 260, 3), (30, 30, 200), np.uint8)             # red bike body
    cv2.rectangle(img, (70, 150), (190, 195), (235, 235, 235), -1)    # white plate
    cv2.putText(img, "BCT 4521", (78, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    found = plate_candidates(img)
    assert len(found) >= 1 and found[0].shape[1] > found[0].shape[0]


def test_plate_voting_prefers_consistent_reads_over_one_confident_outlier():
    from app.features import aggregate_plate
    reads = [("BCT4521", .5), ("BCT4S21", .45), ("BCT4521", .4), ("XQK9", .6)]
    plate = aggregate_plate(reads)
    assert plate["text"].startswith("BCT") and plate["conf"] < 0.6


def test_plate_digit_runs_match_even_when_letters_are_misread():
    from app.features import plate_contained
    assert plate_contained("BFT4314", "8FI4314")      # BFT misread; digits agree
    assert plate_contained("14314", "4314")
    assert not plate_contained("BFT4314", "XYZ5678")
    assert not plate_contained("AB12", "CD12")         # too short to be decisive
