from tests.helpers import make_track, params
from app.matcher import rank_all
from app.reviewer import review_pair

CAL = (0.3, 0.8)


def run(tracks_a, tracks_b, **kw):
    p = params(**kw)
    cands = rank_all(tracks_a, tracks_b, p, CAL)
    ta = {t.id: t for t in tracks_a}
    tb = {t.id: t for t in tracks_b}
    for c in cands:
        c["review"] = review_pair(ta[c["a"]], tb[c["b"]], c, p, CAL)
    return {(c["a"], c["b"]): c for c in cands}


def test_same_motorcycle_ranks_first_and_is_accepted():
    A = [make_track(1, "A", ident=1, plate={"text": "ABC1234", "conf": .9, "n_reads": 3}),
         make_track(2, "A", ident=2, colour="blue", t1=12)]
    B = [make_track(7, "B", ident=1, t0=14, plate={"text": "ABC1234", "conf": .85, "n_reads": 3}),
         make_track(8, "B", ident=3, colour="blue", t0=16)]
    r = run(A, B)
    assert r[(1, 7)]["ctx"]["rank_a"] == 1
    assert r[(1, 7)]["review"]["status"] == "accepted"
    assert r[(1, 8)]["review"]["status"] != "accepted"


def test_similar_looking_but_different_plates_is_rejected():
    """A high visual score must never override clearly different readable plates."""
    A = [make_track(1, "A", ident=1, plate={"text": "ABC1234", "conf": .9, "n_reads": 3})]
    B = [make_track(7, "B", ident=1, t0=14, plate={"text": "XYZ5678", "conf": .9, "n_reads": 3})]
    c = run(A, B)[(1, 7)]
    assert c["components"]["embedding"]["score"] > 0.7   # visually it is the same bike
    assert c["review"]["status"] == "rejected"
    assert any(i["type"] == "plate" for i in c["review"]["contradicting"])


def test_impossible_timing_rejects():
    A = [make_track(1, "A", ident=1, t1=50)]
    B = [make_track(7, "B", ident=1, t0=5)]
    assert run(A, B)[(1, 7)]["review"]["status"] == "rejected"


def test_lookalikes_make_result_uncertain_not_accepted():
    A = [make_track(1, "A", ident=1)]
    B = [make_track(7, "B", ident=1, t0=14), make_track(8, "B", ident=1, t0=15)]  # identical appearance
    r = run(A, B)
    assert r[(1, 7)]["ctx"]["lookalikes_a"] == [8]
    assert r[(1, 7)]["review"]["status"] == "uncertain"
    assert any(i["type"] == "ambiguity" for i in r[(1, 7)]["review"]["contradicting"])


def test_poor_tracking_quality_is_uncertain():
    A = [make_track(1, "A", ident=1, q=0.2, n_obs=5)]
    B = [make_track(7, "B", ident=1, t0=14, q=0.2, n_obs=5)]
    c = run(A, B)[(1, 7)]
    assert c["review"]["status"] != "accepted"
    assert any(i["type"] == "tracking" for i in c["review"]["missing"])


def test_unrelated_motorcycle_is_rejected():
    A = [make_track(1, "A", ident=1, colour="red")]
    B = [make_track(7, "B", ident=99, colour="blue", noise=0.3, t0=14)]
    assert run(A, B)[(1, 7)]["review"]["status"] == "rejected"


def test_missing_evidence_is_reported():
    A, B = [make_track(1, "A", ident=1)], [make_track(7, "B", ident=1, t0=14)]
    c = run(A, B, timing_mode="unknown")[(1, 7)]
    kinds = {i["type"] for i in c["review"]["missing"]}
    assert {"plate", "timing"} <= kinds


def test_black_vs_red_body_cannot_be_accepted_despite_matching_embedding():
    A = [make_track(1, "A", ident=1, colour="red")]
    B = [make_track(7, "B", ident=1, colour="black", t0=14)]
    c = run(A, B)[(1, 7)]
    assert c["colour_gate"] < 1.0
    assert c["review"]["status"] != "accepted"
    assert any(i["type"] == "colour" for i in c["review"]["contradicting"])


def test_adaptive_calibration_spreads_saturated_scores():
    """Every pair has a high raw cosine; the real match must still stand out from the rest."""
    from app.matcher import adaptive_calib
    import numpy as np
    vals = [0.74, 0.76, 0.75, 0.77, 0.74, 0.76, 0.75, 0.93, 0.76]
    lo, hi = adaptive_calib(vals, (0.30, 0.80), 0.12)
    from app.features import ramp
    assert ramp(0.93, lo, hi) == 1.0 and ramp(0.76, lo, hi) < 0.6


def test_same_readable_plate_is_accepted_even_when_views_differ():
    """Different camera views give a weak visual match; a confirmed identical plate must still win."""
    plate = {"text": "BCT4521", "conf": 0.8, "n_reads": 4}
    A = [make_track(1, "A", ident=1, colour="red", plate=dict(plate))]
    B = [make_track(7, "B", ident=2, colour="red", noise=0.3, t0=14, plate=dict(plate, conf=0.7))]
    c = run(A, B)[(1, 7)]
    assert c["review"]["status"] == "accepted"
    assert any(i["type"] == "plate" for i in c["review"]["supporting"])


def test_black_parts_do_not_hide_a_red_body():
    """Regression: a red bike with black tyres/seat/shadow was labelled 'black' and penalised against a red bike."""
    import numpy as np
    from app.features import dominant_colours
    img = np.zeros((90, 130, 3), np.uint8)
    img[:] = (20, 20, 20)                  # mostly black (tyres, seat, shadow)
    img[10:45, 20:110] = (0, 0, 210)       # red tank/fairing, ~30% of the box
    cols = dominant_colours(img)
    assert cols[0]["name"] == "red"
    assert any(c["name"] == "black" for c in cols)


def test_clear_winner_on_both_sides_needs_a_lower_absolute_score():
    A = [make_track(1, "A", ident=1, colour="red"), make_track(2, "A", ident=5, colour="blue", t1=12)]
    B = [make_track(7, "B", ident=1, colour="red", t0=14), make_track(8, "B", ident=9, colour="green", noise=.3, t0=16)]
    mc = run(A, B)[(1, 7)]["confidence"]
    assert run(A, B)[(1, 7)]["ctx"]["margin_b"] >= 0.08
    accepted = run(A, B, accept_thr=mc + 0.05)[(1, 7)]["review"]["status"]      # within 0.10 of threshold, distinct
    uncertain = run(A, B, accept_thr=mc + 0.30)[(1, 7)]["review"]["status"]
    assert accepted == "accepted" and uncertain == "uncertain"


def test_plate_with_an_extra_ocr_character_still_confirms_identity():
    """Regression: plates read as 14314 and 4314 (similarity 0.89) were not treated as a match."""
    A = [make_track(1, "A", ident=1, colour="red", plate={"text": "14314", "conf": 0.8, "n_reads": 4})]
    B = [make_track(7, "B", ident=2, colour="red", noise=0.3, t0=14, plate={"text": "4314", "conf": 0.7, "n_reads": 2})]
    c = run(A, B)[(1, 7)]
    assert c["review"]["status"] == "accepted"
    assert any(i["type"] == "plate" and i["strength"] == "strong" for i in c["review"]["supporting"])


def test_different_plates_with_similar_length_are_not_confirmed():
    A = [make_track(1, "A", ident=1, plate={"text": "14314", "conf": 0.8, "n_reads": 4})]
    B = [make_track(7, "B", ident=1, t0=14, plate={"text": "27890", "conf": 0.8, "n_reads": 4})]
    assert run(A, B)[(1, 7)]["review"]["status"] == "rejected"


def test_low_confidence_but_agreeing_plate_reads_are_accepted():
    """Regression: A#78 / B#5 both carry BFT 4314; small-plate OCR confidence was just under the old 0.45 cut-off."""
    A = [make_track(1, "A", ident=1, colour="red", plate={"text": "14314", "conf": 0.36, "n_reads": 3}),
         make_track(2, "A", ident=7, colour="blue", t1=12)]
    B = [make_track(7, "B", ident=1, colour="red", t0=14, plate={"text": "4314", "conf": 0.33, "n_reads": 2}),
         make_track(8, "B", ident=1, colour="red", t0=15, plate={"text": "NC9127", "conf": 0.5, "n_reads": 2})]
    c = run(A, B)[(1, 7)]
    assert c["review"]["status"] == "accepted"   # a lookalike (B8) exists, but the plate settles identity
