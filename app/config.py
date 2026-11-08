"""Central configuration. Every value the matcher/reviewer uses is tunable from the UI."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields

DATA_DIR = os.environ.get("DATA_DIR", "./data")
SAMPLES_DIR = os.environ.get("SAMPLES_DIR", "./samples")

# Initial engineering weights (not scientifically validated - tune on your own footage).
DEFAULT_WEIGHTS = {
    "embedding": 0.30,
    "colour": 0.25,
    "physical": 0.10,
    "plate": 0.15,
    "temporal": 0.10,
    "movement": 0.10,
}


@dataclass
class Params:
    # --- detection / tracking (changing these needs a re-detect) ---
    yolo_weights: str = "yolov8n-seg.pt"  # segmentation: removes background from crops
    det_conf: float = 0.30
    imgsz: int = 640
    stride: int = 2              # run the detector on every Nth frame
    min_track_obs: int = 4       # drop tracks with fewer observations
    min_box_side: int = 28       # px, ignore tiny boxes
    top_k_frames: int = 12       # best frames per track used for features
    enable_ocr: bool = True

    # --- timing / movement (re-match only) ---
    timing_mode: str = "auto"    # auto (file-name timestamps) | synced | offset | unknown
    b_offset_s: float = 0.0      # camera B video starts this many seconds after A
    clock_tol_s: float = 3.0     # camera clock uncertainty
    min_travel_s: float = 0.0    # plausible A-exit -> B-entry window
    max_travel_s: float = 600.0
    direction_mode: str = "any"  # any | same | opposite

    # --- scoring / review (re-match only) ---
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    emb_auto: bool = True        # calibrate similarity ranges from this run's own pairs
    emb_lo: float | None = None  # cosine -> [0,1] calibration; None = embedder default
    emb_hi: float | None = None
    accept_thr: float = 0.80
    reject_thr: float = 0.55
    margin: float = 0.06         # candidates closer than this are "lookalikes"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Params":
        base = cls()
        for f in fields(cls):
            if d and f.name in d and d[f.name] is not None and f.name != "weights":
                default = getattr(base, f.name)
                v = d[f.name]
                if isinstance(default, bool):
                    v = bool(v)
                elif isinstance(default, int):
                    v = int(float(v))
                elif isinstance(default, float) or default is None:
                    v = float(v)
                setattr(base, f.name, v)
        if d and isinstance(d.get("weights"), dict):
            base.weights = {**DEFAULT_WEIGHTS, **{k: float(v) for k, v in d["weights"].items()}}
        return base

    def detect_key(self) -> str:
        return (f"{self.yolo_weights}|{self.det_conf}|{self.imgsz}|{self.stride}|"
                f"{self.min_track_obs}|{self.min_box_side}|{self.top_k_frames}|{self.enable_ocr}")
