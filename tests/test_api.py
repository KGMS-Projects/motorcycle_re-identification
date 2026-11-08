import os
import tempfile

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
os.environ["DATA_DIR"] = tempfile.mkdtemp()

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def test_health_and_ui():
    assert client.get("/api/health").json() == {"ok": True}
    assert "Motorcycle re-identification review" in client.get("/").text


def test_analyze_requires_valid_videos():
    assert client.post("/api/analyze", json={"video_a": "nope", "video_b": "nope"}).status_code == 400


def test_unknown_job_is_404():
    assert client.get("/api/jobs/missing").status_code == 404
    assert client.get("/api/jobs/missing/result").status_code == 404
