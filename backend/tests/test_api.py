import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from main import app
from models.schemas import AudioFeatures, AudioRegion, FrequencyProfile, TimingIssue
from services import ai_service
from services.ai_service import build_audio_context

client = TestClient(app)


def sse_events(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def anthropic_stream(words: list[str], stop_reason: str = "end_turn") -> bytes:
    """A Messages API response in the wire format, so the real SDK parses it."""
    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        *[("content_block_delta", {"type": "content_block_delta", "index": 0,
                                   "delta": {"type": "text_delta", "text": w}}) for w in words],
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                           "usage": {"output_tokens": 5}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Run the real SDK against a local stand-in for the Messages API."""
    state = {"status": 200, "body": anthropic_stream(["Cut ", "at 300Hz."]), "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
            state["requests"].append({"path": self.path, "headers": dict(self.headers), "body": payload})
            self.send_response(state["status"])
            is_ok = state["status"] == 200
            self.send_header("content-type", "text/event-stream" if is_ok else "application/json")
            self.end_headers()
            self.wfile.write(state["body"] if is_ok else json.dumps(
                {"type": "error", "error": {"type": "api_error", "message": "nope"}}).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_MAX_RETRIES", "0")
    ai_service.get_client.cache_clear()
    yield state
    server.shutdown()
    ai_service.get_client.cache_clear()


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_analyze_returns_features_for_a_region(write_wav, click_track):
    with open(write_wav(click_track()), "rb") as f:
        res = client.post(
            "/analyze",
            files={"file": ("mix.wav", f, "audio/wav")},
            data={"start_time": "2.0", "end_time": "6.0"},
        )
    assert res.status_code == 200
    assert res.json()["duration"] == pytest.approx(4.0, abs=0.01)
    assert res.json()["bpm"] == pytest.approx(120, abs=2)


def test_analyze_rejects_unsupported_type():
    res = client.post("/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 400
    assert "Unsupported" in res.json()["detail"]


def test_analyze_rejects_corrupt_audio_as_client_error():
    res = client.post("/analyze", files={"file": ("bad.wav", b"not audio" * 100, "audio/wav")})
    assert res.status_code == 400


def test_analyze_rejects_invalid_region(write_wav):
    with open(write_wav(np.zeros(44100)), "rb") as f:
        res = client.post("/analyze", files={"file": ("a.wav", f, "audio/wav")},
                          data={"start_time": "5", "end_time": "2"})
    assert res.status_code == 400


def test_analyze_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr("routes.analyze.MAX_FILE_SIZE", 10)
    res = client.post("/analyze", files={"file": ("a.wav", b"x" * 100, "audio/wav")})
    assert res.status_code == 413


def test_preview_converts_aiff_to_wav(tmp_path, click_track):
    path = tmp_path / "bounce.aiff"
    sf.write(path, np.column_stack([click_track(2), click_track(2)]), 44100, format="AIFF", subtype="PCM_24")
    res = client.post("/preview", files={"file": ("bounce.aiff", path.read_bytes(), "audio/aiff")})
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    info = sf.info(io.BytesIO(res.content))
    assert (info.format, info.channels, info.samplerate, info.frames) == ("WAV", 2, 44100, 88200)


def test_preview_rejects_corrupt_audio():
    assert client.post("/preview", files={"file": ("bad.aiff", b"not audio" * 50, "audio/aiff")}).status_code == 400


def test_audio_context_reports_zero_values_and_region():
    features = AudioFeatures(
        session_id="s", duration=4.0, sample_rate=44100, bpm=92.0, key="C minor",
        loudness_lufs=0.0, dynamic_range_db=0.0, timing_checked=True,
        timing_issues=[TimingIssue(timestamp=3.21, description="Onset ~38ms late")],
        frequency_profile=FrequencyProfile(low_energy=0.5, mid_energy=0.3, high_energy=0.2),
    )
    ctx = build_audio_context(features, AudioRegion(start=1.0, end=5.0))
    assert "1.0s – 5.0s" in ctx
    assert "0.0 LUFS" in ctx
    assert "0.0 dB" in ctx
    assert "3.21s" in ctx
    assert "Low (<250Hz) 50%" in ctx


def test_audio_context_says_when_timing_was_not_checked():
    features = AudioFeatures(session_id="s", duration=4.0, sample_rate=44100, timing_checked=False)
    assert "not checked" in build_audio_context(features, None)


def test_chat_streams_deltas_then_done(fake_anthropic):
    res = client.post("/chat", json={"session_id": "s", "message": "Too muddy"})
    assert res.headers["content-type"].startswith("text/event-stream")
    assert sse_events(res.text) == [
        {"type": "delta", "content": "Cut "},
        {"type": "delta", "content": "at 300Hz."},
        {"type": "done"},
    ]


def test_chat_sends_model_context_and_history(fake_anthropic):
    client.post("/chat", json={
        "session_id": "s",
        "message": "Now the chorus",
        "audio_features": {"session_id": "s", "duration": 4.0, "sample_rate": 44100, "bpm": 92.0},
        "region": {"start": 1.0, "end": 5.0},
        "history": [
            {"role": "user", "content": "Too muddy"},
            {"role": "assistant", "content": "Cut 300Hz."},
            {"role": "user", "content": "Interrupted question"},
            {"role": "assistant", "content": ""},
        ],
    })
    request = fake_anthropic["requests"][0]
    body = request["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in request["headers"].get("anthropic-beta", "")
    assert "Music Copilot" in str(body["system"])
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user", "user"]
    assert "BPM:** 92.0" in str(body["messages"][-1]["content"])
    assert "Now the chorus" in str(body["messages"][-1]["content"])


@pytest.mark.parametrize(
    "status,expected",
    [(401, "API key"), (429, "Rate limited"), (500, "(500)")],
)
def test_chat_reports_api_failures_as_error_events(fake_anthropic, status, expected):
    fake_anthropic["status"] = status
    events = sse_events(client.post("/chat", json={"session_id": "s", "message": "hi"}).text)
    assert events[-1]["type"] == "error"
    assert expected in events[-1]["content"]


def test_chat_reports_refusal_as_error(fake_anthropic):
    fake_anthropic["body"] = anthropic_stream([], stop_reason="refusal")
    events = sse_events(client.post("/chat", json={"session_id": "s", "message": "hi"}).text)
    assert events[-1]["type"] == "error"
    assert "declined" in events[-1]["content"]


def test_chat_rejects_empty_message():
    assert client.post("/chat", json={"session_id": "s", "message": ""}).status_code == 422
