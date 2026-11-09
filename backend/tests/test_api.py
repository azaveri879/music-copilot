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
    state = {"status": 200, "body": anthropic_stream(["Cut ", "at 300Hz."]), "queue": [], "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
            state["requests"].append({"path": self.path, "headers": dict(self.headers), "body": payload})
            self.send_response(state["status"])
            is_ok = state["status"] == 200
            self.send_header("content-type", "text/event-stream" if is_ok else "application/json")
            self.end_headers()
            body = state["queue"].pop(0) if state["queue"] else state["body"]
            self.wfile.write(body if is_ok else json.dumps(
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


def test_audio_context_withholds_unreliable_tempo_and_key():
    features = AudioFeatures(session_id="s", duration=4.0, sample_rate=44100, bpm=165.4, key="D# minor")
    ctx = build_audio_context(features, None)
    assert "165" not in ctx
    assert "D# minor" not in ctx
    assert "**BPM:** not reliably detected" in ctx
    assert "**Key:** not reliably detected" in ctx


def test_audio_context_shows_confident_tempo_and_key():
    features = AudioFeatures(
        session_id="s", duration=4.0, sample_rate=44100, bpm=98.0, bpm_confident=True, key="A# major", key_confident=True
    )
    ctx = build_audio_context(features, None)
    assert "**BPM:** 98.0" in ctx
    assert "**Key:** A# major" in ctx


def test_audio_context_prefers_the_users_tempo():
    features = AudioFeatures(session_id="s", duration=4.0, sample_rate=44100, bpm=165.4, bpm_confident=True)
    assert "**BPM:** 98 (set by the user)" in build_audio_context(features, None, song_bpm=98.0)


@pytest.mark.parametrize(
    "file_type,expected",
    [("track", "single track"), ("mix", "full mix"), ("master", "finished master"), (None, "not specified")],
)
def test_audio_context_says_what_the_file_is(file_type, expected):
    features = AudioFeatures(session_id="s", duration=4.0, sample_rate=44100)
    assert expected in build_audio_context(features, None, file_type=file_type).split("\n")[1]


def test_chat_passes_file_type_and_song_tempo_to_claude(fake_anthropic):
    client.post("/chat", json={
        "session_id": "s",
        "message": "Too muddy",
        "audio_features": {"session_id": "s", "duration": 4.0, "sample_rate": 44100},
        "file_type": "track",
        "song_bpm": 98,
    })
    content = str(fake_anthropic["requests"][0]["body"]["messages"][-1]["content"])
    assert "single track" in content
    assert "98 (set by the user)" in content


@pytest.mark.parametrize("field", [{"file_type": "stem"}, {"song_bpm": 5}])
def test_chat_rejects_invalid_context(field):
    assert client.post("/chat", json={"session_id": "s", "message": "Hi", **field}).status_code == 422


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
        "audio_features": {"session_id": "s", "duration": 4.0, "sample_rate": 44100, "bpm": 92.0, "bpm_confident": True},
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


def anthropic_tool_stream(text: str, tool_input: str) -> bytes:
    """A reply that writes some text and then calls propose_edits, in the wire format."""
    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {
            "type": "tool_use", "id": "toolu_1", "name": "propose_edits", "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                 "delta": {"type": "input_json_delta", "partial_json": tool_input}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                           "usage": {"output_tokens": 5}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


AUDIO = {"session_id": "s", "duration": 4.0, "sample_rate": 44100, "bpm": 92.0}


def analyzed_audio() -> dict:
    """Analyze a loud test tone, so the backend holds the audio for measuring."""
    y = 0.5 * np.sin(2 * np.pi * 440 * np.arange(44100 * 3) / 44100)
    return client.post("/analyze", files=wav_upload(y)).json()


GAIN_PROPOSAL = {"summary": "Turn it down", "edits": [{"type": "gain", "reason": "", "gain_db": -6}]}


def test_proposal_is_measured_on_the_real_audio_and_returned_to_claude(fake_anthropic):
    fake_anthropic["queue"] = [
        anthropic_tool_stream("Turning it down.", json.dumps(GAIN_PROPOSAL)),
        anthropic_stream(["Measured ", "6 dB quieter."]),
    ]
    res = client.post("/chat", json={"session_id": "s", "message": "Quieter", "audio_features": analyzed_audio()})
    events = sse_events(res.text)

    assert [e["type"] for e in events] == ["delta", "delta", "delta", "delta", "proposal", "done"]
    assert "".join(e["content"] for e in events if e["type"] == "delta") == "Turning it down.\n\nMeasured 6 dB quieter."
    assert events[-2]["proposal"]["summary"] == "Turn it down"

    second = fake_anthropic["requests"][1]["body"]["messages"]
    assert second[-2]["role"] == "assistant"
    result = second[-1]["content"][0]
    assert result["type"] == "tool_result"
    measured = json.loads(result["content"].split("whole file: ")[1].split("}.")[0] + "}")
    assert measured["after_lufs"] - measured["before_lufs"] == pytest.approx(-6, abs=0.1)
    assert measured["after_peak_dbfs"] - measured["before_peak_dbfs"] == pytest.approx(-6, abs=0.1)


def test_claude_can_revise_and_the_user_sees_only_the_last_proposal(fake_anthropic):
    revised = {"summary": "Gentler", "edits": [{"type": "gain", "reason": "", "gain_db": -2}]}
    fake_anthropic["queue"] = [
        anthropic_tool_stream("Trying this.", json.dumps(GAIN_PROPOSAL)),
        anthropic_tool_stream("Too much, revising.", json.dumps(revised)),
        anthropic_stream(["Done."]),
    ]
    res = client.post("/chat", json={"session_id": "s", "message": "Quieter", "audio_features": analyzed_audio()})
    proposals = [e for e in sse_events(res.text) if e["type"] == "proposal"]
    assert [p["proposal"]["summary"] for p in proposals] == ["Gentler"]


def test_revisions_stop_after_the_limit(fake_anthropic):
    fake_anthropic["body"] = anthropic_tool_stream("Again.", json.dumps(GAIN_PROPOSAL))
    res = client.post("/chat", json={"session_id": "s", "message": "Quieter", "audio_features": analyzed_audio()})
    assert len(fake_anthropic["requests"]) == 3
    assert fake_anthropic["requests"][-1]["body"]["tool_choice"] == {"type": "none"}
    assert sum(e["type"] == "proposal" for e in sse_events(res.text)) == 1


def test_invalid_proposal_goes_back_to_claude_and_never_reaches_the_user(fake_anthropic):
    fake_anthropic["queue"] = [
        anthropic_tool_stream("Louder!", json.dumps({
            "summary": "Way louder", "edits": [{"type": "gain", "reason": "", "gain_db": 90}],
        })),
        anthropic_stream(["Sorry, that was out of range."]),
    ]
    res = client.post("/chat", json={"session_id": "s", "message": "Louder", "audio_features": analyzed_audio()})
    assert not any(e["type"] == "proposal" for e in sse_events(res.text))
    result = fake_anthropic["requests"][1]["body"]["messages"][-1]["content"][0]
    assert result["is_error"] is True


def test_proposal_still_shown_when_the_audio_is_no_longer_in_memory(fake_anthropic):
    fake_anthropic["queue"] = [
        anthropic_tool_stream("Turning it down.", json.dumps(GAIN_PROPOSAL)),
        anthropic_stream(["Can't quote numbers."]),
    ]
    res = client.post("/chat", json={"session_id": "s", "message": "Quieter", "audio_features": AUDIO})
    assert any(e["type"] == "proposal" for e in sse_events(res.text))
    result = fake_anthropic["requests"][1]["body"]["messages"][-1]["content"][0]
    assert "Don't quote numbers" in result["content"]


def test_estimate_measures_without_changing_anything():
    y = 0.5 * np.sin(2 * np.pi * 440 * np.arange(44100 * 3) / 44100)
    res = client.post("/edit/estimate", files=wav_upload(y), data={"edits": json.dumps([{"type": "gain", "gain_db": -6}])})
    assert res.status_code == 200
    body = res.json()
    assert body["after_lufs"] - body["before_lufs"] == pytest.approx(-6, abs=0.1)
    assert body["before_peak_dbfs"] == pytest.approx(-6, abs=0.1)
    assert body["turned_down_db"] == 0


def test_estimate_reports_being_turned_down_to_avoid_clipping():
    y = 0.9 * np.sin(2 * np.pi * 440 * np.arange(44100 * 3) / 44100)
    res = client.post("/edit/estimate", files=wav_upload(y), data={"edits": json.dumps([{"type": "gain", "gain_db": 12}])})
    assert res.json()["turned_down_db"] > 0


def test_estimate_covers_only_the_region():
    y = 0.5 * np.sin(2 * np.pi * 440 * np.arange(44100 * 4) / 44100)
    res = client.post("/edit/estimate", files=wav_upload(y), data={
        "edits": json.dumps([{"type": "gain", "gain_db": -6}]), "start_time": "1.0", "end_time": "3.0",
    })
    assert res.json()["after_lufs"] - res.json()["before_lufs"] == pytest.approx(-6, abs=0.2)


def test_chat_only_offers_editing_when_there_is_audio(fake_anthropic):
    client.post("/chat", json={"session_id": "s", "message": "hi"})
    assert not fake_anthropic["requests"][0]["body"].get("tools")
    client.post("/chat", json={"session_id": "s", "message": "hi", "audio_features": AUDIO})
    tools = fake_anthropic["requests"][1]["body"]["tools"]
    assert tools[0]["name"] == "propose_edits"
    assert tools[0]["eager_input_streaming"] is True


def wav_upload(y: np.ndarray) -> dict:
    buffer = io.BytesIO()
    sf.write(buffer, y, 44100, format="WAV")
    return {"file": ("take.wav", buffer.getvalue(), "audio/wav")}


def test_analyze_reports_peak_level():
    y = 0.5 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
    assert client.post("/analyze", files=wav_upload(y)).json()["peak_dbfs"] == pytest.approx(-6, abs=0.1)


def test_audio_context_includes_peak_level():
    features = AudioFeatures(session_id="s", duration=4.0, sample_rate=44100, peak_dbfs=-0.1)
    assert "**Peak:** -0.1 dBFS" in build_audio_context(features, None)


def test_edit_returns_wav_and_report():
    y = 0.1 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
    res = client.post("/edit", files=wav_upload(y), data={"edits": json.dumps([{"type": "gain", "gain_db": 6}])})
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert json.loads(res.headers["x-edit-report"])["nudges_applied"] == 0
    out, sr = sf.read(io.BytesIO(res.content))
    assert sr == 44100
    assert np.abs(out).max() == pytest.approx(0.1995, abs=0.002)


def test_edit_uses_the_region():
    y = 0.1 * np.sin(2 * np.pi * 440 * np.arange(44100 * 2) / 44100)
    res = client.post("/edit", files=wav_upload(y), data={
        "edits": json.dumps([{"type": "gain", "gain_db": -12}]), "start_time": "1.0", "end_time": "2.0",
    })
    out, _ = sf.read(io.BytesIO(res.content))
    assert np.abs(out[:40000]).max() == pytest.approx(0.1, abs=0.001)
    assert np.abs(out[50000:87000]).max() == pytest.approx(0.0251, abs=0.001)


@pytest.mark.parametrize("edits", ["not json", json.dumps([{"type": "gain", "gain_db": 99}]), "[]"])
def test_edit_rejects_bad_edits(edits):
    y = np.zeros(44100)
    assert client.post("/edit", files=wav_upload(y), data={"edits": edits}).status_code == 400


def test_edit_rejects_a_nudge_with_no_note():
    y = np.zeros(44100)
    res = client.post("/edit", files=wav_upload(y), data={
        "edits": json.dumps([{"type": "nudge", "onset_time": 0.5, "shift_ms": 20}]),
    })
    assert res.status_code == 400
    assert "Couldn't move" in res.json()["detail"]


def test_chat_rejects_empty_message():
    assert client.post("/chat", json={"session_id": "s", "message": ""}).status_code == 422
