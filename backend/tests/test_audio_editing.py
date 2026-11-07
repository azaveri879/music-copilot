import io

import numpy as np
import pytest
import soundfile as sf

from models.schemas import AudioEditList, AudioRegion
from services.audio_analysis import AudioDecodeError, analyze_audio, detect_onsets
from services.audio_editing import EditError, apply_edits
from tests.conftest import SR, click


def to_bytes(y: np.ndarray, sr: int = SR) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format="WAV", subtype="PCM_24")
    return buffer.getvalue()


def edits(*items: dict) -> list:
    return AudioEditList.validate_python(list(items))


def run(y: np.ndarray, *items: dict, region: AudioRegion | None = None, sr: int = SR):
    wav, report = apply_edits(to_bytes(y, sr), edits(*items), region)
    out, out_sr = sf.read(io.BytesIO(wav), dtype="float64")
    assert out_sr == sr
    return out, report


def tone(freq: float, seconds: float = 2.0, amplitude: float = 0.1) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * freq * np.arange(int(seconds * SR)) / SR)


def level_db(y: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(y**2)))


def response_db(freq: float, *items: dict) -> float:
    """Measured change at one frequency, skipping the filter's settling time."""
    y = tone(freq)
    out, _ = run(y, *items)
    return level_db(out[SR // 2 :]) - level_db(y[SR // 2 :])


def test_gain_changes_level_exactly():
    assert response_db(1000, {"type": "gain", "gain_db": -6}) == pytest.approx(-6, abs=0.01)


def test_output_matches_input_length_and_channels():
    stereo = np.column_stack([tone(440), tone(660)])
    out, _ = run(stereo, {"type": "gain", "gain_db": -3})
    assert out.shape == stereo.shape


def test_peak_eq_hits_its_gain_only_near_its_frequency():
    eq = {"type": "eq", "eq_type": "peak", "freq_hz": 1000, "gain_db": 6, "q": 2}
    assert response_db(1000, eq) == pytest.approx(6, abs=0.1)
    assert response_db(100, eq) == pytest.approx(0, abs=0.2)
    assert response_db(10000, eq) == pytest.approx(0, abs=0.2)


def test_high_shelf_affects_highs_and_leaves_lows():
    eq = {"type": "eq", "eq_type": "high_shelf", "freq_hz": 2000, "gain_db": -6}
    assert response_db(15000, eq) == pytest.approx(-6, abs=0.3)
    assert response_db(100, eq) == pytest.approx(0, abs=0.2)


def test_low_shelf_affects_lows_and_leaves_highs():
    eq = {"type": "eq", "eq_type": "low_shelf", "freq_hz": 200, "gain_db": 6}
    assert response_db(40, eq) == pytest.approx(6, abs=0.3)
    assert response_db(5000, eq) == pytest.approx(0, abs=0.2)


def test_highpass_cuts_below_and_passes_above():
    eq = {"type": "eq", "eq_type": "highpass", "freq_hz": 200}
    assert response_db(200, eq) == pytest.approx(-3, abs=0.2)
    assert response_db(50, eq) < -20
    assert response_db(5000, eq) == pytest.approx(0, abs=0.2)


def test_lowpass_cuts_above_and_passes_below():
    eq = {"type": "eq", "eq_type": "lowpass", "freq_hz": 2000}
    assert response_db(2000, eq) == pytest.approx(-3, abs=0.2)
    assert response_db(15000, eq) < -20
    assert response_db(100, eq) == pytest.approx(0, abs=0.2)


def test_eq_above_the_files_limit_is_rejected():
    y = np.zeros(22050)
    with pytest.raises(EditError):
        apply_edits(to_bytes(y, 22050), edits({"type": "eq", "eq_type": "peak", "freq_hz": 15000, "gain_db": 3}))


def test_compressor_turns_down_loud_audio_and_leaves_quiet_audio():
    loud = tone(440, 2.0, 0.5)
    quiet = tone(440, 2.0, 0.01)
    comp = {"type": "compressor", "threshold_db": -20, "ratio": 4, "attack_ms": 5, "release_ms": 50}
    out, _ = run(np.concatenate([loud, quiet]), comp)
    # 14dB over the threshold at 4:1 comes down by 10.5dB.
    assert level_db(out[SR : 2 * SR]) - level_db(loud[SR:]) == pytest.approx(-10.5, abs=0.5)
    assert level_db(out[3 * SR :]) - level_db(quiet[SR:]) == pytest.approx(0, abs=0.1)


def test_compressor_makeup_gain_is_applied():
    y = tone(440, 2.0, 0.01)
    out, _ = run(y, {"type": "compressor", "threshold_db": -6, "ratio": 2, "makeup_db": 6})
    assert level_db(out) - level_db(y) == pytest.approx(6, abs=0.05)


def test_region_edit_leaves_the_rest_of_the_file_alone():
    y = tone(440, 6.0)
    out, _ = run(y, {"type": "gain", "gain_db": -12}, region=AudioRegion(start=2.0, end=4.0))
    assert out[: 2 * SR] == pytest.approx(y[: 2 * SR], abs=1e-6)
    assert out[4 * SR :] == pytest.approx(y[4 * SR :], abs=1e-6)
    assert level_db(out[int(2.5 * SR) : int(3.5 * SR)]) - level_db(y[: SR]) == pytest.approx(-12, abs=0.05)


def test_region_edges_do_not_click():
    y = tone(440, 6.0)
    out, _ = run(y, {"type": "gain", "gain_db": -24}, region=AudioRegion(start=2.0, end=4.0))
    assert np.abs(np.diff(out)).max() <= np.abs(np.diff(y)).max() * 1.1


def test_boost_that_would_clip_is_turned_down_and_reported():
    y = tone(440, 1.0, 0.9)
    out, report = run(y, {"type": "gain", "gain_db": 12})
    assert np.abs(out).max() < 1.0
    assert report.turned_down_db > 0
    assert report.peak_dbfs == pytest.approx(-0.3, abs=0.05)


def test_no_edits_is_an_error():
    with pytest.raises(EditError):
        apply_edits(to_bytes(tone(440)), [])


def test_corrupt_audio_is_a_decode_error():
    with pytest.raises(AudioDecodeError):
        apply_edits(b"not audio" * 50, edits({"type": "gain", "gain_db": 1}))


@pytest.mark.parametrize("bad", [
    {"type": "gain", "gain_db": 100},
    {"type": "eq", "eq_type": "peak", "freq_hz": 5},
    {"type": "nudge", "onset_time": 1.0, "shift_ms": 500},
    {"type": "reverse"},
])
def test_out_of_range_edits_are_rejected(bad):
    with pytest.raises(ValueError):
        edits(bad)


def notes_track(displaced: dict[int, float] | None = None) -> np.ndarray:
    """Notes every half second, each with a decaying body so a move is measurable."""
    y = np.zeros(SR * 20)
    for beat in range(38):
        start = int((beat * 0.5 + (displaced or {}).get(beat, 0.0)) * SR)
        t = np.arange(int(0.35 * SR)) / SR
        y[start : start + t.size] += 0.6 * np.sin(2 * np.pi * (220 + 20 * (beat % 5)) * t) * np.exp(-t / 0.08)
    return y


@pytest.mark.parametrize("error_s,direction", [(0.045, "late"), (-0.045, "early")])
def test_nudge_corrects_a_mistimed_note(write_wav, error_s, direction):
    y = notes_track({12: error_s})
    assert direction in analyze_audio(write_wav(y, "before.wav")).timing_issues[0].description

    out, report = run(y, {"type": "nudge", "onset_time": 6.0 + error_s, "shift_ms": -error_s * 1000})

    assert report.nudges_applied == 1
    assert out.shape == y.shape
    onsets, _ = detect_onsets(out.astype(np.float32), SR)
    assert np.abs(onsets - 6.0).min() < 0.012
    assert analyze_audio(write_wav(out, "after.wav")).timing_issues == []


def test_nudge_only_changes_audio_around_that_note():
    y = notes_track({12: 0.045})
    out, _ = run(y, {"type": "nudge", "onset_time": 6.045, "shift_ms": -45})
    assert out[: int(5.9 * SR)] == pytest.approx(y[: int(5.9 * SR)], abs=1e-6)
    assert out[int(6.55 * SR) :] == pytest.approx(y[int(6.55 * SR) :], abs=1e-6)


@pytest.mark.parametrize("shift_ms", [40, -40])
def test_nudge_does_not_click(shift_ms):
    y = notes_track()
    out, _ = run(y, {"type": "nudge", "onset_time": 6.0, "shift_ms": shift_ms})
    window = slice(int(5.8 * SR), int(6.6 * SR))
    assert np.abs(np.diff(out[window])).max() < np.abs(np.diff(y[window])).max() * 1.5


@pytest.mark.parametrize("shift_ms", [40, -40])
def test_nudge_leaves_no_double_attack(shift_ms):
    # The old attack must be gone from where it was, not just copied to the new spot.
    y = notes_track({12: 0.03})
    out, _ = run(y, {"type": "nudge", "onset_time": 6.03, "shift_ms": shift_ms})
    before, _ = detect_onsets(y.astype(np.float32), SR)
    after, _ = detect_onsets(out.astype(np.float32), SR)
    assert len(after[(after > 5.8) & (after < 6.4)]) <= len(before[(before > 5.8) & (before < 6.4)])
    assert np.abs(after - (before[np.abs(before - 6.03).argmin()] + shift_ms / 1000)).min() < 0.012


def test_nudge_keeps_the_next_notes_attack_intact():
    y = notes_track({12: 0.03})
    out, _ = run(y, {"type": "nudge", "onset_time": 6.03, "shift_ms": 30})
    assert out[int(6.5 * SR) : int(6.6 * SR)] == pytest.approx(y[int(6.5 * SR) : int(6.6 * SR)], abs=1e-6)


def test_nudge_with_no_note_nearby_is_an_error():
    y = tone(440, 3.0)
    with pytest.raises(EditError):
        run(y, {"type": "nudge", "onset_time": 1.5, "shift_ms": 20})


def test_nudge_in_an_exact_loop_moves_every_repeat():
    # Each repeat holds the same late note, so all of them should move and still match.
    beat = 60 / 98
    clip = np.zeros(round(8 * beat * SR))
    for i in range(8):
        late = 0.04 if i == 2 else 0.0
        click(clip, i * beat + late)
    y = np.tile(clip, 8)
    out, report = run(y, {"type": "nudge", "onset_time": 2 * beat + 0.04, "shift_ms": -40})
    assert report.loop_repeats_nudged
    assert report.nudges_applied == 8
    onsets, _ = detect_onsets(out.astype(np.float32), SR)
    for repeat in range(8):
        expected = (repeat * 8 + 2) * beat
        assert np.abs(onsets - expected).min() < 0.015


def test_nudge_runs_before_tone_edits_regardless_of_order():
    y = notes_track({12: 0.045})
    out, report = run(
        y,
        {"type": "gain", "gain_db": -6},
        {"type": "nudge", "onset_time": 6.045, "shift_ms": -45},
    )
    assert report.nudges_applied == 1
    assert level_db(out) - level_db(y) == pytest.approx(-6, abs=0.1)


def test_limiter_holds_the_ceiling_exactly():
    y = tone(440, 3.0, 0.5)
    out, _ = run(y, {"type": "limiter", "ceiling_db": -1, "input_gain_db": 12})
    assert 20 * np.log10(np.abs(out).max()) <= -0.99


def test_limiter_raises_loudness_where_gain_alone_cannot(write_wav):
    # A loud body with short peaks at full scale, like a mastered mix.
    y = tone(220, 6.0, 0.25)
    for start in range(0, 6 * SR, SR // 2):
        y[start : start + 200] = 0.99
    limited, report = run(y, {"type": "limiter", "ceiling_db": -1, "input_gain_db": 6})
    gain_only, gain_report = run(y, {"type": "gain", "gain_db": 6})

    assert report.turned_down_db == 0
    assert gain_report.turned_down_db > 0
    def lufs(signal, name):
        return analyze_audio(write_wav(signal, name)).loudness_lufs
    assert lufs(limited, "a.wav") - lufs(y, "b.wav") > 4.5
    assert lufs(limited, "a.wav") > lufs(gain_only, "c.wav") + 4


def test_limiter_leaves_audio_under_the_ceiling_alone():
    y = tone(440, 2.0, 0.2)
    out, _ = run(y, {"type": "limiter", "ceiling_db": -1})
    assert out == pytest.approx(y, abs=1e-6)


def test_limiter_recovers_after_a_peak():
    y = tone(440, 3.0, 0.2)
    y[SR : SR + 100] = 0.99
    out, _ = run(y, {"type": "limiter", "ceiling_db": -6, "release_ms": 50})
    assert level_db(out[2 * SR :]) - level_db(y[2 * SR :]) == pytest.approx(0, abs=0.05)


def test_limiter_does_not_distort_a_steady_tone_when_pushed():
    y = tone(1000, 2.0, 0.5)
    out, _ = run(y, {"type": "limiter", "ceiling_db": -1, "input_gain_db": 6})
    spectrum = np.abs(np.fft.rfft(out[SR // 2 : SR // 2 + SR] * np.hanning(SR)))
    fundamental = spectrum[1000]
    harmonics = max(spectrum[2000], spectrum[3000], spectrum[5000])
    assert 20 * np.log10(harmonics / fundamental) < -30


def test_limiter_runs_last_whatever_order_it_was_given():
    y = tone(440, 2.0, 0.5)
    out, _ = run(y, {"type": "limiter", "ceiling_db": -3}, {"type": "gain", "gain_db": 6})
    assert 20 * np.log10(np.abs(out).max()) <= -2.99
