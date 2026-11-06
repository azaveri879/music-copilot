import numpy as np
import pytest

from services.audio_analysis import (
    AudioDecodeError,
    InvalidRegionError,
    analyze_audio,
)
from tests.conftest import SR, click


def tone(freq: float, seconds: float, amplitude: float = 0.5) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * freq * np.arange(int(seconds * SR)) / SR)


def chord(freqs: list[float], seconds: float) -> np.ndarray:
    return sum(tone(f, seconds, 0.2) for f in freqs)


def test_detects_tempo(write_wav, click_track):
    assert analyze_audio(write_wav(click_track())).bpm == pytest.approx(120, abs=2)


def test_in_time_track_has_no_timing_issues(write_wav, click_track):
    assert analyze_audio(write_wav(click_track())).timing_issues == []


def test_in_time_offbeat_notes_are_not_flagged(write_wav):
    y = np.zeros(SR * 16)
    for i in range(62):
        click(y, i * 0.25)  # straight 8ths at 120 BPM
    assert analyze_audio(write_wav(y)).timing_issues == []


def test_swung_notes_are_not_flagged(write_wav):
    y = np.zeros(SR * 16)
    for beat in range(30):
        click(y, beat * 0.5)
        click(y, beat * 0.5 + 0.5 * 2 / 3, 700)
    assert analyze_audio(write_wav(y)).timing_issues == []


def test_flags_a_late_note_with_position_and_direction(write_wav, click_track):
    issues = analyze_audio(write_wav(click_track(offsets={12: 0.045}))).timing_issues
    assert len(issues) == 1
    assert issues[0].timestamp == pytest.approx(6.045, abs=0.03)
    assert "late" in issues[0].description


def test_flags_an_early_note(write_wav, click_track):
    issues = analyze_audio(write_wav(click_track(offsets={20: -0.045}))).timing_issues
    assert len(issues) == 1
    assert issues[0].timestamp == pytest.approx(9.955, abs=0.03)
    assert "early" in issues[0].description


def test_region_limits_timing_issues_and_keeps_file_timestamps(write_wav, click_track):
    path = write_wav(click_track(offsets={4: 0.045, 24: 0.045}))
    issues = analyze_audio(path, start_time=10.0, end_time=14.0).timing_issues
    assert len(issues) == 1
    assert issues[0].timestamp == pytest.approx(12.045, abs=0.03)


def test_short_region_still_reports_track_tempo(write_wav, click_track):
    features = analyze_audio(write_wav(click_track()), start_time=5.0, end_time=6.0)
    assert features.duration == pytest.approx(1.0, abs=0.01)
    assert features.bpm == pytest.approx(120, abs=2)


@pytest.mark.parametrize(
    "chords,expected",
    [
        # I-IV-V-I in C major
        ([[261.63, 329.63, 392.0], [349.23, 440.0, 523.25], [392.0, 493.88, 587.33], [261.63, 329.63, 392.0]], "C major"),
        # i-iv-v-i in A minor
        ([[220.0, 261.63, 329.63], [293.66, 349.23, 440.0], [329.63, 392.0, 493.88], [220.0, 261.63, 329.63]], "A minor"),
    ],
)
def test_detects_key(write_wav, chords, expected):
    y = np.concatenate([chord(c, 2.0) for c in chords] * 2)
    assert analyze_audio(write_wav(y)).key == expected


def test_region_key_describes_the_region_not_the_track(write_wav):
    c_major = np.concatenate([chord(c, 2.0) for c in [[261.63, 329.63, 392.0], [349.23, 440.0, 523.25], [392.0, 493.88, 587.33]]])
    f_sharp = np.concatenate([chord(c, 2.0) for c in [[369.99, 466.16, 554.37], [493.88, 622.25, 739.99], [277.18, 349.23, 415.30]]])
    features = analyze_audio(write_wav(np.concatenate([c_major, f_sharp])), start_time=6.0, end_time=12.0)
    assert features.key == "F# major"


def test_loudness_matches_bs1770_reference(write_wav):
    # BS.1770: a full-scale 997Hz sine in both stereo channels reads 0 LUFS, -20 dBFS reads -20.
    sine = tone(997, 5.0, amplitude=0.1)
    features = analyze_audio(write_wav(np.column_stack([sine, sine])))
    assert features.loudness_lufs == pytest.approx(-20.0, abs=0.5)


def test_loudness_is_none_when_too_short_to_measure(write_wav):
    assert analyze_audio(write_wav(tone(440, 0.2))).loudness_lufs is None


def test_dynamic_range_separates_steady_from_dynamic_audio(write_wav):
    steady = analyze_audio(write_wav(tone(440, 6.0), "steady.wav")).dynamic_range_db
    dynamic = analyze_audio(
        write_wav(np.concatenate([tone(440, 3.0, 0.5), tone(440, 3.0, 0.05)]), "dynamic.wav")
    ).dynamic_range_db
    assert steady < 1
    assert dynamic == pytest.approx(20, abs=2)


def test_frequency_profile_follows_the_content(write_wav):
    bass = analyze_audio(write_wav(tone(80, 3.0), "bass.wav")).frequency_profile
    air = analyze_audio(write_wav(tone(8000, 3.0), "air.wav")).frequency_profile
    assert bass.low_energy > 0.8
    assert air.high_energy > 0.8


def test_silence_reports_nothing_instead_of_failing(write_wav):
    features = analyze_audio(write_wav(np.zeros(SR * 3)))
    assert features.duration == 3.0
    assert features.bpm is None
    assert features.key is None
    assert features.loudness_lufs is None
    assert features.frequency_profile is None
    assert features.timing_issues == []


def test_corrupt_file_raises_decode_error(tmp_path):
    path = tmp_path / "bad.wav"
    path.write_bytes(b"this is not audio" * 100)
    with pytest.raises(AudioDecodeError):
        analyze_audio(str(path))


@pytest.mark.parametrize("start,end", [(5.0, 5.0), (6.0, 5.0), (-1.0, 1.0), (100.0, 110.0)])
def test_invalid_region_raises(write_wav, start, end):
    with pytest.raises(InvalidRegionError):
        analyze_audio(write_wav(tone(440, 10.0)), start_time=start, end_time=end)


def test_notes_between_grid_lines_are_not_called_mistimed(write_wav, click_track):
    # Pickups halfway between 16ths at 120 BPM have no clear intended position,
    # so "late" or "early" means nothing for them.
    y = click_track()
    for beat in [3, 9, 15, 21, 27, 33]:
        click(y, beat * 0.5 + 0.0625, 700)
    assert analyze_audio(write_wav(y)).timing_issues == []


def test_real_error_is_found_among_notes_between_grid_lines(write_wav, click_track):
    y = click_track(offsets={12: 0.045})
    for beat in [3, 9, 15, 21, 27, 33]:
        click(y, beat * 0.5 + 0.0625, 700)
    issues = analyze_audio(write_wav(y)).timing_issues
    assert len(issues) == 1
    assert issues[0].timestamp == pytest.approx(6.045, abs=0.03)
    assert "late" in issues[0].description


def test_quiet_notes_are_not_flagged(write_wav, click_track):
    # A ghost note slightly off the grid shouldn't be reported as a mistake.
    y = click_track()
    ghost = np.zeros_like(y)
    click(ghost, 6.0 + 0.04, 700)
    assert analyze_audio(write_wav(y + 0.3 * ghost)).timing_issues == []
