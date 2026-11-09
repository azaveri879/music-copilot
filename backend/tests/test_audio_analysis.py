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
    features = analyze_audio(write_wav(y))
    assert features.key == expected
    assert features.key_confident


def noise_floor(seconds: float) -> np.ndarray:
    return 1e-4 * np.random.default_rng(0).standard_normal(int(seconds * SR))


def test_key_ignores_the_gaps_between_phrases(write_wav):
    phrases = [chord(c, 2.0) for c in [[261.63, 329.63, 392.0], [349.23, 440.0, 523.25], [392.0, 493.88, 587.33]]]
    y = np.concatenate([part for phrase in phrases for part in (phrase, noise_floor(8.0))])
    features = analyze_audio(write_wav(y))
    assert features.key == "C major"
    assert features.key_confident


def test_noise_has_no_confident_key(write_wav):
    y = 0.2 * np.random.default_rng(1).standard_normal(8 * SR)
    assert not analyze_audio(write_wav(y)).key_confident


def test_steady_part_has_a_confident_tempo(write_wav, click_track):
    assert analyze_audio(write_wav(click_track())).bpm_confident


def test_scattered_notes_have_no_confident_tempo(write_wav):
    y = np.zeros(30 * SR)
    for time in np.sort(np.random.default_rng(2).uniform(0.5, 29.0, 40)):
        click(y, float(time))
    assert not analyze_audio(write_wav(y)).bpm_confident


def test_silent_file_has_no_confident_readings(write_wav):
    features = analyze_audio(write_wav(np.zeros(4 * SR)))
    assert not features.bpm_confident
    assert not features.key_confident


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


def test_dynamic_range_ignores_the_gaps_between_phrases(write_wav):
    y = np.concatenate([tone(440, 2.0, 0.5), noise_floor(2.0)] * 3)
    assert analyze_audio(write_wav(y)).dynamic_range_db < 3


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


def strum(y: np.ndarray, time: float, freq: float, seed: int) -> None:
    """A noisy guitar-like attack, harder for a beat tracker than a clean click."""
    n = int(0.25 * SR)
    start = int(time * SR)
    t = np.arange(n) / SR
    noise = np.random.default_rng(seed).standard_normal(n)
    y[start : start + n] += (0.4 * np.sin(2 * np.pi * freq * t) + 0.15 * noise) * np.exp(-t / 0.06)


def loop_98_bpm(late_note_ms: float = 0.0, repeats: int = 16) -> np.ndarray:
    """A two-bar part at exactly 98 BPM with a syncopated rhythm, repeated sample for sample."""
    sixteenth = 60 / 98 / 4
    pattern = [0, 3, 6, 8, 10, 11, 14, 16, 19, 22, 24, 26, 30]
    clip = np.zeros(round(32 * sixteenth * SR))
    for i, step in enumerate(pattern):
        late = late_note_ms / 1000 if step == 16 else 0.0
        strum(clip, step * sixteenth + late, 196 + 40 * (i % 5), seed=i)
    return np.tile(clip, repeats)


def test_exact_loop_reports_its_true_tempo(write_wav):
    assert analyze_audio(write_wav(loop_98_bpm())).bpm == pytest.approx(98.0, abs=0.1)


def test_in_time_exact_loop_has_no_timing_issues(write_wav):
    assert analyze_audio(write_wav(loop_98_bpm())).timing_issues == []


def test_exact_loop_reports_a_late_note_once(write_wav):
    issues = analyze_audio(write_wav(loop_98_bpm(late_note_ms=40))).timing_issues
    assert len(issues) == 1
    assert "late" in issues[0].description
    # Beat 8 of the two-bar loop, 2.449s into a repeat, plus the 40ms push.
    assert (issues[0].timestamp - 2.489) % (8 * 60 / 98) == pytest.approx(0, abs=0.04)


def test_steady_tempo_with_a_different_part_each_bar_uses_the_true_tempo(write_wav):
    # Same grid but no exact repeats, so this relies on the tempo fit alone.
    sixteenth = 60 / 98 / 4
    rng = np.random.default_rng(7)
    y = np.zeros(SR * 40)
    step = 0
    while step * sixteenth < 39:
        strum(y, step * sixteenth, float(rng.choice([165, 196, 220, 262])), seed=step)
        step += int(rng.choice([1, 2, 2, 3, 4]))
    features = analyze_audio(write_wav(y))
    assert features.bpm == pytest.approx(98.0, abs=0.1)
    assert features.timing_issues == []


def test_free_tempo_playing_falls_back_without_a_rigid_grid(write_wav):
    # A gradual slowdown from 110 to 95 BPM can't fit any single tempo.
    y = np.zeros(SR * 40)
    time, beat = 0.0, 60 / 110
    while time < 39:
        click(y, time)
        time += beat
        beat *= 1.002
    features = analyze_audio(write_wav(y))
    assert features.bpm is not None
    assert features.timing_checked is False
    assert features.timing_issues == []


def test_reports_fitted_tempo_when_only_part_of_the_track_is_on_the_grid(write_wav):
    # A loose opening keeps timing unchecked, but the rest still gives away the true tempo.
    rng = np.random.default_rng(11)
    sixteenth = 60 / 98 / 4
    y = np.zeros(SR * 90)
    step = 0
    while step * sixteenth < 89:
        time = step * sixteenth
        loose = 0.03 * np.sin(time / 2) + 0.05 if time < 30 else 0.0
        strum(y, time + loose, float(rng.choice([165, 196, 220, 262])), seed=step)
        step += int(rng.choice([1, 2, 2, 3, 4]))
    features = analyze_audio(write_wav(y))
    assert features.timing_checked is False
    assert features.bpm == pytest.approx(98.0, abs=0.2)


def band(features, name):
    return next(b for b in features.band_levels if b.name == name)


def test_band_levels_show_an_empty_low_end(write_wav):
    features = analyze_audio(write_wav(tone(800, 4.0)))
    assert band(features, "mid").level_db == 0
    assert band(features, "sub").level_db < -40
    assert band(features, "bass").level_db < -40


def test_band_levels_find_real_sub_energy(write_wav):
    features = analyze_audio(write_wav(tone(45, 4.0) + tone(800, 4.0, 0.05)))
    assert band(features, "sub").level_db == 0
    assert band(features, "mid").level_db == pytest.approx(-20, abs=3)


def test_band_levels_ignore_the_gaps_between_phrases(write_wav):
    rumble = 0.0005 * tone(45, 8.0, 1.0)
    y = np.concatenate([tone(800, 2.0), rumble, tone(800, 2.0)])
    assert band(analyze_audio(write_wav(y)), "sub").level_db < -40


def test_prominent_frequencies_find_tones(write_wav):
    y = tone(200, 4.0, 0.3) + tone(600, 4.0, 0.3)
    assert analyze_audio(write_wav(y)).prominent_frequencies_hz == pytest.approx([200, 600], abs=8)


def test_noise_has_no_prominent_frequencies(write_wav):
    y = 0.2 * np.random.default_rng(3).standard_normal(20 * SR)
    assert analyze_audio(write_wav(y)).prominent_frequencies_hz == []


def test_peak_moments_locate_spikes_in_time_order(write_wav):
    y = tone(440, 8.0, 0.05)
    y[3 * SR] = 0.9
    y[6 * SR] = 0.8
    moments = analyze_audio(write_wav(y)).peak_moments
    assert [m.timestamp for m in moments] == pytest.approx([3.0, 6.0], abs=0.02)
    assert moments[0].level_dbfs == pytest.approx(-0.9, abs=0.2)
    assert str(moments[0].level_dbfs) == str(round(moments[0].level_dbfs, 1))
    assert len(str(moments[0].level_dbfs)) <= 5


def test_peak_moments_are_positions_in_the_file_for_a_region(write_wav):
    y = tone(440, 8.0, 0.05)
    y[5 * SR] = 0.9
    moments = analyze_audio(write_wav(y), start_time=4.0, end_time=7.0).peak_moments
    assert moments[0].timestamp == pytest.approx(5.0, abs=0.02)


def test_peak_moments_find_a_peak_at_the_edge_of_the_file(write_wav):
    y = tone(440, 4.0, 0.05)
    y[10] = 0.9
    assert analyze_audio(write_wav(y)).peak_moments[0].timestamp == pytest.approx(0.0, abs=0.02)


def test_silence_has_no_spectrum_detail_or_peaks(write_wav):
    features = analyze_audio(write_wav(np.zeros(4 * SR)))
    assert features.band_levels == []
    assert features.prominent_frequencies_hz == []
    assert features.peak_moments == []
