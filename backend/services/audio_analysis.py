import io
import logging
import uuid

import librosa
import numpy as np
import pyloudnorm
import soundfile as sf
from scipy.signal import medfilt

from models.schemas import AudioFeatures, FrequencyProfile, TimingIssue

logger = logging.getLogger(__name__)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

MIN_REGION_SECONDS = 0.1
SILENCE_RMS = 1e-5

# Positions within a beat that count as "on the grid": straight 16ths plus
# triplets, so swung or triplet parts aren't reported as mistimed.
GRID_FRACTIONS = np.array([0, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4])
TIMING_THRESHOLD_MS = 30
# A note this far towards the midpoint of two grid lines (0 = on the line, 1 = at
# the midpoint) has no clear intended position, so it can't be called late or early.
TIMING_MAX_AMBIGUITY = 0.8
# Quieter attacks, relative to the track's median attack, are likely ghost notes or noise.
TIMING_MIN_RELATIVE_STRENGTH = 0.5
TIMING_MIN_SPACING_S = 0.5
MAX_TIMING_ISSUES = 10

ONSET_HOP_LENGTH = 256  # about 5ms at 48kHz, so offsets aren't measured in coarse steps
MIN_ONSETS_FOR_TEMPO_FIT = 16
TEMPO_SEARCH_RANGE = 0.06  # stay near the tracker's tempo, which sets the half/double-time choice
MAX_PLAUSIBLE_BPM = 180  # above this, the tracker has almost certainly doubled the tempo
# A real rhythmic pattern has to line up far better than random onsets would. Random
# onsets average about 1/sqrt(n) on the fit score.
TEMPO_FIT_MIN_TIMES_CHANCE = 4
# A rigid grid is only trusted if local stretches of the track stay this close to it.
# Without one there's no reliable reference to judge timing against.
GRID_DRIFT_WINDOW_SECONDS = 8.0
GRID_DRIFT_MIN_WINDOW_ONSETS = 4
GRID_MAX_DRIFT_MS = 15.0
# Correlation between repeats that only an exact digital loop reaches.
EXACT_LOOP_MIN_CORRELATION = 0.995
EXACT_LOOP_MIN_SECONDS = 1.0
EXACT_LOOP_BLOCK = 16


class AudioDecodeError(ValueError):
    """The uploaded file could not be read as audio."""


class InvalidRegionError(ValueError):
    """The requested region doesn't describe a usable part of the file."""


def to_playable_wav(content: bytes) -> bytes:
    """Re-encode audio bytes as 16-bit WAV, which every browser can decode."""
    try:
        data, sr = sf.read(io.BytesIO(content), dtype="int16")
    except sf.LibsndfileError as exc:
        raise AudioDecodeError("Couldn't read this file as audio. It may be corrupt or an unsupported format.") from exc

    buffer = io.BytesIO()
    sf.write(buffer, data, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def is_silent(y: np.ndarray) -> bool:
    return y.size == 0 or float(np.sqrt(np.mean(y**2))) < SILENCE_RMS


def detect_key(y: np.ndarray, sr: int) -> str | None:
    """Estimate the musical key with the Krumhansl-Schmuckler algorithm."""
    if is_silent(y):
        return None

    chroma_mean = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    if np.allclose(chroma_mean, chroma_mean[0]):
        return None  # no pitch content to correlate against

    best_key, best_mode, best_corr = None, None, -np.inf
    for tonic in range(12):
        for profile, mode in [(MAJOR_PROFILE, "major"), (MINOR_PROFILE, "minor")]:
            corr = np.corrcoef(chroma_mean, np.roll(profile, tonic))[0, 1]
            if corr > best_corr:
                best_key, best_mode, best_corr = tonic, mode, corr

    return f"{NOTE_NAMES[best_key]} {best_mode}" if best_key is not None else None


def detect_onsets(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Return onset times in seconds and the strength of each attack."""
    envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=ONSET_HOP_LENGTH)
    frames = librosa.onset.onset_detect(onset_envelope=envelope, sr=sr, hop_length=ONSET_HOP_LENGTH)
    return librosa.frames_to_time(frames, sr=sr, hop_length=ONSET_HOP_LENGTH), envelope[frames]


def _fit_constant_tempo(onsets: np.ndarray, near_bpm: float) -> tuple[float, float, float, int]:
    """
    Find the single tempo whose subdivisions best line up with the onsets.

    Each onset becomes a point on a circle one subdivision long. At the right tempo
    the points pile up in one place, and the length of their average measures how
    tightly (the "fit"). Both 16ths and triplets are tried so swung parts can fit too.
    Returns (fit, bpm, position of a subdivision in seconds, subdivisions per beat).
    """

    def score(bpms: np.ndarray, subdivisions: int) -> np.ndarray:
        units = 60 / bpms / subdivisions
        return np.exp(2j * np.pi * onsets[None, :] / units[:, None]).mean(axis=1)

    coarse = np.arange(near_bpm * (1 - TEMPO_SEARCH_RANGE), near_bpm * (1 + TEMPO_SEARCH_RANGE), 0.01)
    best = (-1.0, near_bpm, 0.0, 4)
    for subdivisions in (4, 3):
        rough = coarse[np.abs(score(coarse, subdivisions)).argmax()]
        fine = np.arange(rough - 0.02, rough + 0.02, 0.001)
        phasors = score(fine, subdivisions)
        i = int(np.abs(phasors).argmax())
        if abs(phasors[i]) > best[0]:
            position = float(np.angle(phasors[i])) / (2 * np.pi) * (60 / fine[i] / subdivisions)
            best = (float(abs(phasors[i])), float(fine[i]), position, subdivisions)
    return best


def measure_grid_drift_ms(onsets: np.ndarray, bpm: float, subdivisions: int, position: float) -> float:
    """
    How far local stretches of the track sit from a rigid grid, in milliseconds (95th percentile).

    This separates a moving tempo from a steady part with many off-grid notes: those
    notes lower the fit score, but only a moving tempo shifts where the local grid sits.
    """
    unit = 60 / bpm / subdivisions
    reference = np.exp(-2j * np.pi * position / unit)
    window_index = (onsets // GRID_DRIFT_WINDOW_SECONDS).astype(int)

    drifts = []
    for window in np.unique(window_index):
        members = onsets[window_index == window]
        if len(members) >= GRID_DRIFT_MIN_WINDOW_ONSETS:
            phasor = np.exp(2j * np.pi * members / unit).mean() * reference
            drifts.append(abs(float(np.angle(phasor))) / (2 * np.pi) * unit * 1000)

    return float(np.percentile(drifts, 95)) if drifts else float("inf")


def estimate_rhythm(
    y: np.ndarray, sr: int, onsets: np.ndarray, strengths: np.ndarray
) -> tuple[float | None, np.ndarray, bool]:
    """
    Return the tempo in BPM, the beat times in seconds, and whether the tempo is steady.

    The beat tracker follows the attacks it hears, so its beats wander even on a
    perfectly steady part. It only picks the rough tempo and which subdivision the
    beat falls on. When the onsets fit one constant tempo, the beats come from that.
    """
    if is_silent(y):
        return None, np.array([]), False

    tempo, tracked_beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    # librosa returns tempo as a one-element array, which float() rejects on current NumPy.
    tracked_bpm = float(np.atleast_1d(tempo)[0])
    if tracked_bpm <= 0:
        return None, tracked_beats, False
    while tracked_bpm > MAX_PLAUSIBLE_BPM:
        tracked_bpm /= 2

    strong = onsets[strengths >= TIMING_MIN_RELATIVE_STRENGTH * np.median(strengths)] if len(onsets) else onsets
    if len(strong) >= MIN_ONSETS_FOR_TEMPO_FIT and len(tracked_beats) > 0:
        fit, bpm, position, subdivisions = _fit_constant_tempo(strong, tracked_bpm)
        if fit < TEMPO_FIT_MIN_TIMES_CHANCE / np.sqrt(len(strong)):
            return round(tracked_bpm, 1), tracked_beats, False
        # Too much drift to judge timing against, but still a better tempo reading than the tracker's.
        if measure_grid_drift_ms(strong, bpm, subdivisions, position) > GRID_MAX_DRIFT_MS:
            return round(bpm, 1), tracked_beats, False

        beat = 60 / bpm
        candidates = position + np.arange(subdivisions) * beat / subdivisions
        wrapped = np.abs((tracked_beats[None, :] - candidates[:, None] + beat / 2) % beat - beat / 2)
        first_beat = candidates[np.median(wrapped, axis=1).argmin()] % beat
        return round(bpm, 1), np.arange(first_beat, y.size / sr, beat), True

    return round(tracked_bpm, 1), tracked_beats, False


def find_exact_loop_period(y: np.ndarray, sr: int) -> float | None:
    """Return the repeat length in seconds if the audio is a digital loop, otherwise None."""
    usable = y.size - y.size % EXACT_LOOP_BLOCK
    blocks = y[:usable].reshape(-1, EXACT_LOOP_BLOCK).mean(axis=1, dtype=np.float64)
    n = blocks.size
    block_rate = sr / EXACT_LOOP_BLOCK
    min_lag = int(EXACT_LOOP_MIN_SECONDS * block_rate)
    if n < 2 * min_lag or is_silent(y):
        return None

    spectrum = np.fft.rfft(blocks, 2 * n)
    autocorrelation = np.fft.irfft(np.abs(spectrum) ** 2)[: n // 2 + 1]
    energy = np.concatenate([[0.0], np.cumsum(blocks**2)])
    lags = np.arange(n // 2 + 1)
    # Energy of the two overlapping parts at each lag, so a long lag isn't penalized for less overlap.
    overlap = np.sqrt(energy[n - lags] * (energy[n] - energy[lags]))
    correlation = autocorrelation / np.maximum(overlap, 1e-12)

    repeats = np.flatnonzero(correlation[min_lag:] >= EXACT_LOOP_MIN_CORRELATION)
    if repeats.size == 0:
        return None
    first = min_lag + int(repeats[0])
    # The correlation rises towards its peak, so step up to the top of that first peak.
    while first + 1 < correlation.size and correlation[first + 1] > correlation[first]:
        first += 1
    return first / block_rate


def _build_grid(beat_times: np.ndarray) -> np.ndarray:
    """
    Expand beats into a subdivision grid.

    Beat tracking snaps towards the onsets it finds, so a single late note can
    drag its own beat along and hide the mistake. Median-filtering the beats
    around a straight tempo line keeps slow tempo drift but ignores those
    one-off displacements.
    """
    index = np.arange(len(beat_times))
    tempo_line = np.polyval(np.polyfit(index, beat_times, 1), index)
    residual = beat_times - tempo_line
    kernel = min(9, len(beat_times) if len(beat_times) % 2 else len(beat_times) - 1)
    smoothed = tempo_line + medfilt(residual, kernel_size=kernel)

    starts, lengths = smoothed[:-1], np.diff(smoothed)
    grid = (starts[:, None] + lengths[:, None] * GRID_FRACTIONS).ravel()
    return np.append(grid, smoothed[-1])


def _nearest_index(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.abs(times[:, None] - grid[None, :]).argmin(axis=1)


def detect_timing_issues(
    onsets: np.ndarray,
    strengths: np.ndarray,
    beat_times: np.ndarray,
    start: float = 0.0,
    end: float | None = None,
    loop_period: float | None = None,
) -> list[TimingIssue]:
    """
    Flag onsets between `start` and `end` that sit away from the rhythmic grid.

    Onsets and beats should cover the whole track so the grid is stable for short
    regions; timestamps are always absolute positions in the file. In an exact
    loop, each spot in the loop is reported once, since repeats can't differ.
    """
    if len(beat_times) < 4 or len(onsets) == 0:
        return []

    grid = _build_grid(beat_times)

    # Onset detection reports consistently slightly late or early. Only the
    # departure from that consistent offset says anything about the performance.
    adjusted = onsets - np.median(onsets - grid[_nearest_index(onsets, grid)])
    nearest = _nearest_index(adjusted, grid)
    deviations = adjusted - grid[nearest]

    # How far each note has moved towards the midpoint with the next grid line.
    neighbours = np.clip(nearest + np.where(deviations >= 0, 1, -1), 0, len(grid) - 1)
    half_gaps = np.abs(grid[neighbours] - grid[nearest]) / 2
    ambiguity = np.abs(deviations) / np.maximum(half_gaps, 1e-6)

    in_scope = (onsets >= max(start, grid[0])) & (onsets <= min(end if end is not None else np.inf, grid[-1]))
    flagged = (
        in_scope
        & (np.abs(deviations) * 1000 >= TIMING_THRESHOLD_MS)
        & (ambiguity <= TIMING_MAX_AMBIGUITY)
        & (strengths >= TIMING_MIN_RELATIVE_STRENGTH * np.median(strengths))
    )

    # Strongest attacks first, skipping anything close to an onset we've already reported.
    kept: list[tuple[float, float]] = []

    def repeats_kept_spot(time: float) -> bool:
        if loop_period is None:
            return False
        half = loop_period / 2
        return any(abs((time - other + half) % loop_period - half) < TIMING_MIN_SPACING_S for other, _ in kept)

    for i in np.flatnonzero(flagged)[np.argsort(-strengths[flagged])]:
        time = float(onsets[i])
        if all(abs(time - other) >= TIMING_MIN_SPACING_S for other, _ in kept) and not repeats_kept_spot(time):
            kept.append((time, float(deviations[i])))
        if len(kept) == MAX_TIMING_ISSUES:
            break

    return [
        TimingIssue(
            timestamp=round(time, 2),
            description=f"Onset ~{abs(deviation) * 1000:.0f}ms {'late' if deviation > 0 else 'early'} against the beat grid",
        )
        for time, deviation in sorted(kept)
    ]


def compute_loudness_lufs(y: np.ndarray, sr: int) -> float | None:
    """Integrated loudness per ITU-R BS.1770. `y` is (samples,) or (channels, samples)."""
    if is_silent(y) or y.shape[-1] < int(0.4 * sr):
        return None  # BS.1770 measures over 400ms blocks

    loudness = pyloudnorm.Meter(sr).integrated_loudness(y.T)
    return round(float(loudness), 1) if np.isfinite(loudness) else None


def compute_dynamic_range(y: np.ndarray) -> float | None:
    """Spread between loud and quiet passages: 95th minus 10th percentile of short-term level."""
    if is_silent(y):
        return None

    rms = librosa.feature.rms(y=y)[0]
    levels = 20 * np.log10(rms[rms > SILENCE_RMS])
    if levels.size == 0:
        return None
    return round(float(np.percentile(levels, 95) - np.percentile(levels, 10)), 1)


def compute_frequency_profile(y: np.ndarray, sr: int) -> FrequencyProfile | None:
    """Share of spectral energy in the low / mid / high bands."""
    if is_silent(y):
        return None

    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    total = stft.sum() + 1e-10
    return FrequencyProfile(
        low_energy=round(float(stft[freqs < 250].sum() / total), 3),
        mid_energy=round(float(stft[(freqs >= 250) & (freqs < 4000)].sum() / total), 3),
        high_energy=round(float(stft[freqs >= 4000].sum() / total), 3),
    )


def analyze_audio(
    audio_path: str,
    start_time: float | None = None,
    end_time: float | None = None,
    session_id: str | None = None,
) -> AudioFeatures:
    """
    Analyze an audio file, optionally scoped to a region.

    Tempo and the beat grid always come from the whole track, since they can't
    be measured reliably from a few seconds. Key, loudness, dynamics and spectrum
    describe the region; timing issues are limited to it.
    """
    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=False)
    except Exception as exc:
        logger.info("Could not decode %s: %s", audio_path, exc)
        raise AudioDecodeError("Couldn't read this file as audio. It may be corrupt or an unsupported format.") from exc

    sr = int(sr)
    if audio.size == 0:
        raise AudioDecodeError("This audio file is empty.")

    full_duration = audio.shape[-1] / sr
    start = 0.0 if start_time is None else start_time
    end = full_duration if end_time is None else min(end_time, full_duration)
    if start < 0 or end - start < MIN_REGION_SECONDS:
        raise InvalidRegionError(f"Region must be at least {MIN_REGION_SECONDS}s long and inside the file.")

    region = audio[..., int(start * sr) : int(end * sr)]
    mono_full = librosa.to_mono(audio)
    mono_region = librosa.to_mono(region)

    onsets, strengths = detect_onsets(mono_full, sr) if not is_silent(mono_full) else (np.array([]), np.array([]))
    bpm, beat_times, tempo_is_steady = estimate_rhythm(mono_full, sr, onsets, strengths)
    # Without a steady tempo there's no reliable grid, so timing can't be judged.
    timing_issues = (
        detect_timing_issues(onsets, strengths, beat_times, start, end, find_exact_loop_period(mono_full, sr))
        if tempo_is_steady
        else []
    )
    centroid = None
    onset_strength = None
    if not is_silent(mono_region):
        centroid = round(float(librosa.feature.spectral_centroid(y=mono_region, sr=sr).mean()), 1)
        onset_strength = round(float(librosa.onset.onset_strength(y=mono_region, sr=sr).mean()), 4)

    return AudioFeatures(
        session_id=session_id or str(uuid.uuid4()),
        duration=round(mono_region.size / sr, 2),
        sample_rate=sr,
        bpm=bpm,
        key=detect_key(mono_region, sr),
        loudness_lufs=compute_loudness_lufs(region, sr),
        dynamic_range_db=compute_dynamic_range(mono_region),
        spectral_centroid_mean=centroid,
        onset_strength_mean=onset_strength,
        timing_checked=tempo_is_steady,
        timing_issues=timing_issues,
        frequency_profile=compute_frequency_profile(mono_region, sr),
    )
