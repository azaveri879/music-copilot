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
TIMING_MIN_SPACING_S = 0.5
MAX_TIMING_ISSUES = 10


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


def estimate_rhythm(y: np.ndarray, sr: int) -> tuple[float | None, np.ndarray]:
    """Return the tempo in BPM and the beat times in seconds."""
    if is_silent(y):
        return None, np.array([])

    tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units="time")
    # librosa returns tempo as a one-element array, which float() rejects on current NumPy.
    bpm = float(np.atleast_1d(tempo)[0])
    return (round(bpm, 1) if bpm > 0 else None), beat_times


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


def detect_timing_issues(
    y: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    start: float = 0.0,
    end: float | None = None,
) -> list[TimingIssue]:
    """
    Flag onsets between `start` and `end` that sit away from the rhythmic grid.

    `y` should be the whole track so the grid is stable for short regions;
    timestamps are always absolute positions in the file.
    """
    if len(beat_times) < 4 or is_silent(y):
        return []

    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    if len(onsets) == 0:
        return []

    grid = _build_grid(beat_times)
    nearest = np.abs(onsets[:, None] - grid[None, :]).argmin(axis=1)
    deviations = onsets - grid[nearest]

    # Onset detection reports consistently slightly late or early. Only the
    # departure from that consistent offset says anything about the performance.
    deviations = deviations - np.median(deviations)

    in_scope = (onsets >= max(start, grid[0])) & (onsets <= min(end if end is not None else np.inf, grid[-1]))
    flagged = in_scope & (np.abs(deviations) * 1000 >= TIMING_THRESHOLD_MS)

    # Worst first, skipping anything close to an onset we've already reported.
    kept: list[tuple[float, float]] = []
    for i in np.flatnonzero(flagged)[np.argsort(-np.abs(deviations[flagged]))]:
        time = float(onsets[i])
        if all(abs(time - other) >= TIMING_MIN_SPACING_S for other, _ in kept):
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

    bpm, beat_times = estimate_rhythm(mono_full, sr)
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
        timing_issues=detect_timing_issues(mono_full, sr, beat_times, start, end),
        frequency_profile=compute_frequency_profile(mono_region, sr),
    )
