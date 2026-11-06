import numpy as np
import librosa
from models.schemas import AudioFeatures, TimingIssue, FrequencyProfile
import uuid


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MODES = ["major", "minor"]


def detect_key(y: np.ndarray, sr: int) -> str | None:
    """Estimate the musical key using chroma features and the Krumhansl-Schmuckler key-finding algorithm."""
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)

        # Krumhansl-Schmuckler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        best_key, best_mode, best_corr = 0, "major", -np.inf
        for i in range(12):
            for profile, mode in [(major_profile, "major"), (minor_profile, "minor")]:
                rotated = np.roll(profile, i)
                corr = np.corrcoef(chroma_mean, rotated)[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    best_key = i
                    best_mode = mode

        return f"{NOTE_NAMES[best_key]} {best_mode}"
    except Exception:
        return None


def detect_timing_issues(y: np.ndarray, sr: int) -> list[TimingIssue]:
    """
    Detect potential timing issues by analyzing onset consistency.
    Flags onsets that deviate significantly from the expected beat grid.
    """
    issues = []
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)

        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")

        if len(beat_times) < 2:
            return issues

        beat_interval = float(np.median(np.diff(beat_times)))

        for onset in onsets:
            # Find the closest beat
            diffs = np.abs(beat_times - onset)
            closest_idx = int(np.argmin(diffs))
            closest_beat = beat_times[closest_idx]
            deviation_ms = (onset - closest_beat) * 1000

            # Flag if more than 50ms off the beat grid
            if abs(deviation_ms) > 50:
                direction = "late" if deviation_ms > 0 else "early"
                issues.append(TimingIssue(
                    timestamp=round(onset, 2),
                    description=f"Onset {abs(int(deviation_ms))}ms {direction} relative to beat grid"
                ))

        # Deduplicate by proximity (don't report within 500ms of each other)
        deduped = []
        for issue in sorted(issues, key=lambda x: x.timestamp):
            if not deduped or issue.timestamp - deduped[-1].timestamp > 0.5:
                deduped.append(issue)

        return deduped[:10]  # Cap at 10 issues
    except Exception:
        return issues


def compute_loudness_lufs(y: np.ndarray) -> float | None:
    """Approximate integrated loudness in LUFS using RMS."""
    try:
        rms = float(np.sqrt(np.mean(y ** 2)))
        if rms == 0:
            return None
        lufs = 20 * np.log10(rms) - 0.691
        return round(lufs, 1)
    except Exception:
        return None


def compute_frequency_profile(y: np.ndarray, sr: int) -> FrequencyProfile | None:
    """Compute energy distribution across low / mid / high frequency bands."""
    try:
        stft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)

        low_mask = freqs < 250
        mid_mask = (freqs >= 250) & (freqs < 4000)
        high_mask = freqs >= 4000

        total = stft.sum() + 1e-10
        return FrequencyProfile(
            low_energy=round(float(stft[low_mask].sum() / total), 3),
            mid_energy=round(float(stft[mid_mask].sum() / total), 3),
            high_energy=round(float(stft[high_mask].sum() / total), 3),
        )
    except Exception:
        return None


def analyze_audio(
    audio_path: str,
    start_time: float | None = None,
    end_time: float | None = None,
    session_id: str | None = None,
) -> AudioFeatures:
    """
    Load and analyze an audio file, optionally trimmed to a region.
    Returns structured AudioFeatures for use by the AI layer.
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    full_duration = float(librosa.get_duration(y=y, sr=sr))

    # Trim to region if specified
    if start_time is not None or end_time is not None:
        start_sample = int((start_time or 0.0) * sr)
        end_sample = int((end_time or full_duration) * sr)
        y = y[start_sample:end_sample]

    duration = float(librosa.get_duration(y=y, sr=sr))

    # BPM
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = round(float(tempo), 1)
    except Exception:
        bpm = None

    # Spectral centroid (brightness)
    try:
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = round(float(sc.mean()), 1)
    except Exception:
        spectral_centroid_mean = None

    # Onset strength
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_strength_mean = round(float(onset_env.mean()), 4)
    except Exception:
        onset_strength_mean = None

    # Dynamic range (difference between peak and RMS)
    try:
        rms = librosa.feature.rms(y=y)[0]
        dynamic_range_db = round(float(20 * np.log10(rms.max() / (rms.mean() + 1e-10))), 1)
    except Exception:
        dynamic_range_db = None

    return AudioFeatures(
        session_id=session_id or str(uuid.uuid4()),
        duration=round(duration, 2),
        sample_rate=int(sr),
        bpm=bpm,
        key=detect_key(y, sr),
        loudness_lufs=compute_loudness_lufs(y),
        dynamic_range_db=dynamic_range_db,
        spectral_centroid_mean=spectral_centroid_mean,
        onset_strength_mean=onset_strength_mean,
        timing_issues=detect_timing_issues(y, sr),
        frequency_profile=compute_frequency_profile(y, sr),
    )
