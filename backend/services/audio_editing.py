import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import sosfilt

from models.schemas import AudioEdit, AudioRegion, CompressorEdit, EqEdit, GainEdit, LimiterEdit, NudgeEdit
from services.audio_analysis import (
    AudioDecodeError,
    InvalidRegionError,
    compute_loudness_lufs,
    compute_peak_dbfs,
    detect_onsets,
    find_exact_loop_period,
)

REGION_FADE_SECONDS = 0.01
COMPRESSOR_BLOCK_SECONDS = 0.001
PEAK_CEILING = 10 ** (-0.3 / 20)  # -0.3 dBFS
LIMITER_LOOKAHEAD_SECONDS = 0.005
LIMITER_BLOCK_SECONDS = 0.0005

NUDGE_CROSSFADE_SECONDS = 0.004
NUDGE_ATTACK_SEARCH_SECONDS = 0.03  # detected onsets can trail the true start of an attack
NUDGE_LEVEL_WINDOW_SECONDS = 0.001
NUDGE_MIN_ATTACK_STRENGTH = 0.5  # steady tones produce tiny false onsets; real notes score far higher
NUDGE_SNAP_SECONDS = 0.05  # how close onset_time must be to a detected note
NUDGE_MAX_NOTE_SECONDS = 2.0


class EditError(ValueError):
    """An edit that can't be applied to this audio, with a message for the user."""


@dataclass
class EditReport:
    nudges_applied: int = 0
    loop_repeats_nudged: bool = False
    turned_down_db: float = 0.0
    peak_dbfs: float | None = None


def _biquad_sos(edit: EqEdit, sr: int) -> np.ndarray:
    """Second-order filter coefficients from the Audio EQ Cookbook (R. Bristow-Johnson)."""
    if edit.freq_hz >= sr / 2:
        raise EditError(f"EQ frequency {edit.freq_hz:.0f}Hz is above this file's limit of {sr / 2:.0f}Hz.")

    a_gain = 10 ** (edit.gain_db / 40)
    w0 = 2 * np.pi * edit.freq_hz / sr
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / (2 * edit.q)
    root = 2 * np.sqrt(a_gain) * alpha

    if edit.eq_type == "peak":
        b = [1 + alpha * a_gain, -2 * cos_w0, 1 - alpha * a_gain]
        a = [1 + alpha / a_gain, -2 * cos_w0, 1 - alpha / a_gain]
    elif edit.eq_type == "low_shelf":
        b = [
            a_gain * ((a_gain + 1) - (a_gain - 1) * cos_w0 + root),
            2 * a_gain * ((a_gain - 1) - (a_gain + 1) * cos_w0),
            a_gain * ((a_gain + 1) - (a_gain - 1) * cos_w0 - root),
        ]
        a = [
            (a_gain + 1) + (a_gain - 1) * cos_w0 + root,
            -2 * ((a_gain - 1) + (a_gain + 1) * cos_w0),
            (a_gain + 1) + (a_gain - 1) * cos_w0 - root,
        ]
    elif edit.eq_type == "high_shelf":
        b = [
            a_gain * ((a_gain + 1) + (a_gain - 1) * cos_w0 + root),
            -2 * a_gain * ((a_gain - 1) + (a_gain + 1) * cos_w0),
            a_gain * ((a_gain + 1) + (a_gain - 1) * cos_w0 - root),
        ]
        a = [
            (a_gain + 1) - (a_gain - 1) * cos_w0 + root,
            2 * ((a_gain - 1) - (a_gain + 1) * cos_w0),
            (a_gain + 1) - (a_gain - 1) * cos_w0 - root,
        ]
    elif edit.eq_type == "highpass":
        b = [(1 + cos_w0) / 2, -(1 + cos_w0), (1 + cos_w0) / 2]
        a = [1 + alpha, -2 * cos_w0, 1 - alpha]
    else:  # lowpass
        b = [(1 - cos_w0) / 2, 1 - cos_w0, (1 - cos_w0) / 2]
        a = [1 + alpha, -2 * cos_w0, 1 - alpha]

    return np.array([[*(c / a[0] for c in b), 1.0, a[1] / a[0], a[2] / a[0]]])


def _compress(audio: np.ndarray, sr: int, edit: CompressorEdit) -> np.ndarray:
    """Feed-forward compressor, linked across channels so the stereo image holds."""
    block = max(1, int(COMPRESSOR_BLOCK_SECONDS * sr))
    usable = audio.shape[0] - audio.shape[0] % block
    peaks = np.abs(audio[:usable]).max(axis=1).reshape(-1, block).max(axis=1)
    if audio.shape[0] > usable:
        peaks = np.append(peaks, np.abs(audio[usable:]).max())

    levels_db = 20 * np.log10(np.maximum(peaks, 1e-9))
    target_reduction = np.maximum(levels_db - edit.threshold_db, 0) * (1 - 1 / edit.ratio)

    block_seconds = block / sr
    attack = np.exp(-block_seconds / (edit.attack_ms / 1000))
    release = np.exp(-block_seconds / (edit.release_ms / 1000))
    reduction = np.zeros_like(target_reduction)
    current = 0.0
    for i, target in enumerate(target_reduction):
        coefficient = attack if target > current else release
        current = coefficient * current + (1 - coefficient) * target
        reduction[i] = current

    # Blocks are 1ms, so interpolate to avoid stepping in the gain.
    positions = np.arange(reduction.size) * block + block / 2
    gain_db = edit.makeup_db - np.interp(np.arange(audio.shape[0]), positions, reduction)
    return audio * (10 ** (gain_db / 20))[:, None]


def _limit(audio: np.ndarray, sr: int, edit: LimiterEdit) -> np.ndarray:
    """
    Look-ahead peak limiter, linked across channels. No sample exceeds the ceiling.

    Gain starts falling before each peak arrives, so peaks are caught without the
    distortion of clipping, then recovers over the release time.
    """
    driven = audio * 10 ** (edit.input_gain_db / 20)
    ceiling = 10 ** (edit.ceiling_db / 20)
    needed = np.minimum(1.0, ceiling / np.maximum(np.abs(driven).max(axis=1), 1e-12))

    # Recovery runs on short blocks, since a per-sample Python loop is too slow for a whole song.
    block = max(1, int(LIMITER_BLOCK_SECONDS * sr))
    starts = np.arange(0, needed.size, block)
    block_needed = np.minimum.reduceat(needed, starts)
    recovery = np.exp(-(block / sr) / (edit.release_ms / 1000))
    held = np.empty_like(block_needed)
    current = 1.0
    for i, target in enumerate(block_needed):
        current = min(target, 1.0 - (1.0 - current) * recovery)
        held[i] = current
    gain = np.minimum(needed, np.repeat(held, block)[: needed.size])

    # A minimum filter then an average of the same width ramps smoothly into each peak
    # and never lands above what the peak needs, so the ceiling holds exactly.
    lookahead = max(1, int(LIMITER_LOOKAHEAD_SECONDS * sr)) | 1
    gain = uniform_filter1d(minimum_filter1d(gain, lookahead, mode="nearest"), lookahead, mode="nearest")
    return driven * np.minimum(gain, needed)[:, None]


def _apply_tone(audio: np.ndarray, sr: int, edit: GainEdit | EqEdit | CompressorEdit | LimiterEdit) -> np.ndarray:
    if isinstance(edit, LimiterEdit):
        return _limit(audio, sr, edit)
    if isinstance(edit, GainEdit):
        return audio * 10 ** (edit.gain_db / 20)
    if isinstance(edit, EqEdit):
        return sosfilt(_biquad_sos(edit, sr), audio, axis=0)
    return _compress(audio, sr, edit)


def _within_region(original: np.ndarray, processed: np.ndarray, sr: int, region: AudioRegion) -> np.ndarray:
    """Use the processed audio only inside the region, with short fades so the edges don't click."""
    n = original.shape[0]
    start, end = int(region.start * sr), min(int(region.end * sr), n)
    if start < 0 or end - start <= 0:
        raise InvalidRegionError("The selected region isn't inside this file.")

    fade = min(int(REGION_FADE_SECONDS * sr), (end - start) // 2)
    mask = np.zeros(n)
    mask[start:end] = 1.0
    if fade > 0:
        ramp = np.linspace(0, 1, fade, endpoint=False)
        mask[start : start + fade] = ramp
        mask[end - fade : end] = ramp[::-1]
    return original + (processed - original) * mask[:, None]


def _splice(out: np.ndarray, at: int, segment: np.ndarray, crossfade: int) -> None:
    """Paste a segment into `out`, blending at both ends."""
    weight = np.ones(segment.shape[0])
    ramp = np.linspace(0, 1, crossfade, endpoint=False)
    weight[:crossfade] = ramp
    weight[-crossfade:] = np.minimum(weight[-crossfade:], ramp[::-1])
    region = out[at : at + segment.shape[0]]
    out[at : at + segment.shape[0]] = region + (segment - region) * weight[:, None]


def _attack_start(mono: np.ndarray, sr: int, onset: float) -> int:
    """
    Find the sample where an attack really begins: the quietest point just before the detected onset.

    Onset detection reports a note slightly after it starts. Cutting there would
    leave the front of the attack behind, which sounds like a double hit.
    """
    onset_sample = int(onset * sr)
    search_start = max(0, onset_sample - int(NUDGE_ATTACK_SEARCH_SECONDS * sr))
    window = max(1, int(NUDGE_LEVEL_WINDOW_SECONDS * sr))
    levels = np.convolve(np.abs(mono[search_start : onset_sample + window]), np.ones(window) / window, mode="valid")
    return search_start + int(levels.argmin()) if levels.size else onset_sample


def _move_note(audio: np.ndarray, sr: int, start: int, end: int, shift_ms: float) -> bool:
    """
    Move one note in time without changing the length of the file. Returns False if it can't fit.

    The audio from this note to the next one moves as a block. The gap this opens is
    filled by repeating the audio just before it, which only works for short shifts.
    """
    crossfade = int(NUDGE_CROSSFADE_SECONDS * sr)
    moved = int(abs(shift_ms) / 1000 * sr)
    if moved <= crossfade or end - start <= moved + 2 * crossfade or end > audio.shape[0]:
        return False

    source = audio.copy()
    # Every blend finishes before an attack (this note's or the next one's), so attacks stay intact.
    if shift_ms > 0:
        if start - moved - 2 * crossfade < 0:
            return False
        # Fill the gap with what came just before it, then land the note after it.
        _splice(audio, start - crossfade, source[start - moved - 2 * crossfade : start], crossfade)
        _splice(audio, start + moved - crossfade, source[start - crossfade : end - moved], crossfade)
    else:
        if start - moved - crossfade < 0 or end - 2 * moved - crossfade < start:
            return False
        # Land the note earlier, then fill the gap it leaves with the end of its own tail.
        _splice(audio, start - moved - crossfade, source[start - crossfade : end], crossfade)
        _splice(audio, end - moved - crossfade, source[end - 2 * moved - crossfade : end - moved], crossfade)
    return True


def _apply_nudge(audio: np.ndarray, sr: int, edit: NudgeEdit, report: EditReport) -> None:
    mono = audio.mean(axis=1).astype(np.float32)
    onsets, strengths = detect_onsets(mono, sr)
    onsets = onsets[strengths >= NUDGE_MIN_ATTACK_STRENGTH]
    duration = audio.shape[0] / sr

    # Every repeat of a digital loop holds the same note, so it needs the same move.
    period = find_exact_loop_period(mono, sr)
    if period is None:
        targets = [edit.onset_time]
    else:
        targets = [edit.onset_time % period + k * period for k in range(int(duration / period) + 1)]
        report.loop_repeats_nudged = True

    moved = 0
    for target in targets:
        if len(onsets) == 0:
            break
        i = int(np.abs(onsets - target).argmin())
        if abs(onsets[i] - target) > NUDGE_SNAP_SECONDS:
            continue
        latest_end = min(onsets[i] + NUDGE_MAX_NOTE_SECONDS, duration)
        next_note = min(onsets[i + 1], latest_end) if i + 1 < len(onsets) else latest_end
        start = _attack_start(mono, sr, float(onsets[i]))
        end = _attack_start(mono, sr, float(next_note)) if i + 1 < len(onsets) else int(next_note * sr)
        if _move_note(audio, sr, start, end, edit.shift_ms):
            moved += 1

    if moved == 0:
        raise EditError(f"Couldn't move a note at {edit.onset_time:.2f}s. No note there, or not enough room around it.")
    report.nudges_applied += moved


@dataclass
class EditEstimate:
    """What a set of edits does, measured by rendering them. Covers the region if there is one."""

    before_lufs: float | None
    after_lufs: float | None
    before_peak_dbfs: float | None
    after_peak_dbfs: float | None
    turned_down_db: float
    nudges_applied: int


def _read_audio(content: bytes) -> tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(io.BytesIO(content), dtype="float64", always_2d=True)
    except sf.LibsndfileError as exc:
        raise AudioDecodeError("Couldn't read this file as audio. It may be corrupt or an unsupported format.") from exc
    if audio.size == 0:
        raise AudioDecodeError("This audio file is empty.")
    return audio, sr


def _render(
    audio: np.ndarray, sr: int, edits: list[AudioEdit], region: AudioRegion | None
) -> tuple[np.ndarray, EditReport]:
    """
    Note moves run first, so timing is corrected on the dry signal. Tone edits then
    apply in the given order, limited to the region if there is one, with any limiter
    last because it has to catch whatever the others produce.
    """
    if not edits:
        raise EditError("No edits were selected.")

    report = EditReport()
    result = audio.copy()

    for edit in edits:
        if isinstance(edit, NudgeEdit):
            _apply_nudge(result, sr, edit, report)

    tone_edits = [edit for edit in edits if not isinstance(edit, NudgeEdit)]
    for edit in sorted(tone_edits, key=lambda e: isinstance(e, LimiterEdit)):
        processed = _apply_tone(result, sr, edit)
        result = _within_region(result, processed, sr, region) if region else processed

    # Keep the file from clipping, since a WAV can't hold anything above full scale.
    peak = float(np.abs(result).max())
    if peak > PEAK_CEILING:
        result = result * (PEAK_CEILING / peak)
        report.turned_down_db = round(20 * np.log10(peak / PEAK_CEILING), 1)
    final_peak = float(np.abs(result).max())
    report.peak_dbfs = round(float(20 * np.log10(final_peak)), 1) if final_peak > 0 else None
    return result, report


def apply_edits(content: bytes, edits: list[AudioEdit], region: AudioRegion | None = None) -> tuple[bytes, EditReport]:
    """Apply edits to a copy of the audio and return it as a 24-bit WAV with a report."""
    audio, sr = _read_audio(content)
    result, report = _render(audio, sr, edits, region)
    buffer = io.BytesIO()
    sf.write(buffer, result, sr, format="WAV", subtype="PCM_24")
    return buffer.getvalue(), report


def estimate_edits(content: bytes, edits: list[AudioEdit], region: AudioRegion | None = None) -> EditEstimate:
    """Measure what edits would do, without producing a file to keep."""
    audio, sr = _read_audio(content)
    result, report = _render(audio, sr, edits, region)

    def scoped(y: np.ndarray) -> np.ndarray:
        # Channels first, which is what the loudness and peak helpers expect.
        return (y[int(region.start * sr) : int(region.end * sr)] if region else y).T

    return EditEstimate(
        before_lufs=compute_loudness_lufs(scoped(audio), sr),
        after_lufs=compute_loudness_lufs(scoped(result), sr),
        before_peak_dbfs=compute_peak_dbfs(scoped(audio)),
        after_peak_dbfs=compute_peak_dbfs(scoped(result)),
        turned_down_db=report.turned_down_db,
        nudges_applied=report.nudges_applied,
    )
