import numpy as np
import pytest
import soundfile as sf

SR = 44100


def click(y: np.ndarray, time: float, freq: float = 1000.0) -> None:
    n = int(0.03 * SR)
    start = int(time * SR)
    t = np.arange(n) / SR
    y[start : start + n] += 0.8 * np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.008)


@pytest.fixture
def write_wav(tmp_path):
    def _write(y: np.ndarray, name: str = "audio.wav", sr: int = SR) -> str:
        path = tmp_path / name
        sf.write(path, y, sr)
        return str(path)

    return _write


@pytest.fixture
def click_track():
    """Quarter notes at 120 BPM, with optional per-beat displacement in seconds."""

    def _make(seconds: int = 20, offsets: dict[int, float] | None = None) -> np.ndarray:
        y = np.zeros(SR * seconds)
        for beat in range(seconds * 2 - 1):
            click(y, beat * 0.5 + (offsets or {}).get(beat, 0.0))
        return y

    return _make
