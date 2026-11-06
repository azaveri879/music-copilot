from pydantic import BaseModel
from typing import Optional


class AudioRegion(BaseModel):
    start: float  # seconds
    end: float    # seconds


class TimingIssue(BaseModel):
    timestamp: float
    description: str


class FrequencyProfile(BaseModel):
    low_energy: float
    mid_energy: float
    high_energy: float


class AudioFeatures(BaseModel):
    session_id: str
    duration: float
    sample_rate: int
    bpm: Optional[float] = None
    key: Optional[str] = None
    loudness_lufs: Optional[float] = None
    dynamic_range_db: Optional[float] = None
    spectral_centroid_mean: Optional[float] = None
    onset_strength_mean: Optional[float] = None
    timing_issues: list[TimingIssue] = []
    frequency_profile: Optional[FrequencyProfile] = None


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    audio_features: Optional[AudioFeatures] = None
    region: Optional[AudioRegion] = None
    history: list[ChatHistoryItem] = []
