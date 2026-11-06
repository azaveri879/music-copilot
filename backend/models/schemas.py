from typing import Literal, Optional

from pydantic import BaseModel, Field


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
    timing_checked: bool = False  # False when the tempo isn't steady enough to judge timing against
    timing_issues: list[TimingIssue] = []
    frequency_profile: Optional[FrequencyProfile] = None


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1)
    audio_features: Optional[AudioFeatures] = None
    region: Optional[AudioRegion] = None
    history: list[ChatHistoryItem] = []
