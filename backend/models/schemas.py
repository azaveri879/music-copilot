from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter


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
    peak_dbfs: Optional[float] = None
    dynamic_range_db: Optional[float] = None
    spectral_centroid_mean: Optional[float] = None
    onset_strength_mean: Optional[float] = None
    timing_checked: bool = False  # False when the tempo isn't steady enough to judge timing against
    timing_issues: list[TimingIssue] = []
    frequency_profile: Optional[FrequencyProfile] = None


class GainEdit(BaseModel):
    type: Literal["gain"]
    reason: str = ""
    gain_db: float = Field(ge=-24, le=18)


class EqEdit(BaseModel):
    type: Literal["eq"]
    reason: str = ""
    eq_type: Literal["peak", "low_shelf", "high_shelf", "highpass", "lowpass"]
    freq_hz: float = Field(ge=20, le=20000)
    gain_db: float = Field(default=0.0, ge=-18, le=18)  # ignored by highpass and lowpass
    q: float = Field(default=0.707, ge=0.1, le=10)


class CompressorEdit(BaseModel):
    type: Literal["compressor"]
    reason: str = ""
    threshold_db: float = Field(ge=-60, le=0)
    ratio: float = Field(ge=1, le=20)
    attack_ms: float = Field(default=10.0, ge=0.1, le=200)
    release_ms: float = Field(default=120.0, ge=10, le=2000)
    makeup_db: float = Field(default=0.0, ge=0, le=18)


class LimiterEdit(BaseModel):
    type: Literal["limiter"]
    reason: str = ""
    ceiling_db: float = Field(default=-1.0, ge=-12, le=-0.1)  # peaks never go above this
    input_gain_db: float = Field(default=0.0, ge=0, le=18)  # drive into the limiter to gain loudness
    release_ms: float = Field(default=100.0, ge=10, le=1000)


class NudgeEdit(BaseModel):
    type: Literal["nudge"]
    reason: str = ""
    onset_time: float = Field(ge=0)  # seconds into the file
    shift_ms: float = Field(ge=-100, le=100)  # positive moves the note later


AudioEdit = Annotated[Union[GainEdit, EqEdit, CompressorEdit, LimiterEdit, NudgeEdit], Field(discriminator="type")]
AudioEditList = TypeAdapter(list[AudioEdit])


class EditProposal(BaseModel):
    summary: str
    edits: list[AudioEdit]


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1)
    audio_features: Optional[AudioFeatures] = None
    region: Optional[AudioRegion] = None
    history: list[ChatHistoryItem] = []
