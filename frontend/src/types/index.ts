export interface AudioRegion {
  start: number; // seconds
  end: number;   // seconds
}

export interface TimingIssue {
  timestamp: number;
  description: string;
}

export interface FrequencyProfile {
  low_energy: number;
  mid_energy: number;
  high_energy: number;
}

export interface AudioFeatures {
  session_id: string;
  duration: number;
  sample_rate: number;
  bpm: number | null;
  key: string | null;
  loudness_lufs: number | null;
  peak_dbfs: number | null;
  dynamic_range_db: number | null;
  spectral_centroid_mean: number | null;
  onset_strength_mean: number | null;
  timing_checked: boolean;
  timing_issues: TimingIssue[];
  frequency_profile: FrequencyProfile | null;
}

export interface GainEdit {
  type: "gain";
  reason: string;
  gain_db: number;
}

export interface EqEdit {
  type: "eq";
  reason: string;
  eq_type: "peak" | "low_shelf" | "high_shelf" | "highpass" | "lowpass";
  freq_hz: number;
  gain_db: number;
  q: number;
}

export interface CompressorEdit {
  type: "compressor";
  reason: string;
  threshold_db: number;
  ratio: number;
  attack_ms: number;
  release_ms: number;
  makeup_db: number;
}

export interface LimiterEdit {
  type: "limiter";
  reason: string;
  ceiling_db: number;
  input_gain_db: number;
  release_ms: number;
}

export interface NudgeEdit {
  type: "nudge";
  reason: string;
  onset_time: number;
  shift_ms: number;
}

export type AudioEdit = GainEdit | EqEdit | CompressorEdit | LimiterEdit | NudgeEdit;

export interface EditEstimate {
  before_lufs: number | null;
  after_lufs: number | null;
  before_peak_dbfs: number | null;
  after_peak_dbfs: number | null;
  turned_down_db: number;
  nudges_applied: number;
}

export interface EditProposal {
  summary: string;
  edits: AudioEdit[];
}

export interface EditReport {
  nudges_applied: number;
  loop_repeats_nudged: boolean;
  turned_down_db: number;
  peak_dbfs: number | null;
}

export interface EditResult {
  blob: Blob;
  report: EditReport;
  edits: AudioEdit[];
  region: AudioRegion | null;
  before: AudioFeatures | null;
  after: AudioFeatures | null;
}

export type ChatStreamEvent = { type: "text"; text: string } | { type: "proposal"; proposal: EditProposal };

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  region?: AudioRegion;
  proposal?: EditProposal;
  // The file and region a proposal was made for, so it's only applied to that audio.
  proposalFile?: File;
  proposalRegion?: AudioRegion | null;
}

export interface ChatRequest {
  session_id: string;
  message: string;
  audio_features: AudioFeatures | null;
  region: AudioRegion | null;
  history: Array<{ role: string; content: string }>;
}
