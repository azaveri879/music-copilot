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
  dynamic_range_db: number | null;
  spectral_centroid_mean: number | null;
  onset_strength_mean: number | null;
  timing_checked: boolean;
  timing_issues: TimingIssue[];
  frequency_profile: FrequencyProfile | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  region?: AudioRegion;
}

export interface ChatRequest {
  session_id: string;
  message: string;
  audio_features: AudioFeatures | null;
  region: AudioRegion | null;
  history: Array<{ role: string; content: string }>;
}
