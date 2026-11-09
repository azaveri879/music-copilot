import { MAX_SONG_BPM, MIN_SONG_BPM } from "../utils/tempo";

interface SongTempoInputProps {
  id: string;
  value: string;
  detectedBpm: number | null;
  onChange: (value: string) => void;
  className?: string;
}

export function SongTempoInput({ id, value, detectedBpm, onChange, className = "" }: SongTempoInputProps) {
  return (
    <input
      id={id}
      type="number"
      min={MIN_SONG_BPM}
      max={MAX_SONG_BPM}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={detectedBpm !== null ? String(Math.round(detectedBpm)) : "Enter tempo"}
      title="Type your song's tempo to use it instead of the detected one"
      className={`bg-transparent text-sm font-medium text-gray-200 focus:outline-none ${
        detectedBpm !== null ? "placeholder:text-gray-200" : "placeholder:text-gray-500"
      } ${className}`}
    />
  );
}
