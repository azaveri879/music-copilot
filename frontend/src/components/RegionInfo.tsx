import { AudioRegion } from "../types";
import { formatTime } from "../utils/format";

interface RegionInfoProps {
  region: AudioRegion | null;
  isAnalyzing: boolean;
  onClear: () => void;
}

export function RegionInfo({ region, isAnalyzing, onClear }: RegionInfoProps) {
  if (!region) return null;

  return (
    <div className="flex items-center justify-between gap-2 mb-2 text-xs text-amber-400 bg-amber-500/10 rounded-lg px-2.5 py-1.5">
      <span>
        Asking about {formatTime(region.start)} – {formatTime(region.end)}
        {isAnalyzing && <span className="opacity-70 animate-pulse"> · analyzing…</span>}
      </span>
      <button onClick={onClear} className="text-amber-300/70 hover:text-amber-200 transition-colors">
        Use full track
      </button>
    </div>
  );
}
