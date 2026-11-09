import { SessionTrack } from "../hooks/useSessionAnalysis";
import { trackName } from "../utils/files";
import { SongTempoInput } from "./SongTempoInput";

const NEAR_FULL_SCALE_DBFS = -1;

interface TrackTableProps {
  tracks: SessionTrack[];
  focusedIndex: number | null;
  onFocus: (index: number | null) => void;
  songBpmText: string;
  onSongBpmChange: (value: string) => void;
}

function percent(share: number) {
  return Math.round(share * 100);
}

export function TrackTable({ tracks, focusedIndex, onFocus, songBpmText, onSongBpmChange }: TrackTableProps) {
  return (
    <div className="bg-gray-900 rounded-xl p-4 space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          aria-pressed={focusedIndex === null}
          onClick={() => onFocus(null)}
          className={`text-xs px-3 py-1 rounded-full border transition-colors ${
            focusedIndex === null
              ? "bg-indigo-600 border-indigo-600 text-white"
              : "border-gray-700 text-gray-300 hover:border-indigo-500"
          }`}
        >
          Whole song
        </button>
        <span className="text-xs text-gray-500">{tracks.length} tracks · click one to work on it</span>
        <label htmlFor="session-bpm" className="ml-auto text-xs text-gray-500">
          BPM
        </label>
        <SongTempoInput
          id="session-bpm"
          value={songBpmText}
          detectedBpm={null}
          onChange={onSongBpmChange}
          className="w-24 text-right"
        />
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-gray-500">
              <th className="font-normal py-1 pr-3">Track</th>
              <th className="font-normal py-1 pr-3 text-right">LUFS</th>
              <th className="font-normal py-1 pr-3 text-right">Peak (dBFS)</th>
              <th className="font-normal py-1 text-right">Low / Mid / High %</th>
            </tr>
          </thead>
          <tbody>
            {tracks.map(({ file, features, error }, index) => {
              const profile = features?.frequency_profile;
              const peak = features?.peak_dbfs ?? null;
              return (
                <tr
                  key={`${index}-${file.name}`}
                  onClick={error ? undefined : () => onFocus(index)}
                  className={`border-t border-gray-800 ${error ? "" : "cursor-pointer hover:bg-gray-800/60"} ${
                    focusedIndex === index ? "bg-indigo-500/15" : ""
                  }`}
                >
                  <td className="py-1.5 pr-3 text-gray-200">
                    <button
                      type="button"
                      disabled={!!error}
                      aria-current={focusedIndex === index}
                      className="text-left break-all"
                    >
                      {trackName(file)}
                    </button>
                  </td>
                  {error ? (
                    <td colSpan={3} className="py-1.5 text-right text-red-400">
                      {error}
                    </td>
                  ) : !features ? (
                    <td colSpan={3} className="py-1.5 text-right text-indigo-400 animate-pulse">
                      Analyzing…
                    </td>
                  ) : (
                    <>
                      <td className="py-1.5 pr-3 text-right text-gray-300 tabular-nums">
                        {features.loudness_lufs?.toFixed(1) ?? "—"}
                      </td>
                      <td
                        className={`py-1.5 pr-3 text-right tabular-nums ${
                          peak !== null && peak > NEAR_FULL_SCALE_DBFS ? "text-amber-400" : "text-gray-300"
                        }`}
                      >
                        {peak?.toFixed(1) ?? "—"}
                      </td>
                      <td className="py-1.5 text-right text-gray-300 tabular-nums">
                        {profile
                          ? `${percent(profile.low_energy)} / ${percent(profile.mid_energy)} / ${percent(profile.high_energy)}`
                          : "—"}
                      </td>
                    </>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
