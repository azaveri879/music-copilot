import { useEffect, useState } from "react";
import { AudioFeatures, EditResult } from "../types";
import { describeEdit, formatTime } from "../utils/format";

interface EditResultPanelProps {
  result: EditResult;
  fileName: string;
  listeningToEdited: boolean;
  onListen: (edited: boolean) => void;
  onDiscard: () => void;
}

export function EditResultPanel({ result, fileName, listeningToEdited, onListen, onDiscard }: EditResultPanelProps) {
  const { before, after, report, region, edits } = result;
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);

  useEffect(() => {
    const url = URL.createObjectURL(result.blob);
    setDownloadUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [result.blob]);

  const baseName = fileName.replace(/\.[^.]+$/, "");
  const hasTone = edits.some((edit) => edit.type !== "nudge");

  return (
    <div className="bg-gray-900 border border-indigo-500/30 rounded-xl p-4 space-y-3 text-xs">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="inline-flex rounded-lg bg-gray-950 p-0.5" role="group" aria-label="Choose what to hear">
          {[
            { label: "Original", edited: false },
            { label: "Edited", edited: true },
          ].map(({ label, edited }) => (
            <button
              key={label}
              onClick={() => onListen(edited)}
              aria-pressed={listeningToEdited === edited}
              className={`px-3 py-1 rounded-md transition-colors ${
                listeningToEdited === edited ? "bg-indigo-500 text-white" : "text-gray-400 hover:text-gray-200"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-3">
          {downloadUrl && (
            <a
              href={downloadUrl}
              download={`${baseName} - edited.wav`}
              className="px-3 py-1 bg-indigo-500 hover:bg-indigo-400 text-white rounded-md transition-colors"
            >
              Download WAV
            </a>
          )}
          <button onClick={onDiscard} className="text-gray-500 hover:text-gray-300 transition-colors">
            Discard
          </button>
        </div>
      </div>

      <ul className="text-gray-400 space-y-0.5">
        {edits.map((edit, i) => (
          <li key={i}>{describeEdit(edit)}</li>
        ))}
      </ul>

      {hasTone && region && (
        <p className="text-gray-500">
          Tone edits applied to {formatTime(region.start)} – {formatTime(region.end)}. The numbers below cover the whole
          track.
        </p>
      )}
      {before && after && <Comparison before={before} after={after} />}

      {report.nudges_applied > 0 && (
        <p className="text-gray-400">
          {report.loop_repeats_nudged
            ? `This file is an exact loop, so the note moved in all ${report.nudges_applied} repeats.`
            : `Moved ${report.nudges_applied} ${report.nudges_applied === 1 ? "note" : "notes"}.`}{" "}
          Other flags may appear after a move, because nearby candidates are only listed once the closest flag is gone.
        </p>
      )}
      {report.turned_down_db > 0 && (
        <p className="text-amber-300/80">
          The whole file was turned down {report.turned_down_db} dB to avoid clipping.
        </p>
      )}
      <p className="text-gray-500">Your original file hasn't changed. Trust your ears over these numbers.</p>
    </div>
  );
}

function Comparison({ before, after }: { before: AudioFeatures; after: AudioFeatures }) {
  const percent = (value: number | undefined) => (value === undefined ? "—" : `${Math.round(value * 100)}%`);
  const lufs = (value: number | null) => (value === null ? "—" : `${value.toFixed(1)} LUFS`);
  const dbfs = (value: number | null) => (value === null ? "—" : `${value.toFixed(1)} dBFS`);

  const rows = [
    ["Loudness", lufs(before.loudness_lufs), lufs(after.loudness_lufs)],
    ["Peak", dbfs(before.peak_dbfs), dbfs(after.peak_dbfs)],
    ["Low (<250Hz)", percent(before.frequency_profile?.low_energy), percent(after.frequency_profile?.low_energy)],
    ["Mid", percent(before.frequency_profile?.mid_energy), percent(after.frequency_profile?.mid_energy)],
    ["High (>4kHz)", percent(before.frequency_profile?.high_energy), percent(after.frequency_profile?.high_energy)],
  ];

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left">
        <thead className="text-gray-500">
          <tr>
            <th className="font-normal py-0.5">Whole track</th>
            <th className="font-normal py-0.5">Before</th>
            <th className="font-normal py-0.5">After</th>
          </tr>
        </thead>
        <tbody className="text-gray-300">
          {rows.map(([label, from, to]) => (
            <tr key={label}>
              <td className="py-0.5 text-gray-500">{label}</td>
              <td className="py-0.5">{from}</td>
              <td className={`py-0.5 ${from === to ? "" : "text-indigo-300"}`}>{to}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
