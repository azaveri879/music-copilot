import { useEffect, useMemo, useState } from "react";
import { AudioEdit, AudioRegion, ChatMessage, EditEstimate, EditResult } from "../types";
import { estimateEdits } from "../utils/api";
import { describeEdit, formatTime } from "../utils/format";

export interface ProposalHandlers {
  currentFile: File | null;
  isApplying: boolean;
  error: string | null;
  applied: EditResult | null;
  onApply: (file: File, edits: AudioEdit[], region: AudioRegion | null) => void;
}

interface EditProposalCardProps {
  message: ChatMessage;
  handlers: ProposalHandlers;
}

const ESTIMATE_DEBOUNCE_MS = 300;

export function EditProposalCard({ message, handlers }: EditProposalCardProps) {
  const proposal = message.proposal!;
  const region = message.proposalRegion ?? null;

  // Note moves rely on unconfirmed timing flags, so each one is opt-in.
  const [selected, setSelected] = useState(() => proposal.edits.map((edit) => edit.type !== "nudge"));

  const chosen = useMemo(() => proposal.edits.filter((_, i) => selected[i]), [proposal.edits, selected]);
  const isCurrentFile = !!message.proposalFile && message.proposalFile === handlers.currentFile;

  // Measured on the real audio, and again whenever the selection changes.
  const [estimate, setEstimate] = useState<EditEstimate | null>(null);
  const [estimateError, setEstimateError] = useState<string | null>(null);
  useEffect(() => {
    setEstimate(null);
    setEstimateError(null);
    if (!isCurrentFile || chosen.length === 0) return;

    const controller = new AbortController();
    const timer = setTimeout(() => {
      estimateEdits(message.proposalFile!, chosen, region, controller.signal)
        .then(setEstimate)
        .catch((err) => {
          if (!controller.signal.aborted) setEstimateError(err instanceof Error ? err.message : "Couldn't estimate");
        });
    }, ESTIMATE_DEBOUNCE_MS);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [isCurrentFile, chosen, region, message.proposalFile]);
  const hasNudge = proposal.edits.some((edit) => edit.type === "nudge");
  const hasTone = proposal.edits.some((edit) => edit.type !== "nudge");

  return (
    <div className="mt-3 rounded-lg border border-indigo-500/30 bg-gray-900 p-3 space-y-2.5 text-xs">
      <div>
        <p className="font-medium text-indigo-300">Proposed edits</p>
        <p className="text-gray-300 mt-0.5">{proposal.summary}</p>
      </div>

      <ul className="space-y-2">
        {proposal.edits.map((edit, i) => (
          <li key={i}>
            <label className="flex items-start gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={selected[i]}
                onChange={() => setSelected((prev) => prev.map((value, j) => (j === i ? !value : value)))}
                className="mt-0.5 accent-indigo-500"
              />
              <span>
                <span className="text-gray-200">{describeEdit(edit)}</span>
                {edit.reason && <span className="block text-gray-500">{edit.reason}</span>}
              </span>
            </label>
          </li>
        ))}
      </ul>

      {hasTone && (
        <p className="text-gray-500">
          Tone edits apply to {region ? `${formatTime(region.start)} – ${formatTime(region.end)}` : "the whole track"}.
        </p>
      )}
      {hasNudge && (
        <p className="text-amber-300/80">
          Note moves are off until you tick them. Check the note by ear first, because timing flags can be wrong.
        </p>
      )}

      {isCurrentFile && chosen.length > 0 && (
        <EstimateSummary estimate={estimate} error={estimateError} scoped={!!region} />
      )}

      {isCurrentFile ? (
        <button
          onClick={() => handlers.onApply(message.proposalFile!, chosen, region)}
          disabled={chosen.length === 0 || handlers.isApplying}
          className="w-full px-3 py-1.5 bg-indigo-500 hover:bg-indigo-400 disabled:opacity-30 disabled:cursor-not-allowed text-white rounded-md transition-colors"
        >
          {handlers.isApplying ? "Applying…" : `Apply ${chosen.length} to a copy`}
        </button>
      ) : (
        <p className="text-gray-500">These edits were for a different file.</p>
      )}
      {isCurrentFile && handlers.error && <p className="text-red-400">{handlers.error}</p>}
    </div>
  );
}

function EstimateSummary({
  estimate,
  error,
  scoped,
}: {
  estimate: EditEstimate | null;
  error: string | null;
  scoped: boolean;
}) {
  if (error) return <p className="text-red-400">{error}</p>;
  if (!estimate) return <p className="text-gray-500 animate-pulse">Measuring these edits…</p>;

  const show = (value: number | null, unit: string) => (value === null ? "—" : `${value.toFixed(1)} ${unit}`);

  return (
    <div className="rounded-md bg-gray-950 px-2.5 py-2 space-y-1">
      <p className="text-gray-500">Measured on your audio{scoped ? " (selection)" : ""}</p>
      <p className="text-gray-200">
        Loudness {show(estimate.before_lufs, "LUFS")} → {show(estimate.after_lufs, "LUFS")}
      </p>
      <p className="text-gray-200">
        Peak {show(estimate.before_peak_dbfs, "dBFS")} → {show(estimate.after_peak_dbfs, "dBFS")}
      </p>
      {estimate.turned_down_db > 0 && (
        <p className="text-amber-300/80">
          This would clip, so the whole file gets turned down {estimate.turned_down_db} dB. A limiter may suit better.
        </p>
      )}
    </div>
  );
}
