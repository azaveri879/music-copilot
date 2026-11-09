import { useCallback, useEffect, useRef, useState } from "react";
import { AudioEdit, AudioFeatures, AudioRegion, EditResult } from "../types";
import { analyzeAudio, applyEdits } from "../utils/api";

/** Applies approved edits to a copy of the file and holds the result for comparing. */
export function useAudioEdit(file: File | null, before: AudioFeatures | null) {
  const [result, setResult] = useState<EditResult | null>(null);
  const [listeningToEdited, setListeningToEdited] = useState(false);
  const [isApplying, setIsApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const currentFile = useRef(file);

  useEffect(() => {
    currentFile.current = file;
    setResult(null);
    setListeningToEdited(false);
    setError(null);
  }, [file]);

  const apply = useCallback(
    async (target: File, edits: AudioEdit[], region: AudioRegion | null) => {
      setIsApplying(true);
      setError(null);
      try {
        const { blob, report } = await applyEdits(target, edits, region);
        // Measuring the result is a bonus; the edit is still usable if it fails.
        const after = await analyzeAudio(new File([blob], "edited.wav", { type: "audio/wav" })).catch(() => null);
        if (currentFile.current !== target) return;
        setResult({ blob, report, edits, region, before, after });
        setListeningToEdited(true);
      } catch (err) {
        if (currentFile.current === target) setError(err instanceof Error ? err.message : "Applying the edits failed");
      } finally {
        setIsApplying(false);
      }
    },
    [before]
  );

  const discard = useCallback(() => {
    setResult(null);
    setListeningToEdited(false);
  }, []);

  return { result, listeningToEdited, setListeningToEdited, isApplying, error, apply, discard };
}
