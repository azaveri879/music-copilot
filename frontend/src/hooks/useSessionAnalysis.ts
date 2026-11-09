import { useEffect, useState } from "react";
import { AudioFeatures } from "../types";
import { analyzeAudio } from "../utils/api";

// Analysis is CPU-heavy on the backend, so a whole folder shouldn't run at once.
const CONCURRENT_ANALYSES = 2;

export interface SessionTrack {
  file: File;
  features: AudioFeatures | null;
  error: string | null;
}

/** Analyzes every track in a song. A file that fails is reported on its own row and the rest carry on. */
export function useSessionAnalysis(files: File[]) {
  const [tracks, setTracks] = useState<SessionTrack[]>([]);

  useEffect(() => {
    setTracks(files.map((file) => ({ file, features: null, error: null })));
    if (files.length === 0) return;

    const controller = new AbortController();
    const update = (index: number, change: Partial<SessionTrack>) =>
      setTracks((prev) => prev.map((track, i) => (i === index ? { ...track, ...change } : track)));

    let next = 0;
    const work = async () => {
      while (next < files.length && !controller.signal.aborted) {
        const index = next++;
        try {
          const features = await analyzeAudio(files[index], { remember: false, signal: controller.signal });
          if (!controller.signal.aborted) update(index, { features });
        } catch (err) {
          if (!controller.signal.aborted) {
            update(index, { error: err instanceof Error ? err.message : "Analysis failed" });
          }
        }
      }
    };
    for (let i = 0; i < CONCURRENT_ANALYSES; i++) void work();

    return () => controller.abort();
  }, [files]);

  return { tracks, isAnalyzing: tracks.some((track) => !track.features && !track.error) };
}
