import { useEffect, useRef, useState } from "react";
import { AudioFeatures, AudioRegion } from "../types";
import { analyzeAudio } from "../utils/api";

const REGION_DEBOUNCE_MS = 300;

/**
 * Keeps analysis in step with the current file and selected region. The full
 * track is analyzed once per file; each region gets its own analysis, so the
 * copilot's numbers describe what the user actually selected.
 */
export function useAudioAnalysis(file: File | null, region: AudioRegion | null, songBpm: number | null) {
  const [trackFeatures, setTrackFeatures] = useState<AudioFeatures | null>(null);
  const [regionFeatures, setRegionFeatures] = useState<AudioFeatures | null>(null);
  const [isAnalyzingTrack, setIsAnalyzingTrack] = useState(false);
  const [isAnalyzingRegion, setIsAnalyzingRegion] = useState(false);
  const [trackError, setTrackError] = useState<string | null>(null);
  const [regionError, setRegionError] = useState<string | null>(null);
  const measuredFile = useRef<File | null>(null);

  useEffect(() => {
    // A new tempo re-measures the same file, so keep showing its numbers until the update arrives.
    if (measuredFile.current !== file) setTrackFeatures(null);
    measuredFile.current = file;
    setTrackError(null);
    if (!file) return;

    const controller = new AbortController();
    setIsAnalyzingTrack(true);
    analyzeAudio(file, { songBpm, signal: controller.signal })
      .then(setTrackFeatures)
      .catch((err) => {
        if (!controller.signal.aborted) setTrackError(errorMessage(err));
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsAnalyzingTrack(false);
      });

    return () => controller.abort();
  }, [file, songBpm]);

  useEffect(() => {
    setRegionFeatures(null);
    setRegionError(null);
    setIsAnalyzingRegion(!!(file && region));
    if (!file || !region) return;

    const controller = new AbortController();
    // Wait for the selection to settle: resizing a region fires repeatedly.
    const timer = setTimeout(() => {
      analyzeAudio(file, { start: region.start, end: region.end, songBpm, signal: controller.signal })
        .then(setRegionFeatures)
        .catch((err) => {
          if (!controller.signal.aborted) setRegionError(errorMessage(err));
        })
        .finally(() => {
          if (!controller.signal.aborted) setIsAnalyzingRegion(false);
        });
    }, REGION_DEBOUNCE_MS);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [file, region, songBpm]);

  return {
    trackFeatures,
    regionFeatures,
    isAnalyzingTrack,
    isAnalyzingRegion,
    error: trackError ?? regionError,
  };
}

function errorMessage(err: unknown) {
  return err instanceof Error ? err.message : "Analysis failed";
}
