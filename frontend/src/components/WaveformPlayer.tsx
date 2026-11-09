import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.js";
import { AudioRegion } from "../types";
import { fetchPlayableAudio } from "../utils/api";
import { formatTime } from "../utils/format";

interface WaveformPlayerProps {
  file: File | null;
  // When set, this plays and displays instead of the file. Switching keeps the playback position.
  editedAudio: Blob | null;
  region: AudioRegion | null;
  onRegionChange: (region: AudioRegion | null) => void;
}

export function WaveformPlayer({ file, editedAudio, region, onRegionChange }: WaveformPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<RegionsPlugin | null>(null);
  const loadedFileRef = useRef<File | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;

    const regions = RegionsPlugin.create();
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "#6366f1",
      progressColor: "#a5b4fc",
      cursorColor: "#e2e8f0",
      height: 80,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      plugins: [regions],
    });

    wavesurferRef.current = ws;
    regionsRef.current = regions;

    ws.on("ready", () => setDuration(ws.getDuration()));
    ws.on("timeupdate", (t) => setCurrentTime(t));
    ws.on("play", () => setIsPlaying(true));
    ws.on("pause", () => setIsPlaying(false));
    ws.on("finish", () => setIsPlaying(false));
    ws.on("error", () => setLoadError(true));

    regions.enableDragSelection({ color: "rgba(99, 102, 241, 0.2)" });

    regions.on("region-created", (created) => {
      // Only keep one region at a time
      regions.getRegions().forEach((r) => {
        if (r.id !== created.id) r.remove();
      });
      onRegionChange({ start: created.start, end: created.end });
    });

    regions.on("region-updated", (updated) => {
      onRegionChange({ start: updated.start, end: updated.end });
    });

    return () => {
      ws.destroy();
      wavesurferRef.current = null;
      regionsRef.current = null;
    };
  }, [onRegionChange]);

  useEffect(() => {
    if (!wavesurferRef.current) return;
    const ws = wavesurferRef.current;
    if (!file) {
      ws.pause(); // the player is hidden now, so nothing else could stop it
      return;
    }
    const isSameFile = loadedFileRef.current === file;
    loadedFileRef.current = file;
    const resumeAt = isSameFile ? ws.getCurrentTime() : 0;
    const resumePlaying = isSameFile && ws.isPlaying();

    setLoadError(false);
    if (!isSameFile) {
      setIsPlaying(false);
      setCurrentTime(0);
      setDuration(0);
    }

    const controller = new AbortController();
    let url: string | null = null;

    // Chrome and Firefox can't decode AIFF, a common Logic bounce format.
    const source = editedAudio
      ? Promise.resolve(editedAudio)
      : /\.aiff?$/i.test(file.name)
        ? fetchPlayableAudio(file, controller.signal)
        : Promise.resolve(file);
    source
      .then((audio) => {
        if (controller.signal.aborted) return;
        url = URL.createObjectURL(audio);
        return ws.load(url);
      })
      .then(() => {
        if (controller.signal.aborted || !isSameFile) return;
        ws.setTime(resumeAt);
        if (resumePlaying) ws.play();
      })
      .catch(() => {
        if (!controller.signal.aborted) setLoadError(true);
      });

    return () => {
      controller.abort();
      if (url) URL.revokeObjectURL(url);
    };
  }, [file, editedAudio]);

  // Clearing the selection from outside (new file, "Use full track") removes the highlight.
  useEffect(() => {
    if (!region) regionsRef.current?.clearRegions();
  }, [region]);

  return (
    // The container stays mounted: WaveSurfer attaches to it once, before any file exists.
    <div className={`bg-gray-900 rounded-xl p-4 space-y-3 ${file ? "" : "hidden"}`}>
      {loadError && (
        <p className="text-xs text-amber-300/80">
          Couldn't load this file for playback, so there's no waveform or region selection. Analysis and chat still
          cover the full track if the analysis above succeeded.
        </p>
      )}
      <div ref={containerRef} className={`w-full ${loadError ? "hidden" : ""}`} />
      {!loadError && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <button
            onClick={() => wavesurferRef.current?.playPause()}
            aria-label={isPlaying ? "Pause" : "Play"}
            className="w-8 h-8 flex items-center justify-center rounded-full bg-indigo-500 hover:bg-indigo-400 transition-colors text-white text-sm"
          >
            {isPlaying ? "⏸" : "▶"}
          </button>
          <span className="text-xs text-gray-400 font-mono">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
          <span className="text-xs text-gray-500">Click and drag on the waveform to select a region</span>
        </div>
      )}
    </div>
  );
}
