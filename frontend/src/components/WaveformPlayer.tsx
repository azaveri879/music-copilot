import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.js";
import { AudioRegion } from "../types";

interface WaveformPlayerProps {
  file: File | null;
  onRegionChange: (region: AudioRegion | null) => void;
}

export function WaveformPlayer({ file, onRegionChange }: WaveformPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

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

    ws.on("ready", () => setDuration(ws.getDuration()));
    ws.on("timeupdate", (t) => setCurrentTime(t));
    ws.on("play", () => setIsPlaying(true));
    ws.on("pause", () => setIsPlaying(false));
    ws.on("finish", () => setIsPlaying(false));

    // Allow user to draw regions by clicking and dragging
    regions.enableDragSelection({ color: "rgba(99, 102, 241, 0.2)" });

    regions.on("region-created", (region) => {
      // Only keep one region at a time
      const all = regions.getRegions();
      all.forEach((r) => { if (r.id !== region.id) r.remove(); });
      onRegionChange({ start: region.start, end: region.end });
    });

    regions.on("region-updated", (region) => {
      onRegionChange({ start: region.start, end: region.end });
    });

    regions.on("region-removed", () => {
      if (regions.getRegions().length === 0) onRegionChange(null);
    });

    return () => ws.destroy();
  }, []);

  useEffect(() => {
    if (!wavesurferRef.current || !file) return;
    const url = URL.createObjectURL(file);
    wavesurferRef.current.load(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const togglePlay = () => wavesurferRef.current?.playPause();

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  if (!file) return null;

  return (
    <div className="bg-gray-900 rounded-xl p-4 space-y-3">
      <div ref={containerRef} className="w-full" />
      <div className="flex items-center gap-4">
        <button
          onClick={togglePlay}
          className="w-8 h-8 flex items-center justify-center rounded-full bg-indigo-500 hover:bg-indigo-400 transition-colors text-white text-sm"
        >
          {isPlaying ? "⏸" : "▶"}
        </button>
        <span className="text-xs text-gray-400 font-mono">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
        <span className="text-xs text-gray-500">
          Click and drag on the waveform to select a region
        </span>
      </div>
    </div>
  );
}
