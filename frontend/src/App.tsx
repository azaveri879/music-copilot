import { useState, useCallback } from "react";
import { v4 as uuidv4 } from "uuid";
import { WaveformPlayer } from "./components/WaveformPlayer";
import { ChatPanel } from "./components/ChatPanel";
import { useChat } from "./hooks/useChat";
import { AudioFeatures, AudioRegion } from "./types";
import { analyzeAudio } from "./utils/api";

const SESSION_ID = uuidv4();

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [audioFeatures, setAudioFeatures] = useState<AudioFeatures | null>(null);
  const [selectedRegion, setSelectedRegion] = useState<AudioRegion | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);

  const { messages, isStreaming, error, sendMessage } = useChat(SESSION_ID);

  const handleFileUpload = useCallback(async (f: File) => {
    setFile(f);
    setAudioFeatures(null);
    setSelectedRegion(null);
    setAnalyzeError(null);
    setIsAnalyzing(true);

    try {
      const features = await analyzeAudio(f);
      setAudioFeatures(features);
    } catch (err) {
      setAnalyzeError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const dropped = e.dataTransfer.files[0];
      if (dropped && isAudioFile(dropped)) handleFileUpload(dropped);
    },
    [handleFileUpload]
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selected = e.target.files?.[0];
      if (selected) handleFileUpload(selected);
    },
    [handleFileUpload]
  );

  const handleSend = useCallback(
    (text: string) => sendMessage(text, audioFeatures, selectedRegion),
    [sendMessage, audioFeatures, selectedRegion]
  );

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      {/* Top bar */}
      <header className="border-b border-gray-800 px-6 py-3 flex items-center gap-3">
        <span className="text-lg">🎛️</span>
        <span className="font-semibold text-gray-100">Music Copilot</span>
        <span className="text-xs text-gray-600">AI production assistant</span>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Left panel — audio */}
        <div className="flex-1 flex flex-col p-6 gap-4 overflow-y-auto">
          {/* Upload zone */}
          <div
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer ${
              file
                ? "border-gray-700 bg-gray-900/50"
                : "border-gray-700 hover:border-indigo-500 bg-gray-900/30"
            }`}
            onClick={() => document.getElementById("file-input")?.click()}
          >
            <input
              id="file-input"
              type="file"
              accept="audio/*"
              className="hidden"
              onChange={handleInputChange}
            />
            {file ? (
              <div className="space-y-1">
                <p className="text-sm font-medium text-gray-300">{file.name}</p>
                <p className="text-xs text-gray-500">
                  Click to replace · {(file.size / 1024 / 1024).toFixed(1)} MB
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                <p className="text-3xl">🎵</p>
                <p className="text-sm text-gray-400">
                  Drop an audio file here, or click to browse
                </p>
                <p className="text-xs text-gray-600">
                  WAV, MP3, AIFF, FLAC supported
                </p>
              </div>
            )}
          </div>

          {/* Analysis status */}
          {isAnalyzing && (
            <div className="text-xs text-indigo-400 animate-pulse px-1">
              Analyzing audio…
            </div>
          )}
          {analyzeError && (
            <div className="text-xs text-red-400 bg-red-500/10 rounded-lg px-3 py-2">
              {analyzeError}
            </div>
          )}

          {/* Audio features summary */}
          {audioFeatures && (
            <div className="grid grid-cols-4 gap-3">
              {[
                { label: "BPM", value: audioFeatures.bpm ? Math.round(audioFeatures.bpm) : "—" },
                { label: "Key", value: audioFeatures.key ?? "—" },
                { label: "Loudness", value: audioFeatures.loudness_lufs ? `${audioFeatures.loudness_lufs.toFixed(1)} LUFS` : "—" },
                { label: "Duration", value: formatDuration(audioFeatures.duration) },
              ].map(({ label, value }) => (
                <div key={label} className="bg-gray-900 rounded-lg px-3 py-2">
                  <p className="text-xs text-gray-500">{label}</p>
                  <p className="text-sm font-medium text-gray-200">{value}</p>
                </div>
              ))}
            </div>
          )}

          {/* Timing issues */}
          {audioFeatures?.timing_issues?.length > 0 && (
            <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-3 space-y-1">
              <p className="text-xs font-medium text-amber-400">
                Timing issues detected
              </p>
              {audioFeatures.timing_issues.map((issue, i) => (
                <p key={i} className="text-xs text-amber-300/70">
                  {formatDuration(issue.timestamp)} — {issue.description}
                </p>
              ))}
            </div>
          )}

          {/* Waveform */}
          <WaveformPlayer file={file} onRegionChange={setSelectedRegion} />
        </div>

        {/* Right panel — chat */}
        <div className="w-96 flex-shrink-0 border-l border-gray-800 p-4">
          <ChatPanel
            messages={messages}
            isStreaming={isStreaming}
            error={error}
            audioFeatures={audioFeatures}
            selectedRegion={selectedRegion}
            onSend={handleSend}
          />
        </div>
      </div>
    </div>
  );
}

function isAudioFile(f: File) {
  return f.type.startsWith("audio/") || /\.(wav|mp3|aiff?|flac|m4a|ogg)$/i.test(f.name);
}

function formatDuration(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}
