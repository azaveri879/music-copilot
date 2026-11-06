import { CSSProperties, useCallback, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import { ChatPanel } from "./components/ChatPanel";
import { WaveformPlayer } from "./components/WaveformPlayer";
import { useAudioAnalysis } from "./hooks/useAudioAnalysis";
import { useChat } from "./hooks/useChat";
import { useResizableWidth } from "./hooks/useResizableWidth";
import { AudioRegion } from "./types";
import { formatTime } from "./utils/format";

const SESSION_ID = uuidv4();

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [selectedRegion, setSelectedRegion] = useState<AudioRegion | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);

  const { trackFeatures, regionFeatures, isAnalyzingTrack, isAnalyzingRegion, error: analysisError } =
    useAudioAnalysis(file, selectedRegion);
  const { messages, isStreaming, error: chatError, sendMessage } = useChat(SESSION_ID);

  const { width: chatWidth, handleProps: chatResizeProps } = useResizableWidth();

  const audioFeatures = regionFeatures ?? trackFeatures;

  const handleFile = useCallback((f: File | undefined) => {
    if (!f) return;
    if (!isSupportedAudio(f)) {
      setFileError(`${f.name} isn't a supported format. Use WAV, MP3, AIFF or FLAC.`);
      return;
    }
    setFileError(null);
    setFile(f);
    setSelectedRegion(null);
  }, []);

  const handleSend = useCallback(
    // If region analysis failed, the answer is about the full track, so don't label it as the region.
    (text: string) => sendMessage(text, audioFeatures, regionFeatures ? selectedRegion : null),
    [sendMessage, audioFeatures, regionFeatures, selectedRegion]
  );

  const error = fileError ?? analysisError;

  return (
    <div className="h-screen bg-gray-950 text-gray-100 flex flex-col">
      {/* Top bar */}
      <header className="border-b border-gray-800 px-6 py-3 flex items-center gap-3">
        <span className="text-lg">🎛️</span>
        <span className="font-semibold text-gray-100">Music Copilot</span>
        <span className="text-xs text-gray-500">AI production assistant</span>
      </header>

      <div className="flex flex-col lg:flex-row flex-1 min-h-0 overflow-y-auto lg:overflow-hidden">
        {/* Left panel — audio */}
        <div className="flex-1 min-w-0 flex flex-col p-6 gap-4 lg:overflow-y-auto">
          {/* Upload zone */}
          <div
            onDrop={(e) => {
              e.preventDefault();
              handleFile(e.dataTransfer.files[0]);
            }}
            onDragOver={(e) => e.preventDefault()}
            className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer ${
              file ? "border-gray-700 bg-gray-900/50" : "border-gray-700 hover:border-indigo-500 bg-gray-900/30"
            }`}
            onClick={() => document.getElementById("file-input")?.click()}
          >
            <input
              id="file-input"
              type="file"
              accept=".wav,.mp3,.aif,.aiff,.flac"
              className="hidden"
              onClick={(e) => e.stopPropagation()}
              onChange={(e) => {
                handleFile(e.target.files?.[0]);
                e.target.value = ""; // so choosing the same file again still fires
              }}
            />
            {file ? (
              <div className="space-y-1">
                <p className="text-sm font-medium text-gray-300 break-all">{file.name}</p>
                <p className="text-xs text-gray-500">Click to replace · {(file.size / 1024 / 1024).toFixed(1)} MB</p>
              </div>
            ) : (
              <div className="space-y-2">
                <p className="text-3xl">🎵</p>
                <p className="text-sm text-gray-400">Drop an audio file here, or click to browse</p>
                <p className="text-xs text-gray-500">WAV, MP3, AIFF, FLAC supported</p>
              </div>
            )}
          </div>

          {/* Analysis status */}
          {isAnalyzingTrack && <div className="text-xs text-indigo-400 animate-pulse px-1">Analyzing audio…</div>}
          {error && <div className="text-xs text-red-400 bg-red-500/10 rounded-lg px-3 py-2">{error}</div>}

          {/* Audio features summary */}
          {audioFeatures && (
            <div className="space-y-2">
              <p className="text-xs text-gray-500 px-1">
                {regionFeatures && selectedRegion
                  ? `Selection ${formatTime(selectedRegion.start)} – ${formatTime(selectedRegion.end)}`
                  : "Full track"}
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  { label: "BPM", value: audioFeatures.bpm !== null ? Math.round(audioFeatures.bpm) : "—" },
                  { label: "Key", value: audioFeatures.key ?? "—" },
                  {
                    label: "Loudness",
                    value: audioFeatures.loudness_lufs !== null ? `${audioFeatures.loudness_lufs.toFixed(1)} LUFS` : "—",
                  },
                  { label: "Duration", value: formatTime(audioFeatures.duration) },
                ].map(({ label, value }) => (
                  <div key={label} className="bg-gray-900 rounded-lg px-3 py-2">
                    <p className="text-xs text-gray-500">{label}</p>
                    <p className="text-sm font-medium text-gray-200">{value}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Timing issues */}
          {audioFeatures && audioFeatures.timing_issues.length > 0 && (
            <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-3 space-y-1">
              <p className="text-xs font-medium text-amber-400">Possible timing issues</p>
              {audioFeatures.timing_issues.map((issue) => (
                <p key={issue.timestamp} className="text-xs text-amber-300/70">
                  {formatTime(issue.timestamp)} — {issue.description}
                </p>
              ))}
            </div>
          )}

          {/* Waveform */}
          <WaveformPlayer file={file} region={selectedRegion} onRegionChange={setSelectedRegion} />
        </div>

        {/* Right panel — chat */}
        <div
          {...chatResizeProps}
          aria-label="Resize chat panel"
          className="hidden lg:block w-1.5 flex-shrink-0 cursor-col-resize touch-none bg-gray-800 hover:bg-indigo-500 focus-visible:bg-indigo-500 focus:outline-none transition-colors"
        />
        <div
          style={{ "--chat-width": `${chatWidth}px` } as CSSProperties}
          className="w-full lg:w-[var(--chat-width)] h-[32rem] lg:h-auto flex-shrink-0 border-t lg:border-t-0 border-gray-800 p-4 min-h-0"
        >
          <ChatPanel
            messages={messages}
            isStreaming={isStreaming}
            error={chatError}
            audioFeatures={audioFeatures}
            selectedRegion={selectedRegion}
            isAnalyzingRegion={isAnalyzingRegion}
            onClearRegion={() => setSelectedRegion(null)}
            onSend={handleSend}
          />
        </div>
      </div>
    </div>
  );
}

function isSupportedAudio(f: File) {
  return /\.(wav|mp3|aiff?|flac)$/i.test(f.name);
}
