import { useRef, useEffect, useState, KeyboardEvent } from "react";
import { ChatMessage, AudioFeatures, AudioRegion } from "../types";
import { MessageBubble } from "./MessageBubble";

interface ChatPanelProps {
  messages: ChatMessage[];
  isStreaming: boolean;
  error: string | null;
  audioFeatures: AudioFeatures | null;
  selectedRegion: AudioRegion | null;
  onSend: (text: string) => void;
}

export function ChatPanel({
  messages,
  isStreaming,
  error,
  audioFeatures,
  selectedRegion,
  onSend,
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    if (!input.trim() || isStreaming) return;
    onSend(input.trim());
    setInput("");
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  return (
    <div className="flex flex-col h-full bg-gray-950 rounded-xl overflow-hidden border border-gray-800">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-200">Copilot</span>
          {audioFeatures && (
            <span className="text-xs text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded-full">
              {audioFeatures.bpm ? `${Math.round(audioFeatures.bpm)} BPM` : ""}
              {audioFeatures.key ? ` · ${audioFeatures.key}` : ""}
            </span>
          )}
        </div>
        {selectedRegion && (
          <span className="text-xs text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded-full">
            {formatTime(selectedRegion.start)} – {formatTime(selectedRegion.end)}
          </span>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center space-y-3 py-12">
            <p className="text-gray-500 text-sm">Upload a track to get started.</p>
            <div className="text-xs text-gray-600 space-y-1">
              <p>Try asking:</p>
              <p className="italic">"This guitar feels off beat in the intro"</p>
              <p className="italic">"I want this to feel dark and lo-fi"</p>
              <p className="italic">"The vocals are too harsh — what should I cut?"</p>
            </div>
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {error && (
          <div className="text-xs text-red-400 bg-red-500/10 rounded-lg px-3 py-2">
            {error}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-3 border-t border-gray-800">
        {!audioFeatures && (
          <p className="text-xs text-gray-600 mb-2 text-center">
            Upload a track above before asking questions
          </p>
        )}
        <div className="flex gap-2 items-end">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={!audioFeatures || isStreaming}
            placeholder={
              audioFeatures
                ? selectedRegion
                  ? "Ask about the selected region…"
                  : "Describe what's wrong or what vibe you're going for…"
                : "Upload a track to start"
            }
            rows={2}
            className="flex-1 resize-none bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-indigo-500 disabled:opacity-40 transition-colors"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || !audioFeatures || isStreaming}
            className="px-3 py-2 bg-indigo-500 hover:bg-indigo-400 disabled:opacity-30 disabled:cursor-not-allowed text-white rounded-lg text-sm transition-colors h-fit"
          >
            {isStreaming ? "…" : "→"}
          </button>
        </div>
        <p className="text-xs text-gray-700 mt-1.5">
          Shift+Enter for new line · Enter to send
        </p>
      </div>
    </div>
  );
}
