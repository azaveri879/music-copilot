import { ChatMessage } from "../types";

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-xl px-3 py-2.5 text-sm leading-relaxed ${
          isUser
            ? "bg-indigo-600 text-white"
            : "bg-gray-800 text-gray-200"
        }`}
      >
        {message.region && (
          <div className="text-xs opacity-60 mb-1">
            Region: {formatTime(message.region.start)} – {formatTime(message.region.end)}
          </div>
        )}
        {/* Render newlines from markdown-ish text */}
        {message.content.split("\n").map((line, i) => (
          <span key={i}>
            {line}
            {i < message.content.split("\n").length - 1 && <br />}
          </span>
        ))}
        {message.content === "" && (
          <span className="animate-pulse opacity-50">▌</span>
        )}
      </div>
    </div>
  );
}

function formatTime(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}
