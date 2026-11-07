import ReactMarkdown from "react-markdown";
import { ChatMessage } from "../types";
import { formatTime } from "../utils/format";
import { EditProposalCard, ProposalHandlers } from "./EditProposalCard";

interface MessageBubbleProps {
  message: ChatMessage;
  proposalHandlers: ProposalHandlers;
}

export function MessageBubble({ message, proposalHandlers }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] min-w-0 rounded-xl px-3 py-2.5 text-sm leading-relaxed ${
          isUser ? "bg-indigo-600 text-white" : "bg-gray-800 text-gray-200"
        }`}
      >
        {message.region && (
          <div className="text-xs opacity-60 mb-1">
            Region: {formatTime(message.region.start)} – {formatTime(message.region.end)}
          </div>
        )}
        {isUser ? (
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        ) : message.content === "" && !message.proposal ? (
          <span className="animate-pulse opacity-50">▌</span>
        ) : (
          <div className="markdown">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}
        {message.proposal && <EditProposalCard message={message} handlers={proposalHandlers} />}
      </div>
    </div>
  );
}
