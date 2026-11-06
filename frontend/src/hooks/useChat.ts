import { useState, useCallback } from "react";
import { v4 as uuidv4 } from "uuid";
import { ChatMessage, AudioFeatures, AudioRegion } from "../types";
import { streamChat } from "../utils/api";

export function useChat(sessionId: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(
    async (
      text: string,
      audioFeatures: AudioFeatures | null,
      region: AudioRegion | null
    ) => {
      if (!text.trim() || isStreaming) return;

      setError(null);

      const userMessage: ChatMessage = {
        id: uuidv4(),
        role: "user",
        content: text,
        timestamp: new Date(),
        region: region ?? undefined,
      };

      setMessages((prev) => [...prev, userMessage]);

      const assistantMessage: ChatMessage = {
        id: uuidv4(),
        role: "assistant",
        content: "",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setIsStreaming(true);

      try {
        const history = messages.map((m) => ({
          role: m.role,
          content: m.content,
        }));

        const stream = streamChat({
          session_id: sessionId,
          message: text,
          audio_features: audioFeatures,
          region,
          history,
        });

        for await (const chunk of stream) {
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last.role === "assistant") {
              updated[updated.length - 1] = {
                ...last,
                content: last.content + chunk,
              };
            }
            return updated;
          });
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Something went wrong";
        setError(msg);
        // Drop the bubble only if nothing arrived; a partial answer is still worth keeping.
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          return last?.role === "assistant" && last.content === "" ? prev.slice(0, -1) : prev;
        });
      } finally {
        setIsStreaming(false);
      }
    },
    [messages, sessionId, isStreaming]
  );

  const clearMessages = useCallback(() => setMessages([]), []);

  return { messages, isStreaming, error, sendMessage, clearMessages };
}
