import { AudioFeatures, ChatRequest } from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function analyzeAudio(
  file: File,
  startTime?: number,
  endTime?: number
): Promise<AudioFeatures> {
  const formData = new FormData();
  formData.append("file", file);
  if (startTime !== undefined) formData.append("start_time", String(startTime));
  if (endTime !== undefined) formData.append("end_time", String(endTime));

  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Audio analysis failed");
  }

  return res.json();
}

export async function* streamChat(
  request: ChatRequest
): AsyncGenerator<string> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Chat request failed");
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6).trim();
      if (data === "[DONE]") return;
      try {
        const parsed = JSON.parse(data);
        if (parsed.type === "delta" && parsed.content) {
          yield parsed.content;
        }
      } catch {
        // ignore malformed SSE chunks
      }
    }
  }
}
