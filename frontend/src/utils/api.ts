import { AudioFeatures, ChatRequest } from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const UNREACHABLE = `Can't reach the backend at ${API_BASE}. Is it running?`;

interface AnalyzeOptions {
  start?: number;
  end?: number;
  signal?: AbortSignal;
}

async function post(path: string, init: RequestInit, fallbackError: string) {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { method: "POST", ...init });
  } catch (err) {
    if (init.signal?.aborted) throw err;
    throw new Error(UNREACHABLE);
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : fallbackError);
  }
  return res;
}

export async function analyzeAudio(file: File, { start, end, signal }: AnalyzeOptions = {}): Promise<AudioFeatures> {
  const formData = new FormData();
  formData.append("file", file);
  if (start !== undefined) formData.append("start_time", String(start));
  if (end !== undefined) formData.append("end_time", String(end));

  const res = await post("/analyze", { body: formData, signal }, "Audio analysis failed");
  return res.json();
}

/** A WAV copy of the file, for formats the browser can't decode itself. */
export async function fetchPlayableAudio(file: File, signal: AbortSignal): Promise<Blob> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await post("/preview", { body: formData, signal }, "Couldn't prepare this file for playback");
  return res.blob();
}

export async function* streamChat(request: ChatRequest): AsyncGenerator<string> {
  const res = await post(
    "/chat",
    {
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    },
    "Chat request failed"
  );

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) throw new Error("The connection dropped before the response finished.");

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const event = JSON.parse(line.slice(6));
      if (event.type === "delta") yield event.content;
      else if (event.type === "error") throw new Error(event.content);
      else if (event.type === "done") return;
    }
  }
}
