import { AudioEdit, AudioFeatures, AudioRegion, ChatRequest, ChatStreamEvent, EditEstimate, EditReport } from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const UNREACHABLE = `Can't reach the backend at ${API_BASE}. Is it running?`;

interface AnalyzeOptions {
  start?: number;
  end?: number;
  remember?: boolean;
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

export async function analyzeAudio(file: File, { start, end, remember, signal }: AnalyzeOptions = {}): Promise<AudioFeatures> {
  const formData = new FormData();
  formData.append("file", file);
  if (start !== undefined) formData.append("start_time", String(start));
  if (end !== undefined) formData.append("end_time", String(end));
  if (remember !== undefined) formData.append("remember", String(remember));

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

function editForm(file: File, edits: AudioEdit[], region: AudioRegion | null) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("edits", JSON.stringify(edits));
  if (region) {
    formData.append("start_time", String(region.start));
    formData.append("end_time", String(region.end));
  }
  return formData;
}

export async function estimateEdits(
  file: File,
  edits: AudioEdit[],
  region: AudioRegion | null,
  signal: AbortSignal
): Promise<EditEstimate> {
  const res = await post("/edit/estimate", { body: editForm(file, edits, region), signal }, "Couldn't estimate these edits");
  return res.json();
}

export async function applyEdits(
  file: File,
  edits: AudioEdit[],
  region: AudioRegion | null
): Promise<{ blob: Blob; report: EditReport }> {
  const formData = editForm(file, edits, region);
  const res = await post("/edit", { body: formData }, "Applying the edits failed");
  return { blob: await res.blob(), report: JSON.parse(res.headers.get("X-Edit-Report") ?? "{}") };
}

export async function* streamChat(request: ChatRequest): AsyncGenerator<ChatStreamEvent> {
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
      if (event.type === "delta") yield { type: "text", text: event.content };
      else if (event.type === "proposal") yield { type: "proposal", proposal: event.proposal };
      else if (event.type === "error") throw new Error(event.content);
      else if (event.type === "done") return;
    }
  }
}
