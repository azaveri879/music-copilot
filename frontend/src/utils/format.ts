import { AudioEdit } from "../types";

export function formatTime(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

const EQ_LABELS = {
  peak: "EQ",
  low_shelf: "Low shelf",
  high_shelf: "High shelf",
  highpass: "High-pass",
  lowpass: "Low-pass",
};

export function describeEdit(edit: AudioEdit): string {
  const signed = (db: number) => `${db > 0 ? "+" : ""}${db} dB`;
  switch (edit.type) {
    case "gain":
      return `Gain ${signed(edit.gain_db)}`;
    case "eq": {
      const filterOnly = edit.eq_type === "highpass" || edit.eq_type === "lowpass";
      const q = edit.eq_type === "peak" ? `, Q ${edit.q}` : "";
      return `${EQ_LABELS[edit.eq_type]} at ${edit.freq_hz} Hz${filterOnly ? "" : ` ${signed(edit.gain_db)}${q}`}`;
    }
    case "compressor":
      return `Compressor ${edit.ratio}:1 at ${edit.threshold_db} dB, ${edit.attack_ms}ms / ${edit.release_ms}ms${
        edit.makeup_db ? `, makeup +${edit.makeup_db} dB` : ""
      }`;
    case "limiter":
      return `Limiter with ceiling ${edit.ceiling_db} dB${edit.input_gain_db ? `, drive +${edit.input_gain_db} dB` : ""}`;
    case "nudge":
      return `Move note at ${formatTime(edit.onset_time)} (${edit.onset_time.toFixed(2)}s) ${Math.abs(edit.shift_ms)}ms ${
        edit.shift_ms < 0 ? "earlier" : "later"
      }`;
  }
}
