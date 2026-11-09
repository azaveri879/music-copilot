export const MIN_SONG_BPM = 20;
export const MAX_SONG_BPM = 300;

export function parseSongBpm(text: string) {
  const bpm = Number(text);
  return text !== "" && bpm >= MIN_SONG_BPM && bpm <= MAX_SONG_BPM ? bpm : null;
}
