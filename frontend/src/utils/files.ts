export function isSupportedAudio(f: File) {
  return /\.(wav|mp3|aiff?|flac)$/i.test(f.name);
}

export function trackName(f: File) {
  return f.name.replace(/\.[^.]+$/, "");
}

export function compareByName(a: File, b: File) {
  return a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" });
}

function isVisible(f: File) {
  return !f.name.startsWith(".");
}

function readFile(entry: FileSystemFileEntry) {
  return new Promise<File>((resolve, reject) => entry.file(resolve, reject));
}

async function readFolder(entry: FileSystemDirectoryEntry) {
  const reader = entry.createReader();
  const entries: FileSystemEntry[] = [];
  // Browsers hand back a directory in batches, and an empty batch marks the end.
  for (;;) {
    const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => reader.readEntries(resolve, reject));
    if (batch.length === 0) break;
    entries.push(...batch);
  }
  // Only the folder's own files: a subfolder might hold older exports of the same tracks.
  return Promise.all(entries.filter((e) => e.isFile).map((e) => readFile(e as FileSystemFileEntry)));
}

/** The files in a drop, including the contents of any dropped folders. */
export async function filesFromDrop(data: DataTransfer): Promise<File[]> {
  // DataTransfer items stop being readable once the drop handler yields, so collect these first.
  const entries = Array.from(data.items)
    .map((item) => item.webkitGetAsEntry())
    .filter((entry): entry is FileSystemEntry => entry !== null);
  if (entries.length === 0) return Array.from(data.files).filter(isVisible);

  const nested = await Promise.all(
    entries.map((entry) =>
      entry.isDirectory
        ? readFolder(entry as FileSystemDirectoryEntry)
        : readFile(entry as FileSystemFileEntry).then((f) => [f])
    )
  );
  return nested.flat().filter(isVisible);
}

/** The files from a folder picker, leaving out subfolders as a dropped folder does. */
export function filesFromFolderPicker(list: FileList): File[] {
  return Array.from(list).filter((f) => f.webkitRelativePath.split("/").length === 2 && isVisible(f));
}
