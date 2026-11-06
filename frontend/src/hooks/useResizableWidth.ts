import { KeyboardEvent, PointerEvent, useCallback, useEffect, useRef, useState } from "react";

const STORAGE_KEY = "music-copilot:chat-width";
const DEFAULT_WIDTH = 384;
const MIN_WIDTH = 320;
const MIN_REMAINING = 420; // keep the audio panel usable
const KEYBOARD_STEP = 24;

function clamp(width: number) {
  return Math.max(MIN_WIDTH, Math.min(width, window.innerWidth - MIN_REMAINING));
}

function loadWidth() {
  try {
    const saved = Number(localStorage.getItem(STORAGE_KEY));
    return saved ? clamp(saved) : DEFAULT_WIDTH;
  } catch {
    return DEFAULT_WIDTH;
  }
}

/** Width of a panel on the right edge of the window, set by dragging a handle on its left. */
export function useResizableWidth() {
  const [width, setWidth] = useState(loadWidth);
  const dragging = useRef(false);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, String(width));
    } catch {
      // Not being able to remember the width is harmless.
    }
  }, [width]);

  useEffect(() => {
    const onResize = () => setWidth(clamp);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const handleProps = {
    role: "separator",
    "aria-orientation": "vertical" as const,
    "aria-valuenow": width,
    tabIndex: 0,
    onPointerDown: useCallback((e: PointerEvent<HTMLElement>) => {
      e.preventDefault(); // don't select text while dragging
      e.currentTarget.setPointerCapture(e.pointerId);
      dragging.current = true;
    }, []),
    onPointerMove: useCallback((e: PointerEvent<HTMLElement>) => {
      if (dragging.current) setWidth(clamp(window.innerWidth - e.clientX));
    }, []),
    onPointerUp: useCallback(() => {
      dragging.current = false;
    }, []),
    onPointerCancel: useCallback(() => {
      dragging.current = false;
    }, []),
    onDoubleClick: useCallback(() => setWidth(clamp(DEFAULT_WIDTH)), []),
    onKeyDown: useCallback((e: KeyboardEvent<HTMLElement>) => {
      if (e.key === "ArrowLeft") setWidth((w) => clamp(w + KEYBOARD_STEP));
      if (e.key === "ArrowRight") setWidth((w) => clamp(w - KEYBOARD_STEP));
    }, []),
  };

  return { width, handleProps };
}
