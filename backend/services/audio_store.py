"""
Keeps recently analyzed audio in memory, so chat can measure proposed edits on the
real file. Chat only receives analysis numbers, and each analysis has its own session
id, so regions of one file share a single copy. Nothing is written to disk.
"""

from collections import OrderedDict
from hashlib import sha256

MAX_FILES = 2

_files: OrderedDict[str, bytes] = OrderedDict()
_sessions: dict[str, str] = {}


def remember(session_id: str, content: bytes) -> None:
    key = sha256(content).hexdigest()
    _files[key] = content
    _files.move_to_end(key)
    _sessions[session_id] = key

    while len(_files) > MAX_FILES:
        evicted, _ = _files.popitem(last=False)
        for session in [s for s, k in _sessions.items() if k == evicted]:
            del _sessions[session]


def recall(session_id: str) -> bytes | None:
    key = _sessions.get(session_id)
    return _files.get(key) if key else None


def clear() -> None:
    _files.clear()
    _sessions.clear()
