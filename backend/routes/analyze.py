import dataclasses
import json
import logging
import os
import tempfile
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from pydantic import ValidationError

from models.schemas import AudioEditList, AudioFeatures, AudioRegion
from services import audio_store
from services.audio_editing import EditError, apply_edits, estimate_edits
from services.audio_analysis import AudioDecodeError, InvalidRegionError, analyze_audio, to_playable_wav

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".aiff", ".aif", ".flac"}
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "50")) * 1024 * 1024


async def read_upload(file: UploadFile) -> tuple[bytes, str]:
    """Validate an upload and return its content and lowercase extension."""
    # Browsers report inconsistent MIME types for audio, so go by extension.
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported: {supported}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_SIZE // 1024 // 1024}MB.")
    return content, suffix


@router.post("/preview")
async def preview_endpoint(file: UploadFile = File(...)):
    """Convert audio to WAV for formats browsers can't decode themselves, such as AIFF in Chrome."""
    content, _ = await read_upload(file)
    try:
        wav = await run_in_threadpool(to_playable_wav, content)
    except AudioDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=wav, media_type="audio/wav")


def parse_edit_request(edits: str, start_time: float | None, end_time: float | None):
    try:
        parsed = AudioEditList.validate_json(edits)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="These edits aren't valid.") from exc
    region = AudioRegion(start=start_time, end=end_time) if start_time is not None and end_time is not None else None
    return parsed, region


@router.post("/edit/estimate")
async def estimate_endpoint(
    file: UploadFile = File(...),
    edits: str = Form(...),
    start_time: float | None = Form(None),
    end_time: float | None = Form(None),
):
    """Measure what edits would do to loudness and peak level, without keeping the result."""
    content, _ = await read_upload(file)
    parsed, region = parse_edit_request(edits, start_time, end_time)
    try:
        return dataclasses.asdict(await run_in_threadpool(estimate_edits, content, parsed, region))
    except (AudioDecodeError, InvalidRegionError, EditError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Edit estimate failed")
        raise HTTPException(status_code=500, detail="Estimating the edits failed unexpectedly.") from exc


@router.post("/edit")
async def edit_endpoint(
    file: UploadFile = File(...),
    edits: str = Form(...),
    start_time: float | None = Form(None),
    end_time: float | None = Form(None),
):
    """Apply edits to a copy of the audio. Returns a WAV, with a summary in the X-Edit-Report header."""
    content, _ = await read_upload(file)
    parsed, region = parse_edit_request(edits, start_time, end_time)

    try:
        wav, report = await run_in_threadpool(apply_edits, content, parsed, region)
    except (AudioDecodeError, InvalidRegionError, EditError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Audio edit failed")
        raise HTTPException(status_code=500, detail="Applying the edits failed unexpectedly.") from exc

    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"X-Edit-Report": json.dumps(dataclasses.asdict(report))},
    )


@router.post("/analyze", response_model=AudioFeatures)
async def analyze_endpoint(
    file: UploadFile = File(...),
    start_time: float | None = Form(None),
    end_time: float | None = Form(None),
    remember: bool = Form(True),  # False keeps a bulk analysis from evicting the track being worked on
    song_bpm: float | None = Form(None, ge=20, le=300),  # measures timing against this instead of a detected tempo
):
    content, suffix = await read_upload(file)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    session_id = str(uuid.uuid4())
    try:
        # Analysis is CPU-bound; a thread keeps the server responsive to other requests.
        features = await run_in_threadpool(
            analyze_audio,
            audio_path=tmp_path,
            start_time=start_time,
            end_time=end_time,
            session_id=session_id,
            song_bpm=song_bpm,
        )
        if remember:
            audio_store.remember(session_id, content)
        return features
    except (AudioDecodeError, InvalidRegionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Audio analysis failed")
        raise HTTPException(status_code=500, detail="Audio analysis failed unexpectedly.") from exc
    finally:
        os.unlink(tmp_path)
