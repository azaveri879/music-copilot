import logging
import os
import tempfile
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from models.schemas import AudioFeatures
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


@router.post("/analyze", response_model=AudioFeatures)
async def analyze_endpoint(
    file: UploadFile = File(...),
    start_time: float | None = Form(None),
    end_time: float | None = Form(None),
):
    content, suffix = await read_upload(file)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Analysis is CPU-bound; a thread keeps the server responsive to other requests.
        return await run_in_threadpool(
            analyze_audio,
            audio_path=tmp_path,
            start_time=start_time,
            end_time=end_time,
            session_id=str(uuid.uuid4()),
        )
    except (AudioDecodeError, InvalidRegionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Audio analysis failed")
        raise HTTPException(status_code=500, detail="Audio analysis failed unexpectedly.") from exc
    finally:
        os.unlink(tmp_path)
