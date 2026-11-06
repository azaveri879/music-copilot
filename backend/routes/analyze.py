import os
import uuid
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from models.schemas import AudioFeatures
from services.audio_analysis import analyze_audio

router = APIRouter()

ALLOWED_TYPES = {"audio/wav", "audio/mpeg", "audio/mp3", "audio/aiff", "audio/x-aiff", "audio/flac", "audio/x-flac"}
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "50")) * 1024 * 1024


@router.post("/analyze", response_model=AudioFeatures)
async def analyze_endpoint(
    file: UploadFile = File(...),
    start_time: float | None = Form(None),
    end_time: float | None = Form(None),
):
    # Basic validation
    if file.content_type and file.content_type not in ALLOWED_TYPES:
        # Also allow by extension as a fallback
        name = file.filename or ""
        if not any(name.lower().endswith(ext) for ext in [".wav", ".mp3", ".aiff", ".aif", ".flac", ".m4a"]):
            raise HTTPException(status_code=400, detail="Unsupported file type")

    # Read file content and check size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_SIZE // 1024 // 1024}MB.")

    session_id = str(uuid.uuid4())

    # Write to a temp file for librosa to process
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        features = analyze_audio(
            audio_path=tmp_path,
            start_time=start_time,
            end_time=end_time,
            session_id=session_id,
        )
        return features
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")
    finally:
        os.unlink(tmp_path)
