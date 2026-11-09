import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from models.schemas import ChatRequest, EditProposal
from services.ai_service import ChatError, stream_chat_response

logger = logging.getLogger(__name__)

router = APIRouter()


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    async def event_stream():
        try:
            async for chunk in stream_chat_response(
                message=request.message,
                audio_features=request.audio_features,
                region=request.region,
                history=request.history,
                file_type=request.file_type,
                song_bpm=request.song_bpm,
                session_tracks=request.session_tracks,
                focused_track=request.focused_track,
            ):
                if isinstance(chunk, EditProposal):
                    yield sse({"type": "proposal", "proposal": chunk.model_dump()})
                else:
                    yield sse({"type": "delta", "content": chunk})
            yield sse({"type": "done"})
        except ChatError as exc:
            yield sse({"type": "error", "content": str(exc)})
        except Exception:
            logger.exception("Chat stream failed")
            yield sse({"type": "error", "content": "Something went wrong generating a response."})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
