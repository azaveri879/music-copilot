import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from models.schemas import ChatRequest
from services.ai_service import stream_chat_response

router = APIRouter()


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    async def event_stream():
        try:
            async for chunk in stream_chat_response(
                message=request.message,
                audio_features=request.audio_features,
                region=request.region,
                history=request.history,
            ):
                payload = json.dumps({"type": "delta", "content": chunk})
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_payload = json.dumps({"type": "error", "content": str(e)})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
