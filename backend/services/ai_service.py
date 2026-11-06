import logging
from functools import lru_cache
from typing import AsyncGenerator

import anthropic

from models.schemas import AudioFeatures, AudioRegion, ChatHistoryItem

logger = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"
MAX_TOKENS = 64000
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """You are Music Copilot, an expert AI production assistant for musicians and producers who use Logic Pro.

Your role is to bridge the gap between creative vision and technical execution — like having a world-class mixing and production engineer looking over the user's shoulder. You are given structured analysis of the user's audio (tempo, key, timing, frequency balance, loudness) and you use it to give specific, grounded, actionable advice.

## Your principles

**Be specific, not generic.** Don't say "try adding some reverb." Say "Open Space Designer, choose a large hall impulse response, set decay to around 2.8s and wet to about 25%, and roll off everything above 8kHz so it doesn't get washy."

**Respect the user's creative intent.** When they describe a vibe — "late-night", "lo-fi", "underwater", "warm" — treat it as a real design brief, not a vague request. Translate it into concrete production decisions.

**Reference the audio data, and be honest about its limits.** When the analysis shows something relevant, say so. The figures are automated estimates of a bounced file: you can't hear the audio, see the session, or tell which instrument made a sound. A detected key can land on the relative major or minor, tempo can be read at half or double time, and flagged timing issues are candidates to check by ear rather than confirmed mistakes. Never claim to have heard something the data doesn't show.

**Respect the scope.** When the analysis is for a selected region, the tempo and beat grid come from the whole track while everything else describes that region. Timestamps are positions in the file, in seconds.

**Explain the why briefly.** For each major suggestion, give a one-sentence reason. This helps the user learn and builds trust.

**Use Logic Pro specifics.** Name real stock tools (Channel EQ, Linear Phase EQ, Compressor, Multipressor, DeEsser 2, Adaptive Limiter, Space Designer, ChromaVerb, Tape Delay, Phat FX, Bitcrusher, Flex Time, Flex Pitch, Smart Tempo), real menu locations, and approximate parameter values. If you're not sure a parameter or preset exists under a given name, describe what to adjust rather than inventing a name.

**Be conversational.** You're a production partner, not a manual. Keep it tight and readable.

## Format guidelines

- Use short paragraphs or numbered steps, not walls of text
- When listing plugin settings, use a simple format: `Space Designer → Decay: 2.8s, Wet: 25%`
- Keep responses under 300 words unless the complexity warrants more
- If a question is ambiguous, make a reasonable assumption and state it, then answer
"""


class ChatError(Exception):
    """A chat failure with a message that is safe and useful to show the user."""


@lru_cache
def get_client() -> anthropic.AsyncAnthropic:
    # Created on first use so the server can start, and analysis can work, without credentials.
    return anthropic.AsyncAnthropic()


def build_audio_context(features: AudioFeatures, region: AudioRegion | None) -> str:
    """Format audio analysis data as a readable context block for the AI."""
    lines = ["## Audio Analysis"]

    if region:
        lines.append(f"**Selected region:** {region.start:.1f}s – {region.end:.1f}s ({features.duration:.1f}s long)")
    else:
        lines.append(f"**Duration:** {features.duration:.1f}s (full track)")

    if features.bpm is not None:
        lines.append(f"**BPM:** {features.bpm}" + (" (whole track)" if region else ""))
    if features.key is not None:
        lines.append(f"**Key:** {features.key}")
    if features.loudness_lufs is not None:
        lines.append(f"**Loudness:** {features.loudness_lufs} LUFS integrated")
    if features.dynamic_range_db is not None:
        lines.append(f"**Dynamic range:** {features.dynamic_range_db} dB (95th minus 10th percentile short-term level)")
    if features.spectral_centroid_mean is not None:
        lines.append(f"**Spectral centroid:** {features.spectral_centroid_mean:.0f} Hz (brightness indicator)")

    if features.frequency_profile:
        fp = features.frequency_profile
        lines.append(
            f"**Frequency balance:** Low (<250Hz) {fp.low_energy:.0%} / Mid {fp.mid_energy:.0%} / High (>4kHz) {fp.high_energy:.0%}"
        )

    if features.timing_issues:
        lines.append(f"\n**Possible timing issues ({len(features.timing_issues)}):**")
        lines.extend(f"  - {issue.timestamp:.2f}s: {issue.description}" for issue in features.timing_issues)

    return "\n".join(lines)


async def stream_chat_response(
    message: str,
    audio_features: AudioFeatures | None,
    region: AudioRegion | None,
    history: list[ChatHistoryItem],
) -> AsyncGenerator[str, None]:
    """
    Stream a response from Claude, with audio analysis as context.
    Yields text chunks as they arrive; raises ChatError on failure.
    """
    # An interrupted reply can leave an empty assistant turn, which the API rejects.
    messages = [{"role": item.role, "content": item.content} for item in history if item.content.strip()]

    user_content = message
    if audio_features:
        audio_ctx = build_audio_context(audio_features, region)
        user_content = f"{audio_ctx}\n\n---\n\n**User question:** {message}"

    messages.append({"role": "user", "content": user_content})

    try:
        async with get_client().beta.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
            output_config={"effort": "medium"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
    except anthropic.AuthenticationError as exc:
        raise ChatError("Anthropic rejected the API key. Check ANTHROPIC_API_KEY in backend/.env.") from exc
    except anthropic.RateLimitError as exc:
        raise ChatError("Rate limited by the Anthropic API. Wait a moment and try again.") from exc
    except anthropic.APIConnectionError as exc:
        raise ChatError("Couldn't reach the Anthropic API. Check your connection and try again.") from exc
    except anthropic.APIStatusError as exc:
        logger.error("Anthropic API error %s: %s", exc.status_code, exc.message)
        raise ChatError(f"The Anthropic API returned an error ({exc.status_code}). Try again.") from exc
    except TypeError as exc:
        # The SDK raises TypeError when it can't resolve any credentials.
        if "authentication" not in str(exc).lower() and "api_key" not in str(exc).lower():
            raise
        raise ChatError("No Anthropic API key found. Set ANTHROPIC_API_KEY in backend/.env and restart.") from exc

    if final.stop_reason == "refusal":
        raise ChatError("Claude declined to answer that. Try rephrasing your question.")
