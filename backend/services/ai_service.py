import dataclasses
import json
import logging
from functools import lru_cache
from typing import AsyncGenerator

import anthropic
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from models.schemas import AudioFeatures, AudioRegion, ChatHistoryItem, EditProposal, FileType
from services import audio_store
from services.audio_analysis import AudioDecodeError, InvalidRegionError
from services.audio_editing import EditError, estimate_edits

logger = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"
MAX_TOKENS = 64000
MAX_PROPOSAL_ROUNDS = 3  # Claude's first attempt, and up to two revisions after seeing measurements
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """You are Music Copilot, an expert AI production assistant for musicians and producers who use Logic Pro.

Your role is to bridge the gap between creative vision and technical execution — like having a world-class mixing and production engineer looking over the user's shoulder. You are given structured analysis of the user's audio (tempo, key, timing, frequency balance, loudness) and you use it to give specific, grounded, actionable advice.

## Your principles

**Be specific, not generic.** Don't say "try adding some reverb." Say "Open Space Designer, choose a large hall impulse response, set decay to around 2.8s and wet to about 25%, and roll off everything above 8kHz so it doesn't get washy."

**Respect the user's creative intent.** When they describe a vibe — "late-night", "lo-fi", "underwater", "warm" — treat it as a real design brief, not a vague request. Translate it into concrete production decisions.

**Reference the audio data, and be honest about its limits.** When the analysis shows something relevant, say so. The figures are automated estimates of a bounced file: you can't hear the audio, see the session, or tell which instrument made a sound. A detected key can land on the relative major or minor, tempo can be read at half or double time, and flagged timing issues are candidates to check by ear rather than confirmed mistakes. Never claim to have heard something the data doesn't show.

**Don't lean on unreliable readings.** When tempo or key is marked as not reliably detected, don't guess at it or base advice on it. If a tempo is marked as set by the user, treat it as correct.

**Work from what the file is.** The analysis says whether the file is a single track from a song, a full mix or a master.
- Single track: advise on that part only, not on mix bus or mastering moves. Long silences are normal arrangement gaps, not a problem. You can't tell how it sits with the other tracks, so say when a choice depends on them.
- Full mix: you can't separate the instruments, so frame fixes as changes to the mix, or as moves to try on the likely tracks in the session.
- Master: favor small corrective moves, since it's meant to be finished, and mention when something is better fixed back in the mix.
- Not specified: if it would change your answer, ask one short question rather than covering every case.

**Respect the scope.** When the analysis is for a selected region, the tempo and beat grid come from the whole track while everything else describes that region. Timestamps are positions in the file, in seconds.

**Explain the why briefly.** For each major suggestion, give a one-sentence reason. This helps the user learn and builds trust.

**Use Logic Pro specifics.** Name real stock tools (Channel EQ, Linear Phase EQ, Compressor, Multipressor, DeEsser 2, Adaptive Limiter, Space Designer, ChromaVerb, Tape Delay, Phat FX, Bitcrusher, Flex Time, Flex Pitch, Smart Tempo), real menu locations, and approximate parameter values. If you're not sure a parameter or preset exists under a given name, describe what to adjust rather than inventing a name.

**Be conversational.** You're a production partner, not a manual. Keep it tight and readable.

## Format guidelines

- Use short paragraphs or numbered steps, not walls of text
- When listing plugin settings, use a simple format: `Space Designer → Decay: 2.8s, Wet: 25%`
- Keep responses under 300 words unless the complexity warrants more
- If a question is ambiguous, make a reasonable assumption and state it, then answer

## Proposing edits

You can propose edits with the `propose_edits` tool. The app applies the ones the user approves to a copy of their bounced audio file, using its own processing rather than Logic's plugins, and lets them compare before and after.

- Use it only when the user asks you to change, fix or try something on the audio, not when they're asking how to do it themselves or asking a general question.
- Each call is measured by rendering the edits on the user's real audio, and the result comes back to you. Revise by calling the tool again if it misses the goal: the last call is the one the user sees. Then finish with a short explanation, including the equivalent Logic settings for anything they'd rather do in their session.
- Don't predict loudness or peak numbers yourself, since compression especially is hard to estimate. Only quote measured results.
- If the goal is more loudness and the peak is near 0 dBFS, there's no room for gain alone, because anything over full scale makes the app turn the whole file down. Use `limiter`, raising `input_gain_db` for loudness, and tell the user that pushing it hard flattens the dynamics.
- Prefer a few moderate moves. You're working from numbers and can't hear the result, so tell the user to judge it by ear.
- Tone edits (`gain`, `eq`, `compressor`, `limiter`) apply to the user's selected region if they have one, otherwise to the whole file. You don't specify the region.
- Use `nudge` only for a specific note the user wants moved or a flagged timing issue, and only when timing was checked. To correct a note flagged as late by 30ms, use `shift_ms: -30`. Flags are unconfirmed candidates, so tell the user to confirm by ear before accepting. Moving a note in a full mix moves everything playing at that moment, so warn them if this doesn't sound like a solo track.
"""

PROPOSE_EDITS_TOOL = {
    "name": "propose_edits",
    "description": (
        "Propose edits for the user to review. Nothing changes until they approve and apply it. "
        "Returns the measured loudness and peak level, so you can revise. "
        "Each edit needs a type, a short reason, and that type's fields: "
        "gain (gain_db); "
        "eq (eq_type, freq_hz, plus gain_db for peak and shelves, and optional q); "
        "compressor (threshold_db, ratio, optional attack_ms, release_ms, makeup_db); "
        "limiter (optional ceiling_db, input_gain_db, release_ms; always runs after other tone edits); "
        "nudge (onset_time in seconds into the file, shift_ms where positive moves the note later)."
    ),
    "eager_input_streaming": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "One sentence on what these edits aim to do."},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["gain", "eq", "compressor", "limiter", "nudge"]},
                        "reason": {"type": "string"},
                        "gain_db": {"type": "number", "description": "gain: -24 to 18. eq: -18 to 18."},
                        "eq_type": {
                            "type": "string",
                            "enum": ["peak", "low_shelf", "high_shelf", "highpass", "lowpass"],
                        },
                        "freq_hz": {"type": "number", "description": "20 to 20000"},
                        "q": {"type": "number", "description": "0.1 to 10"},
                        "threshold_db": {"type": "number", "description": "-60 to 0"},
                        "ratio": {"type": "number", "description": "1 to 20"},
                        "attack_ms": {"type": "number", "description": "0.1 to 200"},
                        "release_ms": {"type": "number", "description": "10 to 2000"},
                        "makeup_db": {"type": "number", "description": "0 to 18"},
                        "ceiling_db": {"type": "number", "description": "-12 to -0.1"},
                        "input_gain_db": {"type": "number", "description": "0 to 18"},
                        "onset_time": {"type": "number"},
                        "shift_ms": {"type": "number", "description": "-100 to 100"},
                    },
                    "required": ["type", "reason"],
                },
            },
        },
        "required": ["summary", "edits"],
    },
}


class ChatError(Exception):
    """A chat failure with a message that is safe and useful to show the user."""


@lru_cache
def get_client() -> anthropic.AsyncAnthropic:
    # Created on first use so the server can start, and analysis can work, without credentials.
    return anthropic.AsyncAnthropic()


FILE_TYPE_DESCRIPTIONS: dict[FileType, str] = {
    "track": "single track from a larger song",
    "mix": "full mix",
    "master": "finished master",
}


def build_audio_context(
    features: AudioFeatures,
    region: AudioRegion | None,
    file_type: FileType | None = None,
    song_bpm: float | None = None,
) -> str:
    """Format audio analysis data as a readable context block for the AI."""
    lines = ["## Audio Analysis"]
    lines.append(f"**File type:** {FILE_TYPE_DESCRIPTIONS[file_type] if file_type else 'not specified'}")

    if region:
        lines.append(f"**Selected region:** {region.start:.1f}s – {region.end:.1f}s ({features.duration:.1f}s long)")
    else:
        lines.append(f"**Duration:** {features.duration:.1f}s (full track)")

    if song_bpm is not None:
        lines.append(f"**BPM:** {song_bpm:g} (set by the user)")
    elif features.bpm is not None and features.bpm_confident:
        lines.append(f"**BPM:** {features.bpm}" + (" (whole track)" if region else ""))
    elif features.bpm is not None:
        lines.append("**BPM:** not reliably detected")
    if features.key is not None:
        lines.append(f"**Key:** {features.key if features.key_confident else 'not reliably detected'}")
    if features.loudness_lufs is not None:
        lines.append(f"**Loudness:** {features.loudness_lufs} LUFS integrated")
    if features.peak_dbfs is not None:
        lines.append(f"**Peak:** {features.peak_dbfs} dBFS")
    if features.dynamic_range_db is not None:
        lines.append(f"**Dynamic range:** {features.dynamic_range_db} dB (95th minus 10th percentile short-term level while the part is playing)")
    if features.spectral_centroid_mean is not None:
        lines.append(f"**Spectral centroid:** {features.spectral_centroid_mean:.0f} Hz (brightness indicator)")

    if features.frequency_profile:
        fp = features.frequency_profile
        lines.append(
            f"**Frequency balance:** Low (<250Hz) {fp.low_energy:.0%} / Mid {fp.mid_energy:.0%} / High (>4kHz) {fp.high_energy:.0%}"
        )

    if not features.timing_checked:
        lines.append("\n**Timing:** not checked, because no steady tempo was found to measure against.")
    elif not features.timing_issues:
        lines.append("\n**Timing:** no likely issues found against a steady tempo grid.")
    else:
        lines.append(f"\n**Possible timing issues ({len(features.timing_issues)}):**")
        lines.extend(f"  - {issue.timestamp:.2f}s: {issue.description}" for issue in features.timing_issues)

    return "\n".join(lines)


async def _measure_proposal(
    tool_input: object, content: bytes | None, region: AudioRegion | None
) -> tuple[EditProposal | None, str, bool]:
    """Validate a proposal and measure it on the real audio. Returns (proposal, result for Claude, is_error)."""
    # Streamed tool inputs aren't validated by the API.
    try:
        proposal = EditProposal.model_validate(tool_input)
    except ValidationError as exc:
        logger.info("Invalid edit proposal: %s", exc)
        return None, f"These edits aren't valid, so they weren't shown to the user: {exc}", True

    if content is None:
        return proposal, "Couldn't measure these, because the audio is no longer in memory. Don't quote numbers.", False

    try:
        estimate = await run_in_threadpool(estimate_edits, content, proposal.edits, region)
    except (AudioDecodeError, InvalidRegionError, EditError) as exc:
        return None, f"These edits can't be applied, so they weren't shown to the user: {exc}", True

    scope = f"selected region {region.start:.1f}s-{region.end:.1f}s" if region else "whole file"
    return (
        proposal,
        f"Measured on the {scope}: {json.dumps(dataclasses.asdict(estimate))}. If this misses the goal, call "
        "propose_edits again with adjusted values. Otherwise finish with a short explanation using these numbers.",
        False,
    )


async def stream_chat_response(
    message: str,
    audio_features: AudioFeatures | None,
    region: AudioRegion | None,
    history: list[ChatHistoryItem],
    file_type: FileType | None = None,
    song_bpm: float | None = None,
) -> AsyncGenerator[str | EditProposal, None]:
    """
    Stream a response from Claude, with audio analysis as context.

    Yields text chunks as they arrive, then the final proposed edits if any; raises
    ChatError on failure. When Claude proposes edits, they're measured on the real
    audio and the result goes back to it, so it can revise before the user sees them.
    """
    # An interrupted reply can leave an empty assistant turn, which the API rejects.
    messages = [{"role": item.role, "content": item.content} for item in history if item.content.strip()]

    user_content = message
    if audio_features:
        audio_ctx = build_audio_context(audio_features, region, file_type, song_bpm)
        user_content = f"{audio_ctx}\n\n---\n\n**User question:** {message}"

    messages.append({"role": "user", "content": user_content})

    audio = audio_store.recall(audio_features.session_id) if audio_features else None
    proposal: EditProposal | None = None
    has_text = False

    try:
        for round_number in range(MAX_PROPOSAL_ROUNDS):
            # Editing needs audio to work on. On the last round, Claude has to wrap up.
            tool_options = {"tools": [PROPOSE_EDITS_TOOL]} if audio_features else {}
            if audio_features and round_number == MAX_PROPOSAL_ROUNDS - 1:
                tool_options["tool_choice"] = {"type": "none"}

            round_has_text = False
            async with get_client().beta.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=messages,
                output_config={"effort": "medium"},
                betas=[FALLBACK_BETA],
                fallbacks="default",
                **tool_options,
            ) as stream:
                async for text in stream.text_stream:
                    if has_text and not round_has_text:
                        yield "\n\n"
                    round_has_text = has_text = True
                    yield text
                final = await stream.get_final_message()

            if final.stop_reason == "refusal":
                raise ChatError("Claude declined to answer that. Try rephrasing your question.")

            tool_uses = [b for b in final.content if b.type == "tool_use" and b.name == PROPOSE_EDITS_TOOL["name"]]
            if final.stop_reason != "tool_use" or not tool_uses:
                break

            results = []
            for block in tool_uses:
                measured, result, is_error = await _measure_proposal(block.input, audio, region)
                proposal = measured or proposal
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result, "is_error": is_error}
                )
            messages.append({"role": "assistant", "content": final.content})
            messages.append({"role": "user", "content": results})
    except anthropic.AuthenticationError as exc:
        raise ChatError("Anthropic rejected the API key. Check ANTHROPIC_API_KEY in backend/.env.") from exc
    except anthropic.RateLimitError as exc:
        raise ChatError("Rate limited by the Anthropic API. Wait a moment and try again.") from exc
    except anthropic.APIConnectionError as exc:
        raise ChatError("Couldn't reach the Anthropic API. Check your connection and try again.") from exc
    except anthropic.APIStatusError as exc:
        logger.error("Anthropic API error %s: %s", exc.status_code, exc.message)
        raise ChatError(f"The Anthropic API returned an error ({exc.status_code}). Try again.") from exc
    except ValueError as exc:
        # The SDK raises this from the stream if a tool input isn't parseable JSON.
        logger.warning("Unparseable tool input: %s", exc)
        raise ChatError("Claude's proposed edits got garbled. Try asking again.") from exc
    except TypeError as exc:
        # The SDK raises TypeError when it can't resolve any credentials.
        if "authentication" not in str(exc).lower() and "api_key" not in str(exc).lower():
            raise
        raise ChatError("No Anthropic API key found. Set ANTHROPIC_API_KEY in backend/.env and restart.") from exc

    if proposal:
        yield proposal
