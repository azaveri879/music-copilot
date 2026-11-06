import anthropic
from models.schemas import AudioFeatures, AudioRegion, ChatHistoryItem
from typing import AsyncGenerator

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are Music Copilot, an expert AI production assistant for musicians and producers who use Logic Pro.

Your role is to bridge the gap between creative vision and technical execution — like having a world-class mixing and production engineer looking over the user's shoulder. You have been given structured audio analysis data about the user's track (BPM, key, timing issues, frequency profile, loudness, etc.) and you use it to give specific, grounded, actionable advice.

## Your principles

**Be specific, not generic.** Don't say "try adding some reverb." Say "Open Space Designer, choose the Large Hall preset, set decay time to around 2.8s, mix to 25%, and roll off everything above 8kHz with a low-pass filter to keep it from getting washy."

**Respect the user's creative intent.** When they describe a vibe — "late-night", "lo-fi", "underwater", "warm" — treat it as a real design brief, not a vague request. Translate it into concrete production decisions.

**Reference the audio data.** If you can see from the analysis that there are timing issues, a frequency imbalance, or unusual dynamics, mention them in your response when relevant. Ground your advice in what you actually know about their audio.

**Explain the why briefly.** For each major suggestion, give a one-sentence reason. This helps the user learn and builds trust.

**Use Logic Pro specifics.** Name actual plugins (Space Designer, ChromaVerb, Vintage VU, Channel EQ, Multipressor, Silvertone Bass VI), actual menu locations, and approximate parameter values. Don't be platform-agnostic.

**Be conversational.** You're a production partner, not a manual. Keep it tight and readable.

## Format guidelines

- Use short paragraphs or numbered steps, not walls of text
- When listing plugin settings, use a simple format: `Reverb → Space Designer → Large Hall → Decay: 2.8s, Mix: 25%`
- Keep responses under 300 words unless the complexity warrants more
- If a question is ambiguous, make a reasonable assumption and state it, then answer
"""


def build_audio_context(features: AudioFeatures, region: AudioRegion | None) -> str:
    """Format audio analysis data as a readable context block for the AI."""
    lines = ["## Audio Analysis"]

    if region:
        lines.append(f"**Region:** {region.start:.1f}s – {region.end:.1f}s")
    else:
        lines.append(f"**Duration:** {features.duration:.1f}s (full track)")

    if features.bpm:
        lines.append(f"**BPM:** {features.bpm}")
    if features.key:
        lines.append(f"**Key:** {features.key}")
    if features.loudness_lufs:
        lines.append(f"**Loudness:** {features.loudness_lufs} LUFS")
    if features.dynamic_range_db:
        lines.append(f"**Dynamic range:** {features.dynamic_range_db} dB")
    if features.spectral_centroid_mean:
        lines.append(f"**Spectral centroid:** {features.spectral_centroid_mean:.0f} Hz (brightness indicator)")

    if features.frequency_profile:
        fp = features.frequency_profile
        lines.append(
            f"**Frequency balance:** Low {fp.low_energy:.0%} / Mid {fp.mid_energy:.0%} / High {fp.high_energy:.0%}"
        )

    if features.timing_issues:
        lines.append(f"\n**Timing issues detected ({len(features.timing_issues)}):**")
        for issue in features.timing_issues[:5]:
            lines.append(f"  - {issue.timestamp:.2f}s: {issue.description}")

    return "\n".join(lines)


async def stream_chat_response(
    message: str,
    audio_features: AudioFeatures | None,
    region: AudioRegion | None,
    history: list[ChatHistoryItem],
) -> AsyncGenerator[str, None]:
    """
    Stream a response from Claude, with audio analysis as context.
    Yields text chunks as they arrive.
    """
    messages = []

    # Build conversation history
    for item in history:
        messages.append({"role": item.role, "content": item.content})

    # Build the user message, injecting audio context if available
    user_content = message
    if audio_features:
        audio_ctx = build_audio_context(audio_features, region)
        user_content = f"{audio_ctx}\n\n---\n\n**User question:** {message}"

    messages.append({"role": "user", "content": user_content})

    with client.messages.stream(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text
