"""Dialogue format handler for two-person conversations."""

import os
import re
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import soundfile as sf

from .base import FormatHandler, Segment, AudioOutput

if TYPE_CHECKING:
    from ..backends.base import TTSBackend, VoicePrompt


class DialogueHandler(FormatHandler):
    """Handler for two-person dialogue generation.

    Supports markup formats:
    - [S1]: text / [S2]: text (Dia-style)
    - [SPEAKER1]: text / [SPEAKER2]: text
    - **Speaker1**: text / **Speaker2**: text (Markdown)
    - SPEAKER1: text / SPEAKER2: text (screenplay)
    """

    # Regex patterns for different dialogue formats
    PATTERNS = [
        # [S1]: text or [S2]: text (Dia-style)
        (r'\[S(\d)\]:\s*(.+?)(?=\[S\d\]:|$)', 'numbered'),
        # [SPEAKER_NAME]: text
        (r'\[([A-Z][A-Za-z0-9_]+)\]:\s*(.+?)(?=\[[A-Z]|$)', 'named_bracket'),
        # **Speaker**: text (Markdown bold)
        (r'\*\*([^*]+)\*\*:\s*(.+?)(?=\*\*[^*]+\*\*:|$)', 'markdown'),
        # SPEAKER: text (uppercase, screenplay style)
        (r'^([A-Z][A-Z0-9_]+):\s*(.+?)$', 'screenplay'),
    ]

    def __init__(
        self,
        backend: Optional["TTSBackend"] = None,
        stitcher=None,
        pause_between_speakers_ms: int = 300,
        pause_between_lines_ms: int = 150,
    ):
        """
        Initialize dialogue handler.

        Args:
            backend: TTSBackend instance
            stitcher: AudioStitcher instance
            pause_between_speakers_ms: Pause when speaker changes
            pause_between_lines_ms: Pause between lines from same speaker
        """
        super().__init__(backend, stitcher)
        self.pause_between_speakers_ms = pause_between_speakers_ms
        self.pause_between_lines_ms = pause_between_lines_ms

    def parse(self, input_text: str) -> List[Segment]:
        """
        Parse dialogue markup into segments.

        Args:
            input_text: Dialogue text with speaker markup

        Returns:
            List of Segment objects
        """
        segments = []
        prev_speaker = None

        # Try each pattern
        for pattern, format_type in self.PATTERNS:
            if format_type == 'screenplay':
                # Line-by-line for screenplay format
                matches = []
                for line in input_text.strip().split('\n'):
                    match = re.match(pattern, line.strip())
                    if match:
                        matches.append(match)
            else:
                # Multi-line for other formats
                matches = list(re.finditer(pattern, input_text, re.MULTILINE | re.DOTALL))

            if matches:
                for match in matches:
                    speaker_id = match.group(1).strip()
                    text = match.group(2).strip()

                    if not text:
                        continue

                    # Determine pause based on speaker change
                    if prev_speaker is None:
                        pause_before = 0
                    elif speaker_id != prev_speaker:
                        pause_before = self.pause_between_speakers_ms
                    else:
                        pause_before = self.pause_between_lines_ms

                    segments.append(
                        Segment(
                            text=text,
                            speaker_id=speaker_id,
                            pause_before_ms=pause_before,
                        )
                    )
                    prev_speaker = speaker_id

                break  # Use first matching pattern

        # If no pattern matched, treat as single speaker
        if not segments:
            segments = [Segment(text=input_text.strip())]

        return segments

    def detect_speakers(self, segments: List[Segment]) -> List[str]:
        """
        Get list of unique speakers in order of appearance.

        Args:
            segments: List of parsed segments

        Returns:
            List of speaker IDs
        """
        seen = set()
        speakers = []
        for segment in segments:
            if segment.speaker_id and segment.speaker_id not in seen:
                seen.add(segment.speaker_id)
                speakers.append(segment.speaker_id)
        return speakers

    def generate(
        self,
        segments: List[Segment],
        output_path: Optional[str] = None,
        speaker_refs: Optional[Dict[str, Tuple[str, str]]] = None,
        language: str = "Auto",
        progress_callback=None,
        **kwargs,
    ) -> AudioOutput:
        """
        Generate dialogue audio from segments.

        Args:
            segments: List of Segment objects
            output_path: Optional path to save output
            speaker_refs: Dict mapping speaker_id to (ref_audio, ref_text)
            language: Language code
            progress_callback: Optional callback(current, total, text)
            **kwargs: Additional generation parameters

        Returns:
            AudioOutput with combined audio
        """
        speaker_refs = speaker_refs or {}

        # Load model
        self.backend.load_model()

        # Create voice prompts for each speaker
        speakers = self.detect_speakers(segments)
        voice_prompts: Dict[str, "VoicePrompt"] = {}

        for speaker_id in speakers:
            if speaker_id in speaker_refs:
                ref_audio, ref_text = speaker_refs[speaker_id]
                voice_prompts[speaker_id] = self.backend.create_voice_prompt(
                    reference_audio=ref_audio,
                    reference_text=ref_text,
                )

        # Generate audio for each segment
        audio_chunks = []
        sample_rate = None

        for i, segment in enumerate(segments):
            # Add pause before segment
            if segment.pause_before_ms > 0 and sample_rate:
                silence = np.zeros(
                    int(segment.pause_before_ms * sample_rate / 1000),
                    dtype=np.float32,
                )
                audio_chunks.append(silence)

            # Get voice prompt for this speaker
            voice_prompt = voice_prompts.get(segment.speaker_id)

            # Generate audio
            audio, sr = self.backend.generate(
                text=segment.text,
                voice_prompt=voice_prompt,
                language=language,
                **kwargs,
            )
            sample_rate = sr
            audio_chunks.append(audio)

            if progress_callback:
                progress_callback(i + 1, len(segments), segment.text)

        # Stitch audio (no crossfade for dialogue)
        self.stitcher.sample_rate = sample_rate
        self.stitcher.crossfade_ms = 0  # No crossfade between speakers
        combined = self.stitcher.stitch(audio_chunks)

        # Normalize
        max_val = np.abs(combined).max()
        if max_val > 0.99:
            combined = combined / max_val * 0.99

        # Save if requested
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            sf.write(output_path, combined, sample_rate)

        duration = len(combined) / sample_rate

        return AudioOutput(
            audio=combined,
            sample_rate=sample_rate,
            segments=segments,
            duration_sec=duration,
            metadata={
                "speakers": speakers,
                "language": language,
            },
        )

    def generate_with_emotions(
        self,
        segments: List[Segment],
        output_path: Optional[str] = None,
        emotion_map: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> AudioOutput:
        """
        Generate dialogue with per-line emotions.

        Args:
            segments: List of Segment objects (with emotion field set)
            output_path: Optional path to save output
            emotion_map: Optional default emotions per speaker
            **kwargs: Additional generation parameters

        Returns:
            AudioOutput with combined audio
        """
        from ..voices.emotions import apply_emotion

        emotion_map = emotion_map or {}

        # Apply emotions to generation kwargs per segment
        for segment in segments:
            if segment.emotion:
                segment.metadata["gen_kwargs"] = apply_emotion(
                    kwargs.copy(), segment.emotion
                )
            elif segment.speaker_id in emotion_map:
                segment.metadata["gen_kwargs"] = apply_emotion(
                    kwargs.copy(), emotion_map[segment.speaker_id]
                )

        return self.generate(segments, output_path, **kwargs)
