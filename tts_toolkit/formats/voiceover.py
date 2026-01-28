"""Voiceover format handler for video narration."""

import os
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import soundfile as sf

from .base import FormatHandler, Segment, AudioOutput
from ..core.chunker import TextChunker

if TYPE_CHECKING:
    from ..backends.base import TTSBackend, VoicePrompt


class VoiceoverHandler(FormatHandler):
    """Handler for video voiceover generation.

    Features:
    - Single narrator voice
    - Automatic text chunking
    - Crossfade stitching
    - Optional SRT timing alignment
    """

    def __init__(
        self,
        backend: Optional["TTSBackend"] = None,
        stitcher=None,
        chunk_min: int = 100,
        chunk_max: int = 300,
        chunk_target: int = 200,
    ):
        """
        Initialize voiceover handler.

        Args:
            backend: TTSBackend instance
            stitcher: AudioStitcher instance
            chunk_min: Minimum characters per chunk
            chunk_max: Maximum characters per chunk
            chunk_target: Target characters per chunk
        """
        super().__init__(backend, stitcher)
        self.chunker = TextChunker(
            min_chars=chunk_min,
            max_chars=chunk_max,
            target_chars=chunk_target,
        )

    def parse(self, input_text: str) -> List[Segment]:
        """
        Parse input text into segments.

        Args:
            input_text: Raw text content

        Returns:
            List of Segment objects
        """
        chunks = self.chunker.chunk(input_text)
        return [
            Segment(text=chunk, pause_after_ms=50)
            for chunk in chunks
        ]

    def generate(
        self,
        segments: List[Segment],
        output_path: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        language: str = "Auto",
        progress_callback=None,
        **kwargs,
    ) -> AudioOutput:
        """
        Generate voiceover audio from segments.

        Args:
            segments: List of Segment objects
            output_path: Optional path to save output
            ref_audio: Reference audio for voice cloning
            ref_text: Reference text transcript
            language: Language code
            progress_callback: Optional callback(current, total, text)
            **kwargs: Additional generation parameters

        Returns:
            AudioOutput with combined audio
        """
        # Load model and create voice prompt
        self.backend.load_model()

        voice_prompt = None
        if ref_audio and ref_text:
            voice_prompt = self.backend.create_voice_prompt(
                reference_audio=ref_audio,
                reference_text=ref_text,
            )

        # Generate audio for each segment
        audio_chunks = []
        sample_rate = None

        for i, segment in enumerate(segments):
            audio, sr = self.backend.generate(
                text=segment.text,
                voice_prompt=voice_prompt,
                language=language,
                **kwargs,
            )
            sample_rate = sr

            # Add pause after segment
            if segment.pause_after_ms > 0:
                audio = self._add_silence(audio, segment.pause_after_ms, sr)

            audio_chunks.append(audio)

            if progress_callback:
                progress_callback(i + 1, len(segments), segment.text)

        # Stitch audio
        self.stitcher.sample_rate = sample_rate
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
            metadata={"language": language},
        )

    def process_with_srt(
        self,
        input_text: str,
        srt_path: str,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> AudioOutput:
        """
        Generate voiceover aligned to SRT timing.

        Args:
            input_text: Raw text content
            srt_path: Path to SRT file for timing
            output_path: Optional path to save output
            **kwargs: Additional generation parameters

        Returns:
            AudioOutput with timing-aligned audio
        """
        # Parse SRT for timing info
        from ..timing.srt_parser import parse_srt

        srt_entries = parse_srt(srt_path)

        # Create segments from SRT entries
        segments = []
        for entry in srt_entries:
            segments.append(
                Segment(
                    text=entry.text,
                    metadata={
                        "start_ms": entry.start_ms,
                        "end_ms": entry.end_ms,
                    },
                )
            )

        return self.generate(segments, output_path, **kwargs)
