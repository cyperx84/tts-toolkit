"""Audiobook format handler for long-form narration with chapters."""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import soundfile as sf

from .base import FormatHandler, Segment, AudioOutput
from ..core.chunker import TextChunker

if TYPE_CHECKING:
    from ..backends.base import TTSBackend, VoicePrompt


@dataclass
class Chapter:
    """A chapter in an audiobook."""

    title: str
    content: str
    number: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class AudiobookHandler(FormatHandler):
    """Handler for audiobook generation with chapter support.

    Features:
    - Chapter detection from Markdown headers
    - Consistent voice across long content
    - Chapter markers in output
    - Per-chapter files + combined output
    """

    def __init__(
        self,
        backend: Optional["TTSBackend"] = None,
        stitcher=None,
        chunk_min: int = 100,
        chunk_max: int = 300,
        chunk_target: int = 200,
        chapter_pause_ms: int = 2000,
        paragraph_pause_ms: int = 500,
    ):
        """
        Initialize audiobook handler.

        Args:
            backend: TTSBackend instance
            stitcher: AudioStitcher instance
            chunk_min: Minimum characters per chunk
            chunk_max: Maximum characters per chunk
            chunk_target: Target characters per chunk
            chapter_pause_ms: Pause between chapters
            paragraph_pause_ms: Pause between paragraphs
        """
        super().__init__(backend, stitcher)
        self.chunker = TextChunker(
            min_chars=chunk_min,
            max_chars=chunk_max,
            target_chars=chunk_target,
        )
        self.chapter_pause_ms = chapter_pause_ms
        self.paragraph_pause_ms = paragraph_pause_ms
        self._voice_prompt: Optional["VoicePrompt"] = None

    def parse_chapters(self, input_text: str, format: str = "markdown") -> List[Chapter]:
        """
        Parse input text into chapters.

        Args:
            input_text: Full text content
            format: Input format ("markdown", "plain", "numbered")

        Returns:
            List of Chapter objects
        """
        if format == "markdown":
            return self._parse_markdown_chapters(input_text)
        elif format == "numbered":
            return self._parse_numbered_chapters(input_text)
        else:
            # Treat as single chapter
            return [
                Chapter(
                    title="Full Text",
                    content=input_text.strip(),
                    number=1,
                )
            ]

    def _parse_markdown_chapters(self, text: str) -> List[Chapter]:
        """Parse chapters from Markdown headers (# or ##)."""
        chapters = []

        # Split on headers
        pattern = r'^(#{1,2})\s+(.+)$'
        lines = text.split('\n')

        current_title = None
        current_content = []
        chapter_num = 0

        for line in lines:
            match = re.match(pattern, line)
            if match:
                # Save previous chapter
                if current_title or current_content:
                    chapter_num += 1
                    chapters.append(
                        Chapter(
                            title=current_title or f"Chapter {chapter_num}",
                            content='\n'.join(current_content).strip(),
                            number=chapter_num,
                        )
                    )
                current_title = match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)

        # Don't forget last chapter
        if current_title or current_content:
            chapter_num += 1
            chapters.append(
                Chapter(
                    title=current_title or f"Chapter {chapter_num}",
                    content='\n'.join(current_content).strip(),
                    number=chapter_num,
                )
            )

        return chapters

    def _parse_numbered_chapters(self, text: str) -> List[Chapter]:
        """Parse chapters like 'Chapter 1: Title' or 'CHAPTER ONE'."""
        chapters = []

        # Pattern for various chapter formats
        pattern = r'^(?:CHAPTER|Chapter)\s+(?:\d+|[A-Z]+)(?::\s*(.+))?$'

        sections = re.split(pattern, text, flags=re.MULTILINE)

        chapter_num = 0
        for i in range(0, len(sections), 2):
            content = sections[i].strip()
            title = sections[i + 1].strip() if i + 1 < len(sections) else None

            if content:
                chapter_num += 1
                chapters.append(
                    Chapter(
                        title=title or f"Chapter {chapter_num}",
                        content=content,
                        number=chapter_num,
                    )
                )

        return chapters

    def parse(self, input_text: str) -> List[Segment]:
        """
        Parse input text into segments (for single-chapter mode).

        Args:
            input_text: Raw text content

        Returns:
            List of Segment objects
        """
        # Split into paragraphs
        paragraphs = re.split(r'\n\n+', input_text.strip())

        segments = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Chunk the paragraph
            chunks = self.chunker.chunk(para)
            for i, chunk in enumerate(chunks):
                pause_after = 50 if i < len(chunks) - 1 else self.paragraph_pause_ms
                segments.append(
                    Segment(
                        text=chunk,
                        pause_after_ms=pause_after,
                    )
                )

        return segments

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
        Generate audiobook audio from segments.

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
        # Load model
        self.backend.load_model()

        # Create voice prompt if reference provided
        voice_prompt = self._voice_prompt
        if ref_audio and ref_text:
            voice_prompt = self.backend.create_voice_prompt(
                reference_audio=ref_audio,
                reference_text=ref_text,
            )
            self._voice_prompt = voice_prompt

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

    def generate_book(
        self,
        input_path: str,
        output_dir: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        format: str = "markdown",
        combine: bool = True,
        language: str = "Auto",
        progress_callback=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate full audiobook with chapter files.

        Args:
            input_path: Path to input text/markdown file
            output_dir: Directory to save chapter files
            ref_audio: Reference audio for voice cloning
            ref_text: Reference text transcript
            format: Input format ("markdown", "plain", "numbered")
            combine: Whether to create combined output file
            language: Language code
            progress_callback: Optional callback(chapter, total_chapters, chapter_title)
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with chapter outputs and metadata
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load and parse chapters
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        chapters = self.parse_chapters(text, format=format)

        if not chapters:
            raise ValueError("No chapters found in input")

        # Load model and create voice prompt once
        self.backend.load_model()
        if ref_audio and ref_text:
            self._voice_prompt = self.backend.create_voice_prompt(
                reference_audio=ref_audio,
                reference_text=ref_text,
            )

        # Generate each chapter
        chapter_outputs = []
        sample_rate = None

        for i, chapter in enumerate(chapters):
            if progress_callback:
                progress_callback(i + 1, len(chapters), chapter.title)

            # Parse chapter content
            segments = self.parse(chapter.content)

            if not segments:
                continue

            # Generate chapter audio
            chapter_path = os.path.join(
                output_dir,
                f"chapter_{chapter.number:02d}.wav"
            )

            output = self.generate(
                segments=segments,
                output_path=chapter_path,
                language=language,
                **kwargs,
            )

            sample_rate = output.sample_rate

            chapter_outputs.append({
                "chapter": chapter.number,
                "title": chapter.title,
                "path": chapter_path,
                "duration_sec": output.duration_sec,
                "segments": len(segments),
            })

        # Create combined output
        combined_path = None
        if combine and chapter_outputs:
            combined_path = os.path.join(output_dir, "full_audiobook.wav")

            all_audio = []
            for ch_output in chapter_outputs:
                audio, sr = sf.read(ch_output["path"], dtype="float32")
                all_audio.append(audio)

                # Add chapter pause
                pause = np.zeros(
                    int(self.chapter_pause_ms * sr / 1000),
                    dtype=np.float32,
                )
                all_audio.append(pause)

            combined = np.concatenate(all_audio)

            # Normalize
            max_val = np.abs(combined).max()
            if max_val > 0.99:
                combined = combined / max_val * 0.99

            sf.write(combined_path, combined, sample_rate)

        total_duration = sum(ch["duration_sec"] for ch in chapter_outputs)

        return {
            "chapters": chapter_outputs,
            "total_chapters": len(chapter_outputs),
            "total_duration_sec": total_duration,
            "combined_path": combined_path,
            "output_dir": output_dir,
        }
