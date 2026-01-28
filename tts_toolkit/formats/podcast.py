"""Podcast format handler for multi-speaker podcast generation."""

import os
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import soundfile as sf

from .base import FormatHandler, Segment, AudioOutput
from .dialogue import DialogueHandler

if TYPE_CHECKING:
    from ..backends.base import TTSBackend


class PodcastHandler(FormatHandler):
    """Handler for podcast generation with multiple speakers.

    Features:
    - Multiple speaker voices (host, guest, etc.)
    - Intro/outro music mixing
    - Natural turn-taking with configurable pauses
    - Segment markers
    """

    def __init__(
        self,
        backend: Optional["TTSBackend"] = None,
        stitcher=None,
        pause_between_speakers_ms: int = 400,
        intro_music: Optional[str] = None,
        outro_music: Optional[str] = None,
        background_music: Optional[str] = None,
        background_volume_db: float = -18.0,
    ):
        """
        Initialize podcast handler.

        Args:
            backend: TTSBackend instance
            stitcher: AudioStitcher instance
            pause_between_speakers_ms: Pause when speaker changes
            intro_music: Path to intro music file
            outro_music: Path to outro music file
            background_music: Path to background music file
            background_volume_db: Volume of background music in dB
        """
        super().__init__(backend, stitcher)
        self.pause_between_speakers_ms = pause_between_speakers_ms
        self.intro_music = intro_music
        self.outro_music = outro_music
        self.background_music = background_music
        self.background_volume_db = background_volume_db

        # Use DialogueHandler for parsing and generation
        self._dialogue_handler = DialogueHandler(
            backend=backend,
            stitcher=stitcher,
            pause_between_speakers_ms=pause_between_speakers_ms,
        )

    def set_backend(self, backend: "TTSBackend") -> None:
        """Set the TTS backend."""
        super().set_backend(backend)
        self._dialogue_handler.set_backend(backend)

    def parse(self, input_text: str) -> List[Segment]:
        """
        Parse podcast script into segments.

        Supports additional markup:
        - [HOST]: text
        - [GUEST]: text
        - [INTRO] / [OUTRO] markers
        - [SEGMENT: title] markers

        Args:
            input_text: Podcast script with speaker markup

        Returns:
            List of Segment objects
        """
        segments = []
        lines = input_text.strip().split('\n')

        prev_speaker = None
        in_segment = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for segment markers
            if line.startswith('[INTRO]'):
                segments.append(
                    Segment(
                        text="",
                        metadata={"type": "intro"},
                    )
                )
                continue
            elif line.startswith('[OUTRO]'):
                segments.append(
                    Segment(
                        text="",
                        metadata={"type": "outro"},
                    )
                )
                continue
            elif line.startswith('[SEGMENT:'):
                title = line[9:].rstrip(']').strip()
                in_segment = title
                continue

            # Parse speaker lines
            parsed = self._parse_speaker_line(line)
            if parsed:
                speaker_id, text = parsed

                # Determine pause
                if prev_speaker is None:
                    pause_before = 0
                elif speaker_id != prev_speaker:
                    pause_before = self.pause_between_speakers_ms
                else:
                    pause_before = 150

                segment = Segment(
                    text=text,
                    speaker_id=speaker_id,
                    pause_before_ms=pause_before,
                )
                if in_segment:
                    segment.metadata["segment"] = in_segment

                segments.append(segment)
                prev_speaker = speaker_id

        return segments

    def _parse_speaker_line(self, line: str) -> Optional[Tuple[str, str]]:
        """Parse a single line for speaker and text."""
        # [SPEAKER]: text
        match = re.match(r'\[([A-Z][A-Za-z0-9_]+)\]:\s*(.+)', line)
        if match:
            return match.group(1), match.group(2)

        # SPEAKER: text
        match = re.match(r'^([A-Z][A-Z0-9_]+):\s*(.+)$', line)
        if match:
            return match.group(1), match.group(2)

        return None

    def generate(
        self,
        segments: List[Segment],
        output_path: Optional[str] = None,
        speaker_refs: Optional[Dict[str, Tuple[str, str]]] = None,
        language: str = "Auto",
        add_music: bool = True,
        progress_callback=None,
        **kwargs,
    ) -> AudioOutput:
        """
        Generate podcast audio from segments.

        Args:
            segments: List of Segment objects
            output_path: Optional path to save output
            speaker_refs: Dict mapping speaker_id to (ref_audio, ref_text)
            language: Language code
            add_music: Whether to add intro/outro/background music
            progress_callback: Optional callback(current, total, text)
            **kwargs: Additional generation parameters

        Returns:
            AudioOutput with combined audio
        """
        speaker_refs = speaker_refs or {}

        # Separate content and markers
        intro_marker = None
        outro_marker = None
        content_segments = []

        for segment in segments:
            if segment.metadata.get("type") == "intro":
                intro_marker = segment
            elif segment.metadata.get("type") == "outro":
                outro_marker = segment
            else:
                content_segments.append(segment)

        # Ensure dialogue handler uses same backend
        self._dialogue_handler._backend = self._backend

        # Generate content using dialogue handler
        content_output = self._dialogue_handler.generate(
            segments=content_segments,
            speaker_refs=speaker_refs,
            language=language,
            progress_callback=progress_callback,
            **kwargs,
        )

        combined = content_output.audio
        sample_rate = content_output.sample_rate

        # Add music if requested
        if add_music:
            from ..core.mixer import AudioMixer

            mixer = AudioMixer(sample_rate=sample_rate)

            # Add background music
            if self.background_music and os.path.exists(self.background_music):
                bg = mixer.load_audio(self.background_music)
                combined = mixer.mix_background(
                    foreground=combined,
                    background=bg,
                    background_db=self.background_volume_db,
                )

            # Add intro
            if self.intro_music and os.path.exists(self.intro_music):
                intro = mixer.load_audio(self.intro_music)
                intro = mixer.fade_out(intro, 1000)
                combined = mixer.add_intro_outro(
                    main_audio=combined,
                    intro=intro,
                    crossfade_ms=1000,
                )

            # Add outro
            if self.outro_music and os.path.exists(self.outro_music):
                outro = mixer.load_audio(self.outro_music)
                outro = mixer.fade_in(outro, 1000)
                combined = mixer.add_intro_outro(
                    main_audio=combined,
                    outro=outro,
                    crossfade_ms=1500,
                )

        # Normalize
        max_val = np.abs(combined).max()
        if max_val > 0.99:
            combined = combined / max_val * 0.99

        # Save if requested
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            sf.write(output_path, combined, sample_rate)

        duration = len(combined) / sample_rate

        # Get speakers
        speakers = self._dialogue_handler.detect_speakers(content_segments)

        return AudioOutput(
            audio=combined,
            sample_rate=sample_rate,
            segments=segments,
            duration_sec=duration,
            metadata={
                "speakers": speakers,
                "language": language,
                "has_intro": intro_marker is not None,
                "has_outro": outro_marker is not None,
            },
        )

    def generate_episode(
        self,
        script_path: str,
        output_path: str,
        host_ref: Optional[Tuple[str, str]] = None,
        guest_ref: Optional[Tuple[str, str]] = None,
        **kwargs,
    ) -> AudioOutput:
        """
        Convenience method to generate a full podcast episode.

        Args:
            script_path: Path to podcast script file
            output_path: Path to save output audio
            host_ref: Tuple of (ref_audio, ref_text) for host voice
            guest_ref: Tuple of (ref_audio, ref_text) for guest voice
            **kwargs: Additional generation parameters

        Returns:
            AudioOutput with episode audio
        """
        # Load script
        with open(script_path, "r", encoding="utf-8") as f:
            script = f.read()

        # Parse segments
        segments = self.parse(script)

        # Build speaker refs
        speaker_refs = {}
        if host_ref:
            speaker_refs["HOST"] = host_ref
        if guest_ref:
            speaker_refs["GUEST"] = guest_ref

        return self.generate(
            segments=segments,
            output_path=output_path,
            speaker_refs=speaker_refs,
            **kwargs,
        )
