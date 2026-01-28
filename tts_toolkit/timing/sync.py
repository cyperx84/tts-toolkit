"""Audio timing synchronization with subtitles."""

import os
from typing import Dict, List, Optional

import numpy as np

from .srt_parser import SRTEntry, write_srt, write_vtt
from .duration_estimator import DurationEstimator


class TimingSync:
    """Synchronize audio with subtitle timing."""

    def __init__(self, sample_rate: int = 24000):
        """
        Initialize timing sync.

        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.estimator = DurationEstimator()

    def generate_subtitles_from_segments(
        self,
        segments: list,
        audio_durations: Optional[List[float]] = None,
    ) -> List[SRTEntry]:
        """
        Generate subtitle entries from segments.

        Args:
            segments: List of Segment objects
            audio_durations: Optional list of actual audio durations per segment

        Returns:
            List of SRTEntry objects with timing
        """
        entries = []
        current_time_ms = 0

        for i, segment in enumerate(segments):
            text = segment.text

            # Add pause before
            if hasattr(segment, 'pause_before_ms'):
                current_time_ms += segment.pause_before_ms

            # Determine duration
            if audio_durations and i < len(audio_durations):
                duration_ms = int(audio_durations[i] * 1000)
            else:
                duration_ms = self.estimator.estimate_duration_ms(text)

            start_ms = current_time_ms
            end_ms = current_time_ms + duration_ms

            entries.append(
                SRTEntry(
                    index=i + 1,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                )
            )

            current_time_ms = end_ms

            # Add pause after
            if hasattr(segment, 'pause_after_ms'):
                current_time_ms += segment.pause_after_ms

        return entries

    def generate_subtitles_from_audio(
        self,
        audio: np.ndarray,
        segments: list,
    ) -> List[SRTEntry]:
        """
        Generate subtitle entries using actual audio durations.

        Note: This requires the audio to be already generated
        with known segment boundaries.

        Args:
            audio: Audio array
            segments: List of Segment objects

        Returns:
            List of SRTEntry objects
        """
        total_duration_sec = len(audio) / self.sample_rate

        # Estimate relative durations
        estimated_durations = []
        for segment in segments:
            estimated_durations.append(
                self.estimator.estimate_duration_sec(segment.text)
            )

        total_estimated = sum(estimated_durations)

        # Scale to actual duration
        scale = total_duration_sec / total_estimated if total_estimated > 0 else 1
        actual_durations = [d * scale for d in estimated_durations]

        return self.generate_subtitles_from_segments(
            segments, audio_durations=actual_durations
        )

    def export_srt(
        self,
        entries: List[SRTEntry],
        output_path: str,
    ) -> str:
        """
        Export subtitles as SRT file.

        Args:
            entries: List of SRTEntry objects
            output_path: Output file path

        Returns:
            Output path
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        return write_srt(entries, output_path)

    def export_vtt(
        self,
        entries: List[SRTEntry],
        output_path: str,
    ) -> str:
        """
        Export subtitles as VTT file.

        Args:
            entries: List of SRTEntry objects
            output_path: Output file path

        Returns:
            Output path
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        return write_vtt(entries, output_path)

    def align_audio_to_timestamps(
        self,
        audio: np.ndarray,
        target_duration_sec: float,
    ) -> np.ndarray:
        """
        Stretch/compress audio to match target duration.

        Simple linear interpolation - for better quality,
        use librosa or rubberband for time-stretching.

        Args:
            audio: Input audio array
            target_duration_sec: Target duration in seconds

        Returns:
            Time-adjusted audio array
        """
        current_duration = len(audio) / self.sample_rate
        if abs(current_duration - target_duration_sec) < 0.01:
            return audio

        target_samples = int(target_duration_sec * self.sample_rate)
        indices = np.linspace(0, len(audio) - 1, target_samples)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def add_silence_padding(
        self,
        audio: np.ndarray,
        target_duration_sec: float,
        position: str = "end",
    ) -> np.ndarray:
        """
        Add silence to match target duration.

        Args:
            audio: Input audio array
            target_duration_sec: Target duration in seconds
            position: Where to add silence ("start", "end", "both")

        Returns:
            Padded audio array
        """
        current_duration = len(audio) / self.sample_rate
        if current_duration >= target_duration_sec:
            return audio

        pad_samples = int((target_duration_sec - current_duration) * self.sample_rate)
        silence = np.zeros(pad_samples, dtype=np.float32)

        if position == "start":
            return np.concatenate([silence, audio])
        elif position == "end":
            return np.concatenate([audio, silence])
        elif position == "both":
            half = pad_samples // 2
            return np.concatenate([
                np.zeros(half, dtype=np.float32),
                audio,
                np.zeros(pad_samples - half, dtype=np.float32),
            ])
        else:
            return np.concatenate([audio, silence])

    def create_timed_audio_track(
        self,
        audio_segments: List[np.ndarray],
        srt_entries: List[SRTEntry],
    ) -> np.ndarray:
        """
        Create audio track with timing aligned to subtitles.

        Args:
            audio_segments: List of audio arrays per segment
            srt_entries: Corresponding subtitle entries

        Returns:
            Combined audio with proper timing
        """
        if not srt_entries:
            return np.concatenate(audio_segments) if audio_segments else np.array([])

        # Calculate total duration needed
        total_duration_ms = max(e.end_ms for e in srt_entries)
        total_samples = int(total_duration_ms * self.sample_rate / 1000)

        result = np.zeros(total_samples, dtype=np.float32)

        for audio, entry in zip(audio_segments, srt_entries):
            start_sample = int(entry.start_ms * self.sample_rate / 1000)
            end_sample = start_sample + len(audio)

            # Ensure we don't exceed bounds
            end_sample = min(end_sample, total_samples)
            audio_to_add = audio[:end_sample - start_sample]

            result[start_sample:end_sample] = audio_to_add

        return result
