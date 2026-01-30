"""Main TTS pipeline with backend abstraction and checkpointing."""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from .chunker import TextChunker
from .stitcher import AudioStitcher
from ..utils.memory import (
    warn_if_low_memory,
    clear_gpu_cache,
    log_memory_stats,
    estimate_text_memory,
)

if TYPE_CHECKING:
    from ..backends.base import TTSBackend, VoicePrompt

logger = logging.getLogger("tts-toolkit")


@dataclass
class ChunkStatus:
    """Status of a single chunk."""

    index: int
    text: str
    status: str = "pending"  # pending, completed, failed
    output_path: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    duration_sec: Optional[float] = None


@dataclass
class PipelineCheckpoint:
    """Checkpoint data for resumable processing."""

    chunks: List[Dict]
    sample_rate: Optional[int] = None
    completed_count: int = 0
    failed_count: int = 0
    started_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "PipelineCheckpoint":
        return cls(**data)


class Pipeline:
    """Main TTS pipeline with backend abstraction.

    This is the primary interface for generating long-form TTS.
    It handles text chunking, generation with retry, and audio stitching.

    Args:
        backend: TTS backend instance (QwenBackend, MockBackend, etc.)
        chunk_min: Minimum characters per chunk
        chunk_max: Maximum characters per chunk
        chunk_target: Target characters per chunk
        crossfade_ms: Crossfade duration in milliseconds
        max_retries: Maximum retry attempts per chunk

    Example:
        from tts_toolkit import Pipeline
        from tts_toolkit.backends import QwenBackend

        backend = QwenBackend(device="cuda:0")
        pipeline = Pipeline(backend=backend)

        pipeline.process(
            text="Your long text...",
            ref_audio="voice.wav",
            ref_text="Reference transcript",
            output_path="output.wav",
        )
    """

    def __init__(
        self,
        backend: Optional["TTSBackend"] = None,
        chunk_min: int = 100,
        chunk_max: int = 300,
        chunk_target: int = 200,
        crossfade_ms: int = 75,
        max_retries: int = 3,
    ):
        """Initialize pipeline."""
        self._backend = backend
        self.chunker = TextChunker(
            min_chars=chunk_min,
            max_chars=chunk_max,
            target_chars=chunk_target,
        )
        self.stitcher = AudioStitcher(crossfade_ms=crossfade_ms)
        self.max_retries = max_retries
        self._voice_prompt: Optional["VoicePrompt"] = None

    @property
    def backend(self) -> "TTSBackend":
        """Get the TTS backend, creating default if needed."""
        if self._backend is None:
            # Try to create QwenBackend, fall back to MockBackend
            try:
                from ..backends import QwenBackend
                self._backend = QwenBackend()
                logger.info("Using QwenBackend (default)")
            except ImportError as e:
                from ..backends import MockBackend
                logger.warning(
                    f"qwen-tts not available ({e}), falling back to MockBackend. "
                    "Install with: pip install tts-toolkit[qwen]"
                )
                self._backend = MockBackend()
        return self._backend

    def set_backend(self, backend: "TTSBackend") -> None:
        """Set the TTS backend."""
        self._backend = backend
        self._voice_prompt = None

    def _checkpoint_path(self, work_dir: str) -> str:
        return os.path.join(work_dir, "checkpoint.json")

    def _load_checkpoint(self, work_dir: str) -> Optional[PipelineCheckpoint]:
        """Load checkpoint if it exists."""
        path = self._checkpoint_path(work_dir)
        if os.path.exists(path):
            with open(path, "r") as f:
                return PipelineCheckpoint.from_dict(json.load(f))
        return None

    def _save_checkpoint(self, checkpoint: PipelineCheckpoint, work_dir: str) -> None:
        """Save checkpoint to disk."""
        checkpoint.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        path = self._checkpoint_path(work_dir)
        with open(path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

    def _init_checkpoint(self, chunks: List[str], work_dir: str) -> PipelineCheckpoint:
        """Initialize a new checkpoint."""
        chunk_statuses = [
            asdict(ChunkStatus(index=i, text=chunk))
            for i, chunk in enumerate(chunks)
        ]
        checkpoint = PipelineCheckpoint(
            chunks=chunk_statuses,
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._save_checkpoint(checkpoint, work_dir)
        return checkpoint

    def process(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        output_path: str,
        work_dir: Optional[str] = None,
        language: str = "Auto",
        resume: bool = True,
        progress_callback: Optional[Callable] = None,
        **gen_kwargs,
    ) -> str:
        """Process long text into a single audio file.

        Args:
            text: Text to synthesize
            ref_audio: Path to reference audio for voice cloning
            ref_text: Transcript of reference audio
            output_path: Path for final output WAV file
            work_dir: Working directory for chunks and checkpoint
            language: Language code
            resume: Whether to resume from checkpoint
            progress_callback: Optional callback(current, total, chunk_text)
            **gen_kwargs: Additional generation parameters

        Returns:
            Path to final output file
        """
        # Setup working directory
        if work_dir is None:
            base = os.path.splitext(output_path)[0]
            work_dir = f"{base}_chunks"
        os.makedirs(work_dir, exist_ok=True)

        # Check memory before starting
        device = getattr(self._backend, 'device', 'cpu') if self._backend else 'cpu'
        backend_name = self._backend.__class__.__name__.lower().replace('backend', '') if self._backend else 'unknown'
        warn_if_low_memory(backend_name, len(text), device)

        # Log initial memory state
        logger.debug("Initial memory state:")
        log_memory_stats(device)

        # Chunk the text
        chunks = self.chunker.chunk(text)
        if not chunks:
            raise ValueError("No text to process after chunking")

        logger.info(f"Split text into {len(chunks)} chunks")

        # Load or initialize checkpoint
        checkpoint = None
        if resume:
            checkpoint = self._load_checkpoint(work_dir)
            if checkpoint:
                if len(checkpoint.chunks) != len(chunks):
                    logger.info("Chunk count mismatch, starting fresh")
                    checkpoint = None
                else:
                    for i, (cp_chunk, new_chunk) in enumerate(
                        zip(checkpoint.chunks, chunks)
                    ):
                        if cp_chunk["text"] != new_chunk:
                            logger.info(f"Chunk {i} text mismatch, starting fresh")
                            checkpoint = None
                            break

        if checkpoint is None:
            checkpoint = self._init_checkpoint(chunks, work_dir)
            logger.info("Created new checkpoint")
        else:
            logger.info(
                f"Resuming from checkpoint: {checkpoint.completed_count}/{len(chunks)} completed"
            )

        # Load backend and create voice prompt
        self.backend.load_model()
        voice_prompt = self.backend.create_voice_prompt(
            reference_audio=ref_audio,
            reference_text=ref_text,
        )

        # Process each chunk
        import soundfile as sf

        for i, chunk_data in enumerate(checkpoint.chunks):
            chunk_status = ChunkStatus(**chunk_data)

            if chunk_status.status == "completed" and chunk_status.output_path:
                if os.path.exists(chunk_status.output_path):
                    logger.debug(f"Chunk {i + 1}/{len(chunks)}: skipping (already completed)")
                    if progress_callback:
                        progress_callback(i + 1, len(chunks), chunk_status.text)
                    continue

            chunk_status.status = "pending"
            output_chunk_path = os.path.join(work_dir, f"chunk_{i:04d}.wav")

            success = False
            for attempt in range(self.max_retries):
                try:
                    chunk_status.attempts = attempt + 1
                    logger.info(
                        f"Chunk {i + 1}/{len(chunks)} (attempt {attempt + 1}): generating..."
                    )

                    t0 = time.time()
                    audio, sr = self.backend.generate(
                        text=chunk_status.text,
                        voice_prompt=voice_prompt,
                        language=language,
                        **gen_kwargs,
                    )

                    # Save chunk
                    sf.write(output_chunk_path, audio, sr)
                    t1 = time.time()

                    chunk_status.status = "completed"
                    chunk_status.output_path = output_chunk_path
                    chunk_status.duration_sec = t1 - t0
                    chunk_status.error = None
                    checkpoint.completed_count += 1
                    checkpoint.sample_rate = sr

                    logger.info(f"Chunk {i + 1}/{len(chunks)}: completed in {t1 - t0:.1f}s")
                    success = True
                    break

                except Exception as e:
                    chunk_status.error = str(e)
                    wait_time = 2 ** attempt
                    logger.warning(f"Chunk {i + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)

            if not success:
                chunk_status.status = "failed"
                checkpoint.failed_count += 1
                logger.error(f"Chunk {i + 1} failed after {self.max_retries} attempts")

            checkpoint.chunks[i] = asdict(chunk_status)
            self._save_checkpoint(checkpoint, work_dir)

            if progress_callback:
                progress_callback(i + 1, len(chunks), chunk_status.text)

        # Check for failures
        if checkpoint.failed_count > 0:
            raise RuntimeError(
                f"{checkpoint.failed_count} chunks failed. "
                f"Fix issues and run with resume=True to continue."
            )

        # Collect chunk files
        chunk_paths = []
        for chunk_data in checkpoint.chunks:
            path = chunk_data.get("output_path")
            if path and os.path.exists(path):
                chunk_paths.append(path)

        if len(chunk_paths) != len(chunks):
            raise RuntimeError(
                f"Expected {len(chunks)} chunk files, found {len(chunk_paths)}"
            )

        # Stitch together
        logger.info(f"Stitching {len(chunk_paths)} chunks...")
        self.stitcher.sample_rate = checkpoint.sample_rate
        self.stitcher.stitch_files(chunk_paths, output_path)

        # Clean up GPU memory after processing
        if hasattr(self._backend, 'cleanup_gpu_memory'):
            self._backend.cleanup_gpu_memory()
        clear_gpu_cache()

        logger.info(f"Output saved to: {output_path}")
        return output_path

    def process_file(
        self,
        input_path: str,
        ref_audio: str,
        ref_text: str,
        output_path: str,
        encoding: str = "utf-8",
        **kwargs,
    ) -> str:
        """Process a text file into audio."""
        with open(input_path, "r", encoding=encoding) as f:
            text = f.read()
        return self.process(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            output_path=output_path,
            **kwargs,
        )

    def generate_quick(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        output_path: Optional[str] = None,
        language: str = "Auto",
        **gen_kwargs,
    ) -> np.ndarray:
        """Quick generation for short text (no chunking).

        Args:
            text: Short text to synthesize
            ref_audio: Reference audio path
            ref_text: Reference transcript
            output_path: Optional path to save
            language: Language code
            **gen_kwargs: Generation parameters

        Returns:
            Audio array
        """
        self.backend.load_model()
        voice_prompt = self.backend.create_voice_prompt(
            reference_audio=ref_audio,
            reference_text=ref_text,
        )

        audio, sr = self.backend.generate(
            text=text,
            voice_prompt=voice_prompt,
            language=language,
            **gen_kwargs,
        )

        if output_path:
            import soundfile as sf
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            sf.write(output_path, audio, sr)

        return audio

    def process_parallel(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        output_path: str,
        work_dir: Optional[str] = None,
        language: str = "Auto",
        workers: int = 2,
        progress_callback: Optional[Callable] = None,
        **gen_kwargs,
    ) -> str:
        """Process text with parallel chunk generation.

        Note: Parallel processing may not be faster for GPU backends
        due to memory constraints. Best for CPU backends or API-based backends.

        Args:
            text: Text to synthesize
            ref_audio: Path to reference audio for voice cloning
            ref_text: Transcript of reference audio
            output_path: Path for final output WAV file
            work_dir: Working directory for chunks
            language: Language code
            workers: Number of parallel workers (default: 2)
            progress_callback: Optional callback(current, total, chunk_text)
            **gen_kwargs: Additional generation parameters

        Returns:
            Path to final output file
        """
        import soundfile as sf

        # Setup working directory
        if work_dir is None:
            base = os.path.splitext(output_path)[0]
            work_dir = f"{base}_chunks"
        os.makedirs(work_dir, exist_ok=True)

        # Chunk the text
        chunks = self.chunker.chunk(text)
        if not chunks:
            raise ValueError("No text to process after chunking")

        logger.info(f"Split text into {len(chunks)} chunks, processing with {workers} workers")

        # Load backend and create voice prompt once
        self.backend.load_model()
        voice_prompt = self.backend.create_voice_prompt(
            reference_audio=ref_audio,
            reference_text=ref_text,
        )

        # Process chunks in parallel
        results = {}
        completed = 0
        sample_rate = None

        def process_chunk(index: int, chunk_text: str) -> Dict[str, Any]:
            """Process a single chunk."""
            output_chunk_path = os.path.join(work_dir, f"chunk_{index:04d}.wav")

            try:
                t0 = time.time()
                audio, sr = self.backend.generate(
                    text=chunk_text,
                    voice_prompt=voice_prompt,
                    language=language,
                    **gen_kwargs,
                )
                sf.write(output_chunk_path, audio, sr)
                t1 = time.time()

                return {
                    "index": index,
                    "success": True,
                    "path": output_chunk_path,
                    "sample_rate": sr,
                    "duration": t1 - t0,
                }
            except Exception as e:
                logger.error(f"Chunk {index} failed: {e}")
                return {
                    "index": index,
                    "success": False,
                    "error": str(e),
                }

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_chunk, i, chunk): i
                for i, chunk in enumerate(chunks)
            }

            for future in as_completed(futures):
                result = future.result()
                results[result["index"]] = result

                if result["success"]:
                    sample_rate = result.get("sample_rate", sample_rate)
                    completed += 1
                    logger.info(
                        f"Chunk {result['index'] + 1}/{len(chunks)} completed "
                        f"in {result.get('duration', 0):.1f}s"
                    )
                else:
                    logger.error(
                        f"Chunk {result['index'] + 1}/{len(chunks)} failed: "
                        f"{result.get('error', 'Unknown error')}"
                    )

                if progress_callback:
                    progress_callback(
                        len(results), len(chunks), chunks[result["index"]]
                    )

        # Check for failures
        failed = [i for i, r in results.items() if not r["success"]]
        if failed:
            raise RuntimeError(
                f"{len(failed)} chunks failed: {failed}. "
                "Consider reducing workers or using sequential processing."
            )

        # Collect chunk files in order
        chunk_paths = [results[i]["path"] for i in range(len(chunks))]

        # Stitch together
        logger.info(f"Stitching {len(chunk_paths)} chunks...")
        self.stitcher.sample_rate = sample_rate
        self.stitcher.stitch_files(chunk_paths, output_path)

        logger.info(f"Output saved to: {output_path}")
        return output_path
