"""Batch processing utilities for TTS Toolkit.

Process multiple files in parallel with progress tracking.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import time

logger = logging.getLogger("tts-toolkit")


@dataclass
class BatchJob:
    """A single job in a batch."""

    input_path: str
    output_path: str
    text: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    voice_profile: Optional[str] = None
    language: str = "Auto"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of a batch job."""

    job: BatchJob
    success: bool
    duration_sec: float = 0.0
    error: Optional[str] = None
    output_path: Optional[str] = None
    processing_time_sec: float = 0.0


@dataclass
class BatchSummary:
    """Summary of batch processing."""

    total_jobs: int
    successful: int
    failed: int
    total_duration_sec: float
    total_processing_time_sec: float
    results: List[BatchResult]

    @property
    def success_rate(self) -> float:
        return (self.successful / self.total_jobs * 100) if self.total_jobs > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_jobs": self.total_jobs,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": f"{self.success_rate:.1f}%",
            "total_duration_sec": self.total_duration_sec,
            "total_processing_time_sec": self.total_processing_time_sec,
            "jobs": [
                {
                    "input": r.job.input_path,
                    "output": r.output_path,
                    "success": r.success,
                    "duration_sec": r.duration_sec,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


class BatchProcessor:
    """Process multiple TTS jobs in parallel.

    Example:
        processor = BatchProcessor(
            backend=QwenBackend(device="cuda"),
            workers=2,
        )

        jobs = [
            BatchJob(input_path="script1.txt", output_path="out1.wav", ref_audio="voice.wav"),
            BatchJob(input_path="script2.txt", output_path="out2.wav", ref_audio="voice.wav"),
        ]

        summary = processor.process(jobs, progress_callback=print_progress)
    """

    def __init__(
        self,
        backend=None,
        workers: int = 1,
        timeout: int = 300,
        use_processes: bool = False,
    ):
        """Initialize batch processor.

        Args:
            backend: TTS backend instance (shared across jobs)
            workers: Number of parallel workers
            timeout: Timeout per job in seconds
            use_processes: Use ProcessPoolExecutor instead of ThreadPoolExecutor
        """
        self.backend = backend
        self.workers = workers
        self.timeout = timeout
        self.use_processes = use_processes

    def process(
        self,
        jobs: List[BatchJob],
        progress_callback: Optional[Callable[[int, int, BatchResult], None]] = None,
    ) -> BatchSummary:
        """Process batch of jobs.

        Args:
            jobs: List of BatchJob to process
            progress_callback: Called with (completed, total, result) after each job

        Returns:
            BatchSummary with results
        """
        if not jobs:
            return BatchSummary(
                total_jobs=0,
                successful=0,
                failed=0,
                total_duration_sec=0,
                total_processing_time_sec=0,
                results=[],
            )

        start_time = time.time()
        results = []

        if self.workers == 1:
            # Sequential processing
            for i, job in enumerate(jobs):
                result = self._process_single(job)
                results.append(result)
                if progress_callback:
                    progress_callback(i + 1, len(jobs), result)
        else:
            # Parallel processing
            # Note: For GPU backends, parallel processing may not help
            # due to GPU memory constraints
            ExecutorClass = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

            with ExecutorClass(max_workers=self.workers) as executor:
                futures = {
                    executor.submit(self._process_single, job): job
                    for job in jobs
                }

                completed = 0
                for future in as_completed(futures, timeout=self.timeout * len(jobs)):
                    try:
                        result = future.result(timeout=self.timeout)
                    except Exception as e:
                        job = futures[future]
                        result = BatchResult(
                            job=job,
                            success=False,
                            error=str(e),
                        )

                    results.append(result)
                    completed += 1

                    if progress_callback:
                        progress_callback(completed, len(jobs), result)

        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.success)
        total_duration = sum(r.duration_sec for r in results)

        return BatchSummary(
            total_jobs=len(jobs),
            successful=successful,
            failed=len(jobs) - successful,
            total_duration_sec=total_duration,
            total_processing_time_sec=total_time,
            results=results,
        )

    def _process_single(self, job: BatchJob) -> BatchResult:
        """Process a single job."""
        start_time = time.time()

        try:
            # Get text from file or direct
            if job.text:
                text = job.text
            elif job.input_path:
                with open(job.input_path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                raise ValueError("No text or input_path provided")

            if not text.strip():
                raise ValueError("Empty text")

            # Get voice reference
            ref_audio = job.ref_audio
            ref_text = job.ref_text

            if job.voice_profile and not ref_audio:
                from ..voices.registry import VoiceRegistry
                registry = VoiceRegistry()
                profile = registry.get(job.voice_profile)
                if profile:
                    ref_audio = profile.reference_audio
                    ref_text = profile.reference_text

            if not ref_audio or not ref_text:
                raise ValueError("No reference audio/text or voice profile")

            # Ensure backend is loaded
            if self.backend is None:
                raise ValueError("No backend configured")

            self.backend.load_model()

            # Create voice prompt
            voice_prompt = self.backend.create_voice_prompt(ref_audio, ref_text)

            # Generate audio
            audio, sr = self.backend.generate(
                text=text,
                voice_prompt=voice_prompt,
                language=job.language,
            )

            # Save output
            import soundfile as sf
            output_path = job.output_path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, audio, sr)

            # Calculate duration
            duration_sec = len(audio) / sr

            processing_time = time.time() - start_time

            return BatchResult(
                job=job,
                success=True,
                duration_sec=duration_sec,
                output_path=output_path,
                processing_time_sec=processing_time,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Job failed: {job.input_path} - {e}")

            return BatchResult(
                job=job,
                success=False,
                error=str(e),
                processing_time_sec=processing_time,
            )


def create_jobs_from_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    ref_audio: str,
    ref_text: str,
    pattern: str = "*.txt",
    language: str = "Auto",
) -> List[BatchJob]:
    """Create batch jobs from a directory of text files.

    Args:
        input_dir: Directory containing input text files
        output_dir: Directory for output audio files
        ref_audio: Reference audio file path
        ref_text: Reference audio transcript
        pattern: Glob pattern for input files
        language: Language for generation

    Returns:
        List of BatchJob
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    jobs = []
    for input_file in sorted(input_dir.glob(pattern)):
        output_file = output_dir / f"{input_file.stem}.wav"

        jobs.append(BatchJob(
            input_path=str(input_file),
            output_path=str(output_file),
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
        ))

    return jobs


def create_jobs_from_manifest(
    manifest_path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> List[BatchJob]:
    """Create batch jobs from a JSON manifest file.

    Manifest format:
        {
            "defaults": {
                "ref_audio": "voice.wav",
                "ref_text": "Reference transcript",
                "language": "English"
            },
            "jobs": [
                {"input": "script1.txt", "output": "out1.wav"},
                {"input": "script2.txt", "output": "out2.wav", "language": "Chinese"}
            ]
        }

    Args:
        manifest_path: Path to JSON manifest
        base_dir: Base directory for relative paths

    Returns:
        List of BatchJob
    """
    manifest_path = Path(manifest_path)
    base_dir = Path(base_dir) if base_dir else manifest_path.parent

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    defaults = data.get("defaults", {})
    jobs = []

    for job_data in data.get("jobs", []):
        input_path = job_data.get("input", "")
        output_path = job_data.get("output", "")

        # Resolve relative paths
        if not os.path.isabs(input_path):
            input_path = str(base_dir / input_path)
        if not os.path.isabs(output_path):
            output_path = str(base_dir / output_path)

        ref_audio = job_data.get("ref_audio", defaults.get("ref_audio", ""))
        if ref_audio and not os.path.isabs(ref_audio):
            ref_audio = str(base_dir / ref_audio)

        jobs.append(BatchJob(
            input_path=input_path,
            output_path=output_path,
            text=job_data.get("text"),
            ref_audio=ref_audio,
            ref_text=job_data.get("ref_text", defaults.get("ref_text", "")),
            voice_profile=job_data.get("voice_profile", defaults.get("voice_profile")),
            language=job_data.get("language", defaults.get("language", "Auto")),
            metadata=job_data.get("metadata", {}),
        ))

    return jobs
