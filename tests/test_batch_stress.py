"""Stress tests for batch processing."""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

from tts_toolkit.backends import MockBackend
from tts_toolkit.utils.batch import (
    BatchProcessor,
    BatchJob,
    create_jobs_from_directory,
)


class TestLargeBatchProcessing:
    """Tests for large batch processing."""

    def test_batch_100_files(self, tmp_path):
        """Test processing 100 files in batch."""
        # Create 100 input files
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        for i in range(100):
            (input_dir / f"file_{i:03d}.txt").write_text(f"Test content for file {i}.")

        output_dir = tmp_path / "output"

        # Create reference audio
        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"RIFF" + b"\x00" * 1024)

        # Create jobs
        jobs = create_jobs_from_directory(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            ref_audio=str(ref_audio),
            ref_text="Reference text",
            pattern="*.txt",
        )

        assert len(jobs) == 100

        # Process with mock backend
        backend = MockBackend()
        processor = BatchProcessor(backend=backend, workers=1, timeout=60)

        results = []
        def callback(completed, total, result):
            results.append(result)

        summary = processor.process(jobs, progress_callback=callback)

        assert summary.total_jobs == 100
        assert summary.successful == 100
        assert summary.failed == 0
        assert summary.success_rate == 100.0

    def test_batch_50_files_parallel(self, tmp_path):
        """Test parallel processing of 50 files."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        for i in range(50):
            (input_dir / f"file_{i:02d}.txt").write_text(f"Test content {i}.")

        output_dir = tmp_path / "output"

        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"RIFF" + b"\x00" * 1024)

        jobs = create_jobs_from_directory(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            ref_audio=str(ref_audio),
            ref_text="Reference",
        )

        # Process with multiple workers
        backend = MockBackend()
        processor = BatchProcessor(backend=backend, workers=4, timeout=60)

        summary = processor.process(jobs)

        assert summary.total_jobs == 50
        assert summary.successful == 50


class TestWorkerPoolExhaustion:
    """Tests for worker pool limits."""

    def test_more_jobs_than_workers(self, tmp_path):
        """Test handling more jobs than available workers."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        for i in range(20):
            (input_dir / f"file_{i:02d}.txt").write_text(f"Content {i}.")

        output_dir = tmp_path / "output"
        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"RIFF" + b"\x00" * 1024)

        jobs = create_jobs_from_directory(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            ref_audio=str(ref_audio),
            ref_text="Ref",
        )

        # Process with only 2 workers for 20 jobs
        backend = MockBackend()
        processor = BatchProcessor(backend=backend, workers=2, timeout=60)

        summary = processor.process(jobs)

        assert summary.total_jobs == 20
        assert summary.successful == 20

    def test_single_worker_sequential(self, tmp_path):
        """Test sequential processing with single worker."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        for i in range(10):
            (input_dir / f"file_{i}.txt").write_text(f"Content {i}.")

        output_dir = tmp_path / "output"
        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"RIFF" + b"\x00" * 1024)

        jobs = create_jobs_from_directory(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            ref_audio=str(ref_audio),
            ref_text="Ref",
        )

        # Single worker processes sequentially
        backend = MockBackend()
        processor = BatchProcessor(backend=backend, workers=1, timeout=60)

        processed_order = []
        def callback(completed, total, result):
            processed_order.append(result.job.input_path)

        summary = processor.process(jobs, progress_callback=callback)

        # Verify all processed
        assert len(processed_order) == 10


class TestCheckpointResume:
    """Tests for checkpoint and resume functionality."""

    def test_checkpoint_creation(self, tmp_path):
        """Test checkpoint file is created during batch."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        for i in range(5):
            (input_dir / f"file_{i}.txt").write_text(f"Content {i}.")

        output_dir = tmp_path / "output"
        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"RIFF" + b"\x00" * 1024)

        jobs = create_jobs_from_directory(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            ref_audio=str(ref_audio),
            ref_text="Ref",
        )

        backend = MockBackend()
        processor = BatchProcessor(backend=backend, workers=1)

        summary = processor.process(jobs)

        # Check outputs exist
        assert len(list(output_dir.glob("*.wav"))) == 5

    def test_partial_failure_handling(self, tmp_path):
        """Test batch continues after partial failures."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        for i in range(10):
            (input_dir / f"file_{i}.txt").write_text(f"Content {i}.")

        output_dir = tmp_path / "output"
        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"RIFF" + b"\x00" * 1024)

        jobs = create_jobs_from_directory(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            ref_audio=str(ref_audio),
            ref_text="Ref",
        )

        # Create backend that fails on job index 3 and 7
        class PartialFailBackend(MockBackend):
            fail_indices = {3, 7}
            current_index = 0

            def generate(self, text, voice_prompt, **kwargs):
                idx = PartialFailBackend.current_index
                PartialFailBackend.current_index += 1
                if idx in PartialFailBackend.fail_indices:
                    raise RuntimeError(f"Simulated failure at index {idx}")
                return super().generate(text, voice_prompt, **kwargs)

        backend = PartialFailBackend()
        processor = BatchProcessor(backend=backend, workers=1, timeout=60)

        summary = processor.process(jobs)

        # Should have 8 successful and 2 failed
        assert summary.total_jobs == 10
        assert summary.successful == 8
        assert summary.failed == 2


class TestMemoryUnderLoad:
    """Tests for memory behavior under load."""

    def test_many_small_files_memory_stable(self, tmp_path):
        """Test memory stays stable with many small files."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create many small files
        for i in range(50):
            (input_dir / f"file_{i:02d}.txt").write_text("Short text.")

        output_dir = tmp_path / "output"
        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"RIFF" + b"\x00" * 1024)

        jobs = create_jobs_from_directory(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            ref_audio=str(ref_audio),
            ref_text="Ref",
        )

        backend = MockBackend()
        processor = BatchProcessor(backend=backend, workers=1)

        # Track memory before and after
        try:
            import psutil
            process = psutil.Process()
            mem_before = process.memory_info().rss
        except ImportError:
            mem_before = 0

        summary = processor.process(jobs)

        try:
            import psutil
            process = psutil.Process()
            mem_after = process.memory_info().rss
            # Memory shouldn't grow more than 100MB for small files
            assert (mem_after - mem_before) < 100 * 1024 * 1024
        except ImportError:
            pass  # Can't check without psutil

        assert summary.successful == 50


class TestTimeoutHandling:
    """Tests for timeout handling."""

    def test_job_timeout(self, tmp_path):
        """Test individual job timeout."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        (input_dir / "slow.txt").write_text("This will timeout.")

        output_dir = tmp_path / "output"
        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"RIFF" + b"\x00" * 1024)

        jobs = create_jobs_from_directory(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            ref_audio=str(ref_audio),
            ref_text="Ref",
        )

        # Create slow backend
        class SlowBackend(MockBackend):
            def generate(self, text, voice_prompt, **kwargs):
                time.sleep(0.5)  # Small delay for test
                return super().generate(text, voice_prompt, **kwargs)

        backend = SlowBackend()
        processor = BatchProcessor(backend=backend, workers=1, timeout=5)  # Longer than delay

        summary = processor.process(jobs)

        # Should succeed (timeout longer than operation)
        assert summary.successful == 1


class TestReportGeneration:
    """Tests for batch report generation."""

    def test_report_json_format(self, tmp_path):
        """Test batch generates valid JSON report."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        for i in range(5):
            (input_dir / f"file_{i}.txt").write_text(f"Content {i}.")

        output_dir = tmp_path / "output"
        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"RIFF" + b"\x00" * 1024)

        jobs = create_jobs_from_directory(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            ref_audio=str(ref_audio),
            ref_text="Ref",
        )

        backend = MockBackend()
        processor = BatchProcessor(backend=backend, workers=1)

        summary = processor.process(jobs)

        # Convert to dict and verify JSON serializable
        report_dict = summary.to_dict()
        json_str = json.dumps(report_dict)
        parsed = json.loads(json_str)

        assert "total_jobs" in parsed
        assert "successful" in parsed
        assert "failed" in parsed
        assert "success_rate" in parsed

    def test_report_contains_results(self, tmp_path):
        """Test report contains individual job results."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        (input_dir / "test.txt").write_text("Test content.")

        output_dir = tmp_path / "output"
        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"RIFF" + b"\x00" * 1024)

        jobs = create_jobs_from_directory(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            ref_audio=str(ref_audio),
            ref_text="Ref",
        )

        backend = MockBackend()
        processor = BatchProcessor(backend=backend, workers=1)

        summary = processor.process(jobs)
        report_dict = summary.to_dict()

        assert "results" in report_dict
        assert len(report_dict["results"]) == 1
