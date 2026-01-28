"""TTS Evaluator classes.

Provides TTSEvaluator for single-file evaluation and BatchEvaluator
for evaluating multiple files or directories.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

import numpy as np

from .metrics import (
    predict_mos,
    speaker_similarity,
    word_error_rate,
    signal_metrics,
)


@dataclass
class EvaluationResult:
    """Result of a single TTS evaluation."""

    audio_path: str
    utmos: Optional[float] = None
    speaker_similarity: Optional[float] = None
    wer: Optional[float] = None
    pesq: Optional[float] = None
    stoi: Optional[float] = None
    mcd: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "audio_path": self.audio_path,
            "utmos": self.utmos,
            "speaker_similarity": self.speaker_similarity,
            "wer": self.wer,
            "pesq": self.pesq,
            "stoi": self.stoi,
            "mcd": self.mcd,
            "metadata": self.metadata,
        }


class TTSEvaluator:
    """Evaluate TTS quality for single audio files.

    Computes multiple metrics including MOS prediction, speaker similarity,
    and word error rate.

    Example:
        evaluator = TTSEvaluator()
        results = evaluator.evaluate(
            audio_path="generated.wav",
            text="The text that was synthesized",
            reference_audio="speaker_reference.wav",
        )
        print(f"MOS: {results['utmos']:.2f}")
    """

    def __init__(
        self,
        compute_mos: bool = True,
        compute_speaker_sim: bool = True,
        compute_wer: bool = True,
        compute_signal_metrics: bool = False,
        whisper_model: str = "base",
        speaker_model: str = "ecapa",
    ):
        """Initialize evaluator.

        Args:
            compute_mos: Whether to compute MOS prediction
            compute_speaker_sim: Whether to compute speaker similarity
            compute_wer: Whether to compute word error rate
            compute_signal_metrics: Whether to compute PESQ/STOI/MCD
            whisper_model: Whisper model size for WER
            speaker_model: Speaker embedding model
        """
        self.compute_mos = compute_mos
        self.compute_speaker_sim = compute_speaker_sim
        self.compute_wer = compute_wer
        self.compute_signal_metrics = compute_signal_metrics
        self.whisper_model = whisper_model
        self.speaker_model = speaker_model

    def evaluate(
        self,
        audio_path: Union[str, Path],
        text: Optional[str] = None,
        reference_audio: Optional[Union[str, Path]] = None,
        reference_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single TTS audio file.

        Args:
            audio_path: Path to generated audio
            text: Original text that was synthesized (for WER)
            reference_audio: Speaker reference audio (for similarity)
            reference_path: Ground truth audio (for signal metrics)

        Returns:
            Dictionary with evaluation metrics
        """
        audio_path = str(audio_path)
        results: Dict[str, Any] = {"audio_path": audio_path}

        # MOS prediction
        if self.compute_mos:
            try:
                results["utmos"] = predict_mos(audio_path)
            except Exception as e:
                results["utmos"] = None
                results["utmos_error"] = str(e)

        # Speaker similarity
        if self.compute_speaker_sim and reference_audio is not None:
            try:
                results["speaker_similarity"] = speaker_similarity(
                    audio_path,
                    str(reference_audio),
                    model=self.speaker_model,
                )
            except Exception as e:
                results["speaker_similarity"] = None
                results["speaker_similarity_error"] = str(e)

        # Word Error Rate
        if self.compute_wer and text is not None:
            try:
                results["wer"] = word_error_rate(
                    audio_path,
                    text,
                    model_size=self.whisper_model,
                )
            except Exception as e:
                results["wer"] = None
                results["wer_error"] = str(e)

        # Signal metrics (PESQ, STOI, MCD)
        if self.compute_signal_metrics and reference_path is not None:
            try:
                sig_metrics = signal_metrics(audio_path, str(reference_path))
                results.update(sig_metrics)
            except Exception as e:
                results["signal_metrics_error"] = str(e)

        return results


class BatchEvaluator:
    """Evaluate multiple TTS audio files.

    Provides batch evaluation with summary statistics.

    Example:
        evaluator = BatchEvaluator()
        results = evaluator.evaluate_directory(
            generated_dir="outputs/",
            transcripts_file="transcripts.json",
        )
        print(evaluator.summary())
    """

    def __init__(self, **kwargs):
        """Initialize batch evaluator.

        Args:
            **kwargs: Arguments passed to TTSEvaluator
        """
        self.evaluator = TTSEvaluator(**kwargs)
        self.results: List[Dict[str, Any]] = []

    def evaluate_directory(
        self,
        generated_dir: Union[str, Path],
        reference_dir: Optional[Union[str, Path]] = None,
        speaker_reference: Optional[Union[str, Path]] = None,
        transcripts_file: Optional[Union[str, Path]] = None,
        pattern: str = "*.wav",
    ) -> List[Dict[str, Any]]:
        """Evaluate all audio files in a directory.

        Args:
            generated_dir: Directory with generated audio files
            reference_dir: Directory with ground-truth audio (optional)
            speaker_reference: Single speaker reference file (optional)
            transcripts_file: JSON file mapping filenames to transcripts
            pattern: Glob pattern for audio files

        Returns:
            List of evaluation results
        """
        from pathlib import Path

        generated_dir = Path(generated_dir)
        if reference_dir:
            reference_dir = Path(reference_dir)

        # Load transcripts if provided
        transcripts = {}
        if transcripts_file:
            with open(transcripts_file) as f:
                transcripts = json.load(f)

        self.results = []

        for audio_file in sorted(generated_dir.glob(pattern)):
            filename = audio_file.name

            # Get transcript
            text = transcripts.get(filename) or transcripts.get(audio_file.stem)

            # Get reference audio (ground truth)
            ref_path = None
            if reference_dir:
                ref_file = reference_dir / filename
                if ref_file.exists():
                    ref_path = ref_file

            # Evaluate
            result = self.evaluator.evaluate(
                audio_path=audio_file,
                text=text,
                reference_audio=speaker_reference,
                reference_path=ref_path,
            )

            self.results.append(result)

        return self.results

    def evaluate_files(
        self,
        files: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Evaluate a list of files with metadata.

        Args:
            files: List of dicts with 'audio_path' and optional 'text',
                   'reference_audio', 'reference_path'

        Returns:
            List of evaluation results
        """
        self.results = []

        for file_info in files:
            result = self.evaluator.evaluate(**file_info)
            self.results.append(result)

        return self.results

    def summary(self) -> str:
        """Generate summary statistics.

        Returns:
            Formatted summary string
        """
        if not self.results:
            return "No results to summarize."

        lines = ["TTS Evaluation Summary", "=" * 40]

        # Compute statistics for each metric
        metrics = ["utmos", "speaker_similarity", "wer", "pesq", "stoi", "mcd"]
        metric_names = {
            "utmos": "Predicted MOS",
            "speaker_similarity": "Speaker Similarity",
            "wer": "Word Error Rate",
            "pesq": "PESQ",
            "stoi": "STOI",
            "mcd": "MCD",
        }

        for metric in metrics:
            values = [r.get(metric) for r in self.results if r.get(metric) is not None]

            if values:
                mean = np.mean(values)
                std = np.std(values)

                if metric == "wer":
                    lines.append(f"{metric_names[metric]}: {mean:.1%} ± {std:.1%}")
                elif metric == "mcd":
                    lines.append(f"{metric_names[metric]}: {mean:.2f} ± {std:.2f} (lower=better)")
                else:
                    lines.append(f"{metric_names[metric]}: {mean:.2f} ± {std:.2f}")

        lines.append(f"\nTotal files evaluated: {len(self.results)}")

        return "\n".join(lines)

    def to_dataframe(self):
        """Convert results to pandas DataFrame.

        Returns:
            pandas.DataFrame with results

        Raises:
            ImportError: If pandas not installed
        """
        try:
            import pandas as pd
            return pd.DataFrame(self.results)
        except ImportError:
            raise ImportError("pandas required: pip install pandas")

    def save_results(self, path: Union[str, Path]) -> None:
        """Save results to JSON file.

        Args:
            path: Output file path
        """
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
