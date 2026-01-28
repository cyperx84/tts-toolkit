"""Backend comparison utilities.

Compare multiple TTS backends on the same test set.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import tempfile
import os

import numpy as np

from .evaluator import TTSEvaluator


@dataclass
class ComparisonResult:
    """Results from comparing multiple backends."""

    backends: List[str]
    metrics: Dict[str, Dict[str, float]]
    per_sentence: List[Dict[str, Any]] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate markdown table of results."""
        lines = ["# Backend Comparison Results", ""]

        # Build header
        metric_names = ["MOS", "Speaker Sim", "WER"]
        header = "| Backend | " + " | ".join(metric_names) + " |"
        separator = "|---------|" + "|".join(["-------"] * len(metric_names)) + "|"

        lines.extend([header, separator])

        # Build rows
        for backend in self.backends:
            m = self.metrics.get(backend, {})
            row = f"| {backend} | "
            row += f"{m.get('utmos', 0):.2f} | "
            row += f"{m.get('speaker_similarity', 0):.2f} | "
            wer = m.get('wer', 0)
            row += f"{wer:.1%} |" if wer else "N/A |"
            lines.append(row)

        return "\n".join(lines)

    def best_backend(self, metric: str = "utmos") -> str:
        """Get the best performing backend for a metric.

        Args:
            metric: Metric to optimize ("utmos", "speaker_similarity", "wer")

        Returns:
            Name of best backend
        """
        reverse = metric != "wer"  # Lower WER is better

        scores = [(name, self.metrics[name].get(metric, 0)) for name in self.backends]
        scores = [(n, s) for n, s in scores if s is not None]

        if not scores:
            return self.backends[0]

        scores.sort(key=lambda x: x[1], reverse=reverse)
        return scores[0][0]


def compare_backends(
    backends: Dict[str, Any],
    test_sentences: List[str],
    reference_audio: Union[str, Path],
    reference_text: str,
    output_dir: Optional[Union[str, Path]] = None,
    evaluator_kwargs: Optional[Dict[str, Any]] = None,
) -> ComparisonResult:
    """Compare multiple TTS backends on the same test sentences.

    Args:
        backends: Dictionary mapping backend names to backend instances
        test_sentences: List of sentences to synthesize
        reference_audio: Speaker reference audio for voice cloning
        reference_text: Transcript of reference audio
        output_dir: Directory to save generated audio (temp if None)
        evaluator_kwargs: Arguments for TTSEvaluator

    Returns:
        ComparisonResult with metrics for each backend

    Example:
        from tts_toolkit.backends import QwenBackend, ChatterboxBackend

        results = compare_backends(
            backends={
                "qwen": QwenBackend(device="cuda"),
                "chatterbox": ChatterboxBackend(device="cuda"),
            },
            test_sentences=["Hello world.", "How are you?"],
            reference_audio="speaker.wav",
            reference_text="Reference transcript",
        )
        print(results.to_markdown())
    """
    evaluator_kwargs = evaluator_kwargs or {}
    evaluator = TTSEvaluator(**evaluator_kwargs)

    # Create output directory
    if output_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="tts_comparison_")
        output_dir = Path(temp_dir)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Collect results
    all_results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in backends}
    per_sentence: List[Dict[str, Any]] = []

    # Generate and evaluate for each backend
    for backend_name, backend in backends.items():
        print(f"Evaluating {backend_name}...")

        # Load model
        backend.load_model()

        # Create voice prompt
        voice_prompt = backend.create_voice_prompt(
            str(reference_audio),
            reference_text,
        )

        for i, sentence in enumerate(test_sentences):
            # Generate audio
            output_path = output_dir / f"{backend_name}_{i:03d}.wav"

            try:
                audio, sr = backend.generate(sentence, voice_prompt)

                # Save audio
                import soundfile as sf
                sf.write(str(output_path), audio, sr)

                # Evaluate
                result = evaluator.evaluate(
                    audio_path=output_path,
                    text=sentence,
                    reference_audio=reference_audio,
                )
                result["backend"] = backend_name
                result["sentence_idx"] = i
                result["sentence"] = sentence

                all_results[backend_name].append(result)
                per_sentence.append(result)

            except Exception as e:
                print(f"  Error on sentence {i}: {e}")
                all_results[backend_name].append({
                    "backend": backend_name,
                    "sentence_idx": i,
                    "error": str(e),
                })

    # Aggregate metrics per backend
    metrics: Dict[str, Dict[str, float]] = {}

    for backend_name, results in all_results.items():
        metrics[backend_name] = {}

        for metric in ["utmos", "speaker_similarity", "wer"]:
            values = [r.get(metric) for r in results if r.get(metric) is not None]
            if values:
                metrics[backend_name][metric] = float(np.mean(values))

    return ComparisonResult(
        backends=list(backends.keys()),
        metrics=metrics,
        per_sentence=per_sentence,
    )


def ab_test(
    system_a_dir: Union[str, Path],
    system_b_dir: Union[str, Path],
    output_html: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Set up an A/B test between two TTS systems.

    Args:
        system_a_dir: Directory with audio from system A
        system_b_dir: Directory with audio from system B
        output_html: Path to save evaluation HTML page

    Returns:
        Dictionary with test configuration
    """
    system_a_dir = Path(system_a_dir)
    system_b_dir = Path(system_b_dir)

    # Find matching files
    a_files = {f.name: f for f in system_a_dir.glob("*.wav")}
    b_files = {f.name: f for f in system_b_dir.glob("*.wav")}

    common_files = set(a_files.keys()) & set(b_files.keys())

    pairs = []
    for filename in sorted(common_files):
        pairs.append({
            "id": filename,
            "system_a": str(a_files[filename]),
            "system_b": str(b_files[filename]),
        })

    result = {
        "num_pairs": len(pairs),
        "pairs": pairs,
    }

    if output_html:
        _generate_ab_html(pairs, output_html)
        result["html_path"] = str(output_html)

    return result


def _generate_ab_html(pairs: List[Dict], output_path: Union[str, Path]) -> None:
    """Generate HTML page for A/B testing."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>TTS A/B Test</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .pair { margin: 20px 0; padding: 20px; border: 1px solid #ccc; border-radius: 8px; }
        .buttons { margin-top: 10px; }
        button { padding: 10px 20px; margin: 5px; cursor: pointer; }
        .selected { background-color: #4CAF50; color: white; }
    </style>
</head>
<body>
    <h1>TTS A/B Test</h1>
    <p>Listen to both samples and select which sounds better.</p>

    <div id="pairs">
"""

    for i, pair in enumerate(pairs):
        html += f"""
        <div class="pair" id="pair-{i}">
            <h3>Sample {i + 1}</h3>
            <div>
                <strong>Option A:</strong>
                <audio controls src="{pair['system_a']}"></audio>
            </div>
            <div>
                <strong>Option B:</strong>
                <audio controls src="{pair['system_b']}"></audio>
            </div>
            <div class="buttons">
                <button onclick="select({i}, 'A')">A is better</button>
                <button onclick="select({i}, 'B')">B is better</button>
                <button onclick="select({i}, 'tie')">No preference</button>
            </div>
        </div>
"""

    html += """
    </div>

    <button onclick="exportResults()">Export Results</button>

    <script>
        const results = {};

        function select(pairId, choice) {
            results[pairId] = choice;
            document.querySelectorAll(`#pair-${pairId} button`).forEach(b => b.classList.remove('selected'));
            event.target.classList.add('selected');
        }

        function exportResults() {
            const blob = new Blob([JSON.stringify(results, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'ab_test_results.json';
            a.click();
        }
    </script>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)
