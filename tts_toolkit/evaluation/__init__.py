"""TTS Evaluation module.

This module provides tools for evaluating TTS quality using various metrics:
- MOS prediction (UTMOS)
- Speaker similarity
- Word error rate (WER)
- Signal-based metrics (PESQ, STOI, MCD)

Example:
    from tts_toolkit.evaluation import TTSEvaluator

    evaluator = TTSEvaluator()
    results = evaluator.evaluate(
        audio_path="generated.wav",
        text="The text that was synthesized",
    )
    print(f"Predicted MOS: {results['utmos']:.2f}")
"""

from .evaluator import TTSEvaluator, BatchEvaluator
from .metrics import (
    predict_mos,
    speaker_similarity,
    word_error_rate,
    signal_metrics,
)
from .comparison import compare_backends

__all__ = [
    "TTSEvaluator",
    "BatchEvaluator",
    "predict_mos",
    "speaker_similarity",
    "word_error_rate",
    "signal_metrics",
    "compare_backends",
]
