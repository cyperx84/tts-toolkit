# TTS Evaluation Guide

This guide covers methods for evaluating text-to-speech quality, speaker similarity, and intelligibility.

## Overview

TTS evaluation uses both **subjective** (human-rated) and **objective** (automated) metrics:

| Metric Type | Examples | Use Case |
|-------------|----------|----------|
| Subjective | MOS, CMOS, MUSHRA | Gold standard, final validation |
| Objective | PESQ, STOI, MCD | Development iteration, CI/CD |
| Neural | UTMOS, SpeechBERT | Correlates better with human judgment |
| Task-specific | WER, Speaker Similarity | Specific quality aspects |

## Quick Start

```python
from tts_toolkit.evaluation import TTSEvaluator

evaluator = TTSEvaluator()

# Evaluate a single audio file
results = evaluator.evaluate(
    audio_path="generated.wav",
    reference_path="ground_truth.wav",  # Optional
    text="The text that was synthesized",
)

print(f"Predicted MOS: {results['utmos']:.2f}")
print(f"Speaker Similarity: {results['speaker_similarity']:.2f}")
print(f"Word Error Rate: {results['wer']:.1%}")
```

## Evaluation Metrics

### 1. Mean Opinion Score (MOS)

The gold standard for TTS evaluation. Human listeners rate naturalness on a 1-5 scale.

| Score | Quality |
|-------|---------|
| 5 | Excellent - Indistinguishable from human |
| 4 | Good - Minor artifacts |
| 3 | Fair - Noticeable but acceptable |
| 2 | Poor - Distracting artifacts |
| 1 | Bad - Unintelligible |

**Automated MOS Prediction (UTMOS):**

```python
from tts_toolkit.evaluation import predict_mos

# Predict MOS without human listeners
predicted_mos = predict_mos("generated.wav")
print(f"Predicted MOS: {predicted_mos:.2f}")
```

### 2. Speaker Similarity

Measures how well the generated voice matches the target speaker.

```python
from tts_toolkit.evaluation import speaker_similarity

similarity = speaker_similarity(
    generated_audio="output.wav",
    reference_audio="target_speaker.wav",
)
# Returns 0.0 - 1.0 (cosine similarity of speaker embeddings)
print(f"Speaker Similarity: {similarity:.2f}")
```

| Score | Interpretation |
|-------|----------------|
| > 0.85 | Excellent match |
| 0.70 - 0.85 | Good match |
| 0.50 - 0.70 | Moderate match |
| < 0.50 | Poor match |

### 3. Word Error Rate (WER)

Measures intelligibility by transcribing generated audio and comparing to input text.

```python
from tts_toolkit.evaluation import word_error_rate

wer = word_error_rate(
    audio_path="generated.wav",
    expected_text="The quick brown fox jumps over the lazy dog.",
)
print(f"WER: {wer:.1%}")
```

| WER | Quality |
|-----|---------|
| < 5% | Excellent |
| 5-10% | Good |
| 10-20% | Acceptable |
| > 20% | Poor |

### 4. Signal-Based Metrics

These require a ground-truth reference audio.

```python
from tts_toolkit.evaluation import signal_metrics

metrics = signal_metrics(
    generated="output.wav",
    reference="ground_truth.wav",
)

print(f"PESQ: {metrics['pesq']:.2f}")  # -0.5 to 4.5
print(f"STOI: {metrics['stoi']:.2f}")  # 0 to 1
print(f"MCD: {metrics['mcd']:.2f}")    # Lower is better
```

| Metric | Range | Description |
|--------|-------|-------------|
| PESQ | -0.5 to 4.5 | Perceptual quality (phone-band) |
| STOI | 0 to 1 | Speech intelligibility |
| MCD | 0 to ∞ | Mel-cepstral distortion (lower = better) |

## Batch Evaluation

Evaluate multiple files or compare backends:

```python
from tts_toolkit.evaluation import BatchEvaluator

evaluator = BatchEvaluator()

# Evaluate a directory of generated files
results = evaluator.evaluate_directory(
    generated_dir="outputs/",
    reference_dir="references/",  # Optional
    transcripts_file="transcripts.json",
)

# Get summary statistics
print(evaluator.summary())
# Output:
# Average MOS: 4.12 ± 0.34
# Average Speaker Similarity: 0.82 ± 0.08
# Average WER: 3.2%
```

## Backend Comparison

Compare TTS backends on the same test set:

```python
from tts_toolkit.evaluation import compare_backends
from tts_toolkit.backends import QwenBackend, ChatterboxBackend, KokoroBackend

results = compare_backends(
    backends={
        "qwen": QwenBackend(device="cuda"),
        "chatterbox": ChatterboxBackend(device="cuda"),
        "kokoro": KokoroBackend(voice="af_heart"),
    },
    test_sentences=[
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "How much wood would a woodchuck chuck?",
    ],
    reference_audio="speaker.wav",
    reference_text="Speaker reference transcript",
)

# Print comparison table
print(results.to_markdown())
```

## Evaluation in CI/CD

Add quality gates to your pipeline:

```python
from tts_toolkit.evaluation import TTSEvaluator

def test_tts_quality():
    evaluator = TTSEvaluator()

    # Generate test audio
    backend.generate("Test sentence", voice_prompt)

    # Evaluate
    results = evaluator.evaluate("output.wav", text="Test sentence")

    # Assert quality thresholds
    assert results["utmos"] >= 3.5, f"MOS too low: {results['utmos']}"
    assert results["wer"] <= 0.10, f"WER too high: {results['wer']}"
    assert results["speaker_similarity"] >= 0.70, "Speaker mismatch"
```

## Advanced: Custom Evaluation

### Using TTSDS2 (Distribution Score)

The TTSDS2 metric evaluates how natural the generated speech distribution is:

```python
from tts_toolkit.evaluation import ttsds_score

score = ttsds_score(
    generated_dir="outputs/",
    natural_reference_dir="natural_speech/",
)
print(f"TTSDS2 Score: {score:.3f}")
```

### A/B Testing with Human Evaluation

```python
from tts_toolkit.evaluation import ABTest

test = ABTest(
    system_a="outputs_qwen/",
    system_b="outputs_chatterbox/",
)

# Generate evaluation interface
test.create_evaluation_page("evaluation.html")

# Collect results
results = test.analyze_results("responses.json")
print(f"System A preference: {results['a_preference']:.1%}")
print(f"Statistical significance: p={results['p_value']:.4f}")
```

## Metrics Reference

### Subjective Metrics

| Metric | Scale | Description |
|--------|-------|-------------|
| MOS | 1-5 | Overall naturalness |
| CMOS | -3 to +3 | Comparative (A vs B) |
| MUSHRA | 0-100 | Multiple systems ranking |
| DMOS | 1-5 | Degradation from reference |

### Objective Metrics

| Metric | Range | Best | Notes |
|--------|-------|------|-------|
| UTMOS | 1-5 | Higher | Neural MOS predictor |
| PESQ | -0.5 to 4.5 | Higher | Phone-band quality |
| POLQA | 1-5 | Higher | Modern PESQ replacement |
| STOI | 0-1 | Higher | Intelligibility |
| MCD | 0+ | Lower | Spectral distance |
| F0 RMSE | 0+ | Lower | Pitch accuracy |
| V/UV Error | 0-1 | Lower | Voicing accuracy |

### Task-Specific Metrics

| Metric | Range | Best | Use Case |
|--------|-------|------|----------|
| WER | 0-1 | Lower | Intelligibility via ASR |
| Speaker Sim | 0-1 | Higher | Voice cloning quality |
| Emotion Acc | 0-1 | Higher | Emotion transfer |
| Prosody DTW | 0+ | Lower | Rhythm/intonation |

## Installation

```bash
# Basic evaluation
pip install tts-toolkit[eval]

# Full evaluation suite (includes neural metrics)
pip install tts-toolkit[eval-full]
```

## References

- [UTMOS](https://github.com/tarepan/SpeechMOS) - Neural MOS predictor
- [PESQ](https://github.com/ludlows/PESQ) - Perceptual quality
- [speechbrain](https://speechbrain.github.io/) - Speaker embeddings
- [whisper](https://github.com/openai/whisper) - ASR for WER
- [TTSDS2](https://arxiv.org/abs/2407.12707) - Distribution score
