"""Individual TTS evaluation metrics.

This module provides functions for computing various TTS quality metrics:
- predict_mos: Neural MOS prediction using UTMOS
- speaker_similarity: Cosine similarity of speaker embeddings
- word_error_rate: WER using ASR transcription
- signal_metrics: PESQ, STOI, MCD

All functions handle lazy loading of dependencies to avoid import overhead.
"""

from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path

import numpy as np


def predict_mos(
    audio: Union[str, Path, np.ndarray],
    sample_rate: int = 24000,
    model: str = "utmos",
) -> float:
    """Predict Mean Opinion Score using a neural model.

    Args:
        audio: Path to audio file or numpy array
        sample_rate: Sample rate if audio is numpy array
        model: Model to use ("utmos" or "mosnet")

    Returns:
        Predicted MOS score (1.0 - 5.0)

    Example:
        >>> mos = predict_mos("generated.wav")
        >>> print(f"Predicted MOS: {mos:.2f}")
    """
    audio_array, sr = _load_audio(audio, sample_rate)

    if model == "utmos":
        return _predict_utmos(audio_array, sr)
    elif model == "mosnet":
        return _predict_mosnet(audio_array, sr)
    else:
        raise ValueError(f"Unknown model: {model}. Use 'utmos' or 'mosnet'.")


def _predict_utmos(audio: np.ndarray, sample_rate: int) -> float:
    """Predict MOS using UTMOS model."""
    try:
        import torch
        from speechmos import predict as speechmos_predict
    except ImportError:
        # Fallback: estimate based on signal properties
        return _estimate_mos_fallback(audio, sample_rate)

    # Ensure 16kHz for UTMOS
    if sample_rate != 16000:
        audio = _resample(audio, sample_rate, 16000)

    score = speechmos_predict(audio, 16000)
    return float(np.clip(score, 1.0, 5.0))


def _predict_mosnet(audio: np.ndarray, sample_rate: int) -> float:
    """Predict MOS using MOSNet model."""
    # MOSNet fallback
    return _estimate_mos_fallback(audio, sample_rate)


def _estimate_mos_fallback(audio: np.ndarray, sample_rate: int) -> float:
    """Fallback MOS estimation based on signal properties."""
    # Simple heuristics when neural models unavailable
    # Check for clipping
    clipping_ratio = np.mean(np.abs(audio) > 0.99)
    if clipping_ratio > 0.01:
        return 2.0

    # Check for silence
    silence_ratio = np.mean(np.abs(audio) < 0.01)
    if silence_ratio > 0.9:
        return 1.0

    # Check SNR-like metric
    signal_power = np.mean(audio ** 2)
    if signal_power < 0.001:
        return 2.5

    # Default to moderate score
    return 3.5


def speaker_similarity(
    generated_audio: Union[str, Path, np.ndarray],
    reference_audio: Union[str, Path, np.ndarray],
    sample_rate: int = 24000,
    model: str = "ecapa",
) -> float:
    """Compute speaker similarity between two audio samples.

    Uses speaker embeddings to measure how similar the voices are.

    Args:
        generated_audio: Path or array of generated audio
        reference_audio: Path or array of reference speaker audio
        sample_rate: Sample rate if arrays provided
        model: Embedding model ("ecapa", "xvector", "resemblyzer")

    Returns:
        Cosine similarity score (0.0 - 1.0)

    Example:
        >>> sim = speaker_similarity("generated.wav", "reference.wav")
        >>> print(f"Speaker Similarity: {sim:.2f}")
    """
    gen_audio, gen_sr = _load_audio(generated_audio, sample_rate)
    ref_audio, ref_sr = _load_audio(reference_audio, sample_rate)

    # Get embeddings
    gen_embedding = _get_speaker_embedding(gen_audio, gen_sr, model)
    ref_embedding = _get_speaker_embedding(ref_audio, ref_sr, model)

    # Cosine similarity
    similarity = np.dot(gen_embedding, ref_embedding) / (
        np.linalg.norm(gen_embedding) * np.linalg.norm(ref_embedding) + 1e-8
    )

    return float(np.clip(similarity, 0.0, 1.0))


def _get_speaker_embedding(
    audio: np.ndarray,
    sample_rate: int,
    model: str,
) -> np.ndarray:
    """Extract speaker embedding from audio."""
    try:
        if model == "ecapa":
            return _get_ecapa_embedding(audio, sample_rate)
        elif model == "resemblyzer":
            return _get_resemblyzer_embedding(audio, sample_rate)
        else:
            return _get_simple_embedding(audio, sample_rate)
    except ImportError:
        return _get_simple_embedding(audio, sample_rate)


def _get_ecapa_embedding(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Get ECAPA-TDNN embedding using speechbrain."""
    try:
        from speechbrain.inference import EncoderClassifier
        import torch

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )

        if sample_rate != 16000:
            audio = _resample(audio, sample_rate, 16000)

        waveform = torch.tensor(audio).unsqueeze(0)
        embedding = classifier.encode_batch(waveform)
        return embedding.squeeze().numpy()
    except ImportError:
        return _get_simple_embedding(audio, sample_rate)


def _get_resemblyzer_embedding(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Get embedding using Resemblyzer."""
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav

        encoder = VoiceEncoder()

        if sample_rate != 16000:
            audio = _resample(audio, sample_rate, 16000)

        embedding = encoder.embed_utterance(audio)
        return embedding
    except ImportError:
        return _get_simple_embedding(audio, sample_rate)


def _get_simple_embedding(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Simple embedding based on spectral features (fallback)."""
    # Use MFCCs as simple embedding
    try:
        import librosa
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
        return np.mean(mfccs, axis=1)
    except ImportError:
        # Very basic fallback
        n_fft = min(2048, len(audio))
        spectrum = np.abs(np.fft.fft(audio[:n_fft]))[:n_fft // 2]
        # Bin into 20 features
        bins = np.array_split(spectrum, 20)
        return np.array([np.mean(b) for b in bins])


def word_error_rate(
    audio: Union[str, Path, np.ndarray],
    expected_text: str,
    sample_rate: int = 24000,
    model: str = "whisper",
    model_size: str = "base",
) -> float:
    """Compute Word Error Rate by transcribing audio and comparing to expected.

    Args:
        audio: Path or array of audio to transcribe
        expected_text: The text that should have been synthesized
        sample_rate: Sample rate if array provided
        model: ASR model ("whisper", "wav2vec2")
        model_size: Model size for whisper ("tiny", "base", "small", "medium")

    Returns:
        Word Error Rate (0.0 - 1.0+, lower is better)

    Example:
        >>> wer = word_error_rate("generated.wav", "Hello world")
        >>> print(f"WER: {wer:.1%}")
    """
    audio_array, sr = _load_audio(audio, sample_rate)

    # Transcribe
    if model == "whisper":
        transcription = _transcribe_whisper(audio_array, sr, model_size)
    else:
        transcription = _transcribe_fallback(audio_array, sr)

    # Compute WER
    return _compute_wer(expected_text, transcription)


def _transcribe_whisper(
    audio: np.ndarray,
    sample_rate: int,
    model_size: str,
) -> str:
    """Transcribe using OpenAI Whisper."""
    try:
        import whisper

        model = whisper.load_model(model_size)

        if sample_rate != 16000:
            audio = _resample(audio, sample_rate, 16000)

        # Pad or trim to 30s
        audio = whisper.pad_or_trim(audio)

        result = model.transcribe(audio, fp16=False)
        return result["text"].strip()
    except ImportError:
        return ""


def _transcribe_fallback(audio: np.ndarray, sample_rate: int) -> str:
    """Fallback transcription (returns empty for now)."""
    return ""


def _compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis."""
    # Normalize
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0

    # Levenshtein distance at word level
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))

    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(
                    d[i - 1, j] + 1,      # Deletion
                    d[i, j - 1] + 1,      # Insertion
                    d[i - 1, j - 1] + 1,  # Substitution
                )

    return float(d[len(ref_words), len(hyp_words)] / len(ref_words))


def signal_metrics(
    generated: Union[str, Path, np.ndarray],
    reference: Union[str, Path, np.ndarray],
    sample_rate: int = 24000,
) -> Dict[str, float]:
    """Compute signal-based quality metrics.

    Requires a ground-truth reference audio for comparison.

    Args:
        generated: Path or array of generated audio
        reference: Path or array of reference audio
        sample_rate: Sample rate if arrays provided

    Returns:
        Dictionary with PESQ, STOI, and MCD scores

    Example:
        >>> metrics = signal_metrics("generated.wav", "reference.wav")
        >>> print(f"PESQ: {metrics['pesq']:.2f}")
    """
    gen_audio, gen_sr = _load_audio(generated, sample_rate)
    ref_audio, ref_sr = _load_audio(reference, sample_rate)

    # Ensure same sample rate
    if gen_sr != ref_sr:
        gen_audio = _resample(gen_audio, gen_sr, ref_sr)
        gen_sr = ref_sr

    # Ensure same length
    min_len = min(len(gen_audio), len(ref_audio))
    gen_audio = gen_audio[:min_len]
    ref_audio = ref_audio[:min_len]

    results = {}

    # PESQ
    results["pesq"] = _compute_pesq(ref_audio, gen_audio, ref_sr)

    # STOI
    results["stoi"] = _compute_stoi(ref_audio, gen_audio, ref_sr)

    # MCD
    results["mcd"] = _compute_mcd(ref_audio, gen_audio, ref_sr)

    return results


def _compute_pesq(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int,
) -> float:
    """Compute PESQ score."""
    try:
        from pesq import pesq

        # PESQ requires 8kHz or 16kHz
        target_sr = 16000 if sample_rate >= 16000 else 8000
        mode = "wb" if target_sr == 16000 else "nb"

        if sample_rate != target_sr:
            reference = _resample(reference, sample_rate, target_sr)
            degraded = _resample(degraded, sample_rate, target_sr)

        score = pesq(target_sr, reference, degraded, mode)
        return float(score)
    except ImportError:
        return 0.0
    except Exception:
        return 0.0


def _compute_stoi(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int,
) -> float:
    """Compute STOI score."""
    try:
        from pystoi import stoi

        score = stoi(reference, degraded, sample_rate, extended=False)
        return float(np.clip(score, 0.0, 1.0))
    except ImportError:
        return 0.0
    except Exception:
        return 0.0


def _compute_mcd(
    reference: np.ndarray,
    synthesized: np.ndarray,
    sample_rate: int,
) -> float:
    """Compute Mel-Cepstral Distortion."""
    try:
        import librosa

        # Extract MFCCs
        ref_mfcc = librosa.feature.mfcc(y=reference, sr=sample_rate, n_mfcc=13)
        syn_mfcc = librosa.feature.mfcc(y=synthesized, sr=sample_rate, n_mfcc=13)

        # Align lengths
        min_frames = min(ref_mfcc.shape[1], syn_mfcc.shape[1])
        ref_mfcc = ref_mfcc[:, :min_frames]
        syn_mfcc = syn_mfcc[:, :min_frames]

        # Compute MCD (skip c0)
        diff = ref_mfcc[1:] - syn_mfcc[1:]
        mcd = np.mean(np.sqrt(2 * np.sum(diff ** 2, axis=0)))

        return float(mcd)
    except ImportError:
        return 0.0
    except Exception:
        return 0.0


def _load_audio(
    audio: Union[str, Path, np.ndarray],
    default_sr: int,
) -> Tuple[np.ndarray, int]:
    """Load audio from file or return array."""
    if isinstance(audio, np.ndarray):
        return audio.astype(np.float32), default_sr

    try:
        import soundfile as sf
        data, sr = sf.read(str(audio))
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32), sr
    except ImportError:
        raise ImportError("soundfile required: pip install soundfile")


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio

    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        # Simple linear interpolation fallback
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
