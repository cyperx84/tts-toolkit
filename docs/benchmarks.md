# Performance Benchmarks

This document provides performance expectations for different TTS backends and configurations.

## Backend Comparison

### Synthesis Speed (Real-Time Factor)

| Backend | Device | RTF* | Notes |
|---------|--------|------|-------|
| Qwen 0.6B | GPU (A100) | 0.15x | Fastest GPU option |
| Qwen 0.6B | GPU (RTX 3090) | 0.25x | Good consumer GPU |
| Qwen 0.6B | GPU (RTX 3060) | 0.40x | Budget GPU |
| Qwen 0.6B | CPU (M1 Pro) | 1.2x | Apple Silicon |
| Qwen 0.6B | CPU (i7) | 2.5x | Intel desktop |
| Qwen 1.7B | GPU (A100) | 0.30x | Higher quality |
| Kokoro | CPU | 0.05x | Very fast, CPU-only |
| Fish Speech | API | 0.3-0.5x | Network dependent |
| Mock | Any | 0.001x | Testing only |

*RTF (Real-Time Factor): Time to generate / Audio duration. Lower is faster.
Example: RTF 0.25x means 15 seconds to generate 1 minute of audio.

### Memory Requirements

| Backend | Model Size | VRAM (GPU) | RAM (CPU) |
|---------|-----------|------------|-----------|
| Qwen 0.6B | ~2.5 GB | 4-6 GB | 8-12 GB |
| Qwen 1.7B | ~7 GB | 10-14 GB | 16-24 GB |
| Chatterbox | ~3 GB | 6-8 GB | 10-14 GB |
| Bark | ~6 GB | 8-12 GB | 16-20 GB |
| Kokoro | ~300 MB | N/A (CPU) | 1-2 GB |
| CosyVoice | ~2.5 GB | 4-6 GB | 8-12 GB |
| Fish Speech | N/A | N/A | ~500 MB |

### Quality vs Speed Trade-offs

| Setting | Impact on Speed | Impact on Quality |
|---------|-----------------|-------------------|
| Higher temperature | None | More expressive but variable |
| Larger model | Slower | Better prosody and naturalness |
| More chunk overlap | Slower stitching | Smoother transitions |
| x_vector_only mode | 30% faster | Slightly less accurate cloning |

## Hardware Recommendations

### Minimum Requirements

- **CPU-only workloads:**
  - 8 GB RAM
  - Any modern 4-core CPU
  - 10 GB disk space

- **GPU-accelerated:**
  - NVIDIA GPU with 6+ GB VRAM
  - 16 GB system RAM
  - CUDA 11.8 or 12.1

### Recommended for Production

- **Single user / development:**
  - RTX 3060 12GB or better
  - 32 GB RAM
  - SSD storage

- **Batch processing / multiple users:**
  - RTX 4090 24GB or A100
  - 64 GB RAM
  - NVMe storage

### Apple Silicon

- M1/M2/M3 chips work well with MPS acceleration
- Use `--device mps` for GPU acceleration
- Performance comparable to mid-range NVIDIA GPUs

## Batch Processing Performance

### Parallel Workers

| Workers | Speedup* | Memory | Recommended Use |
|---------|---------|--------|-----------------|
| 1 | 1.0x | Low | Single GPU |
| 2 | 1.8x | Medium | API backends |
| 4 | 3.2x | High | Fish Speech API |
| 8+ | Varies | Very High | Cloud/CPU clusters |

*Speedup depends heavily on backend. API backends benefit most from parallelism.

### Batch Processing Tips

1. **For GPU backends:** Use 1 worker (GPU is already parallel internally)
2. **For API backends:** Use 2-4 workers (network I/O bound)
3. **For CPU backends:** Use workers = CPU cores - 1

### Expected Throughput

| Scenario | Backend | Workers | Throughput |
|----------|---------|---------|------------|
| 100 short texts (~50 words) | Qwen GPU | 1 | 5-10 min |
| 100 short texts (~50 words) | Fish API | 4 | 3-5 min |
| 10 long texts (~1000 words) | Qwen GPU | 1 | 10-15 min |
| Audiobook (50,000 words) | Qwen GPU | 1 | 2-3 hours |

## Optimization Tips

### Reduce Latency

1. **Pre-load models:**
   ```python
   backend = QwenBackend()
   backend.load_model()  # Do this once at startup
   ```

2. **Cache voice prompts:**
   ```python
   voice_prompt = backend.create_voice_prompt(ref_audio, ref_text)
   # Reuse voice_prompt for multiple generations
   ```

3. **Use appropriate chunk sizes:**
   - Short texts (< 100 chars): No chunking needed
   - Medium texts (100-1000 chars): Default settings
   - Long texts (> 1000 chars): Increase chunk_max to 400

### Reduce Memory Usage

1. **Unload models when done:**
   ```python
   backend.unload_model()
   ```

2. **Clear GPU cache:**
   ```python
   from tts_toolkit.utils.memory import clear_gpu_cache
   clear_gpu_cache()
   ```

3. **Process large batches sequentially:**
   ```bash
   tts-toolkit batch --workers 1 ...
   ```

### Improve Quality

1. **Use high-quality reference audio:**
   - 10-30 seconds of clear speech
   - 24kHz or 44.1kHz sample rate
   - Minimal background noise

2. **Match reference to target:**
   - Similar speaking style
   - Same language
   - Matching emotional tone

3. **Fine-tune generation parameters:**
   ```bash
   --temperature 0.85  # More consistent
   --top_k 30          # Less random
   --top_p 0.9         # Focused sampling
   ```

## Running Benchmarks

To run benchmarks on your own hardware:

```bash
# Install with benchmark dependencies
pip install tts-toolkit[eval]

# Run benchmark script (if available)
python -m tts_toolkit.benchmarks.run

# Or manually time operations:
time tts-toolkit pipeline \
    --text "Your benchmark text here..." \
    --ref-audio ref.wav \
    --ref-text "Reference" \
    --output benchmark.wav \
    --backend qwen
```

## Reporting Performance Issues

When reporting performance issues, please include:

1. **Hardware specs:**
   - CPU model
   - GPU model and VRAM
   - RAM amount
   - Storage type (SSD/HDD)

2. **Software versions:**
   ```bash
   python --version
   pip show tts-toolkit torch
   nvidia-smi  # for GPU info
   ```

3. **Benchmark results:**
   - Input text length
   - Output audio duration
   - Wall clock time
   - Peak memory usage

4. **Configuration:**
   - Backend used
   - Chunk settings
   - Number of workers
