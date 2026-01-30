# Troubleshooting Guide

This guide covers common issues and their solutions when using TTS Toolkit.

## Installation Issues

### ImportError: No module named 'qwen_tts'

**Problem:** The Qwen backend requires the `qwen-tts` package.

**Solution:**
```bash
pip install tts-toolkit[qwen]
# or
pip install qwen-tts torch
```

### ImportError: No module named 'chatterbox'

**Problem:** The Chatterbox backend requires the `chatterbox-tts` package.

**Solution:**
```bash
pip install tts-toolkit[chatterbox]
# or
pip install chatterbox-tts torch torchaudio
```

### torch/CUDA version mismatch

**Problem:** PyTorch CUDA version doesn't match your system CUDA.

**Solution:**
1. Check your CUDA version: `nvidia-smi`
2. Install matching PyTorch:
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Memory Issues

### CUDA out of memory

**Problem:** GPU runs out of memory during generation.

**Solutions:**
1. Use a smaller model:
   ```bash
   tts-toolkit pipeline --model Qwen/Qwen3-TTS-12Hz-0.6B-Base ...
   ```

2. Process in smaller chunks:
   ```bash
   tts-toolkit pipeline --chunk-max 150 --chunk-target 100 ...
   ```

3. Use CPU instead (slower but unlimited memory):
   ```bash
   tts-toolkit pipeline --device cpu ...
   ```

4. Clear GPU cache between operations:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### System memory exhausted

**Problem:** Large text files consume too much RAM.

**Solutions:**
1. Split large files into smaller pieces
2. Process files in batches:
   ```bash
   tts-toolkit batch --workers 1 --timeout 300 ...
   ```

3. Monitor memory usage:
   ```python
   from tts_toolkit.utils.memory import log_memory_stats
   log_memory_stats("cuda:0")
   ```

## Audio Quality Issues

### Robotic or choppy output

**Problem:** Generated audio sounds unnatural.

**Solutions:**
1. Use better reference audio (10-30 seconds, clear speech)
2. Increase temperature for more variation:
   ```bash
   tts-toolkit say "text" --temperature 0.95 ...
   ```
3. Use higher quality backend:
   ```bash
   tts-toolkit say "text" --backend qwen ...
   ```

### Silence or very short output

**Problem:** Generated audio is silent or much shorter than expected.

**Solutions:**
1. Check reference audio is valid:
   ```bash
   ffprobe reference.wav
   ```
2. Verify reference text matches the audio exactly
3. Try a different backend to isolate the issue

### Audio clipping or distortion

**Problem:** Audio peaks cause distortion.

**Solutions:**
1. Enable normalization (default):
   ```python
   stitcher.stitch_files(files, output, normalize=True)
   ```
2. Reduce crossfade if artifacts at transitions:
   ```bash
   tts-toolkit pipeline --crossfade-ms 50 ...
   ```

## API Backend Issues

### Fish Audio: API key errors

**Problem:** FishSpeechBackend fails with authentication error.

**Solution:**
```bash
export FISH_AUDIO_API_KEY="your-api-key"
# or in Python:
backend = FishSpeechBackend(api_key="your-api-key")
```

### Timeout errors

**Problem:** API calls time out.

**Solutions:**
1. Increase timeout:
   ```bash
   tts-toolkit batch --timeout 600 ...
   ```
2. Check network connectivity
3. Reduce text length per request

## Configuration Issues

### Config file not loading

**Problem:** Settings in config file are ignored.

**Solutions:**
1. Check file location:
   - Local: `.tts_toolkit.yml` or `.tts_toolkit.yaml`
   - Global: `~/.tts_toolkit/config.yml`

2. Validate YAML syntax:
   ```bash
   python -c "import yaml; yaml.safe_load(open('.tts_toolkit.yml'))"
   ```

3. Check for typos in key names:
   ```bash
   tts-toolkit config show
   ```

### Invalid config values

**Problem:** Config validation fails.

**Solution:** Check valid values:
- `backend`: qwen, chatterbox, kokoro, fish_speech, bark, cosyvoice, coqui_xtts, mock
- `temperature`: 0.0 - 2.0
- `top_k`: >= 1
- `top_p`: 0.0 - 1.0
- `output_format`: wav, mp3, flac, ogg

## Batch Processing Issues

### Jobs stuck or not progressing

**Problem:** Batch processing hangs.

**Solutions:**
1. Reduce workers to identify problematic files:
   ```bash
   tts-toolkit batch --workers 1 ...
   ```
2. Check individual files for issues
3. Add timeout to fail slow jobs:
   ```bash
   tts-toolkit batch --timeout 120 ...
   ```

### Resume not working

**Problem:** Batch doesn't resume from checkpoint.

**Solution:**
- Ensure same output directory is used
- Check checkpoint.json exists in work directory
- Don't modify source files between runs

## Debugging Tips

### Enable verbose logging

```bash
# CLI
tts-toolkit --verbose pipeline ...

# Python
import logging
logging.getLogger("tts-toolkit").setLevel(logging.DEBUG)
```

### Test with mock backend

```bash
# Verify CLI without actual TTS
tts-toolkit say "test" --backend mock -o test.wav
```

### Check system resources

```python
from tts_toolkit.utils.memory import get_system_memory, get_gpu_memory

print(get_system_memory())
print(get_gpu_memory(0))
```

## Getting Help

1. Check the [GitHub Issues](https://github.com/cyperx84/tts-toolkit/issues)
2. Search existing issues before creating new ones
3. Include in bug reports:
   - Python version: `python --version`
   - Package versions: `pip list | grep -E "tts|torch|qwen"`
   - Full error traceback
   - Minimal reproduction steps
