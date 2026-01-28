# Advanced Features

This guide covers advanced TTS Toolkit features for production workflows.

## Configuration Files

TTS Toolkit supports YAML configuration files for consistent settings.

### File Locations (Priority Order)

1. `.tts_toolkit.yml` - Project-local config
2. `~/.tts_toolkit/config.yml` - Global user config

### Initialize Config

```bash
# Create local config
tts-toolkit config init

# Create global config
tts-toolkit config init --global
```

### Example Config File

```yaml
# .tts_toolkit.yml
backend: qwen
device: cuda:0
language: English
temperature: 0.9

# Chunking settings
chunk_min: 100
chunk_max: 300
chunk_target: 200
crossfade_ms: 75

# Default voice
default_voice: narrator

# Custom voices mapping
voices:
  narrator: ~/.tts_toolkit/voices/narrator
  host: ~/.tts_toolkit/voices/host

# Batch processing
batch_workers: 2
batch_timeout: 300

# Memory limits
max_text_chars: 100000
warn_vram_threshold_mb: 1000
```

### Config Commands

```bash
# Show current config
tts-toolkit config show

# Set a value
tts-toolkit config set backend chatterbox
tts-toolkit config set device cuda:0 --global
```

### Using Config in Python

```python
from tts_toolkit.utils import load_config, TTSConfig

# Load merged config (local + global)
config = load_config()
print(f"Backend: {config.backend}")
print(f"Device: {config.device}")

# Create custom config
config = TTSConfig(
    backend="kokoro",
    temperature=0.7,
)
```

## Batch Processing

Process multiple files efficiently with parallel workers.

### CLI Usage

```bash
# Process directory of text files
tts-toolkit batch \
    --input-dir ./scripts \
    --output-dir ./audio \
    --ref-audio voice.wav \
    --ref-text "Reference transcript" \
    --workers 2 \
    --report batch_report.json

# Process from manifest file
tts-toolkit batch \
    --manifest jobs.json \
    --output-dir ./audio \
    --workers 4
```

### Manifest File Format

```json
{
    "defaults": {
        "ref_audio": "voice.wav",
        "ref_text": "Reference transcript",
        "language": "English"
    },
    "jobs": [
        {
            "input": "chapter1.txt",
            "output": "chapter1.wav"
        },
        {
            "input": "chapter2.txt",
            "output": "chapter2.wav",
            "language": "Chinese"
        },
        {
            "text": "Direct text input",
            "output": "direct.wav"
        }
    ]
}
```

### Python API

```python
from tts_toolkit.utils import BatchProcessor, BatchJob, create_jobs_from_directory
from tts_toolkit.backends import QwenBackend

# Create jobs from directory
jobs = create_jobs_from_directory(
    input_dir="./scripts",
    output_dir="./audio",
    ref_audio="voice.wav",
    ref_text="Reference",
    pattern="*.txt",
)

# Or create manually
jobs = [
    BatchJob(
        input_path="script1.txt",
        output_path="output1.wav",
        ref_audio="voice.wav",
        ref_text="Reference",
    ),
    BatchJob(
        input_path="script2.txt",
        output_path="output2.wav",
        ref_audio="voice.wav",
        ref_text="Reference",
    ),
]

# Process
backend = QwenBackend(device="cuda:0")
processor = BatchProcessor(backend=backend, workers=2)

summary = processor.process(
    jobs,
    progress_callback=lambda c, t, r: print(f"{c}/{t}: {'✓' if r.success else '✗'}"),
)

print(f"Success rate: {summary.success_rate:.1f}%")
print(f"Total duration: {summary.total_duration_sec / 60:.1f} minutes")
```

## Parallel Chunk Processing

For long-form TTS, process chunks in parallel (best for CPU/API backends).

```python
from tts_toolkit import Pipeline
from tts_toolkit.backends import FishSpeechBackend

# API backends benefit most from parallel processing
backend = FishSpeechBackend(api_key="...")
pipeline = Pipeline(backend=backend)

# Process with 4 parallel workers
pipeline.process_parallel(
    text=long_text,
    ref_audio="voice.wav",
    ref_text="Reference",
    output_path="output.wav",
    workers=4,
)
```

**Note**: GPU backends may not benefit from parallel processing due to memory constraints.

## Memory Management

Monitor and manage memory usage for large workloads.

### Check Memory

```python
from tts_toolkit.utils import (
    get_system_memory,
    get_gpu_memory,
    estimate_model_memory,
    check_memory_available,
)

# System memory
mem = get_system_memory()
print(f"Available: {mem.available_gb:.1f} GB")

# GPU memory
gpu = get_gpu_memory(device_id=0)
if gpu:
    print(f"GPU {gpu.name}: {gpu.free_gb:.1f} GB free")

# Estimate requirements
model_mb = estimate_model_memory("qwen")
available, msg = check_memory_available(model_mb, use_gpu=True)
print(msg)
```

### Memory Warnings

```python
from tts_toolkit.utils import warn_if_low_memory

# Warn before processing large text
warn_if_low_memory(
    backend_name="qwen",
    text_length=len(long_text),
    device="cuda:0",
    threshold_mb=1000,
)
```

### Clear GPU Cache

```python
from tts_toolkit.utils import clear_gpu_cache

# After processing, free GPU memory
clear_gpu_cache()
```

## Audio Metadata Embedding

Embed metadata in generated audio files.

### Supported Formats

| Format | Metadata Type | Fields |
|--------|--------------|--------|
| WAV | INFO chunks | title, artist, comment, date |
| MP3 | ID3 tags | All standard ID3 fields + cover art |

### Python API

```python
from tts_toolkit.utils import embed_metadata, create_tts_metadata

# Create metadata
metadata = create_tts_metadata(
    text="Chapter 1: The Beginning",
    backend_name="qwen",
    voice_name="narrator",
    language="English",
    duration_sec=120.5,
)

# Embed in audio file
embed_metadata("output.mp3", metadata, cover_image="cover.jpg")

# Custom metadata
embed_metadata("output.wav", {
    "title": "Chapter 1",
    "artist": "AI Narrator",
    "album": "My Audiobook",
    "genre": "Audiobook",
    "date": "2024",
})
```

### Reading Metadata

```python
from tts_toolkit.utils import read_metadata

metadata = read_metadata("audiobook.mp3")
print(f"Title: {metadata.get('title')}")
print(f"Duration: {metadata.get('duration'):.1f}s")
```

## Pipeline with All Features

Example combining multiple advanced features:

```python
from tts_toolkit import Pipeline
from tts_toolkit.backends import QwenBackend
from tts_toolkit.utils import (
    load_config,
    warn_if_low_memory,
    embed_metadata,
    create_tts_metadata,
    clear_gpu_cache,
)

# Load config
config = load_config()

# Check memory before starting
warn_if_low_memory(
    backend_name=config.backend,
    text_length=len(text),
    device=config.device,
)

# Create backend and pipeline
backend = QwenBackend(device=config.device)
pipeline = Pipeline(
    backend=backend,
    chunk_min=config.chunk_min,
    chunk_max=config.chunk_max,
    crossfade_ms=config.crossfade_ms,
)

# Process
output_path = pipeline.process(
    text=text,
    ref_audio=ref_audio,
    ref_text=ref_text,
    output_path="output.wav",
)

# Embed metadata
duration_sec = ...  # Get from audio file
metadata = create_tts_metadata(
    text=text,
    backend_name=config.backend,
    duration_sec=duration_sec,
)
embed_metadata(output_path, metadata)

# Cleanup
clear_gpu_cache()
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `FISH_AUDIO_API_KEY` | Fish Audio API key |
| `TTS_TOOLKIT_CONFIG` | Custom config file path |
| `TTS_TOOLKIT_CACHE` | Cache directory |
| `CUDA_VISIBLE_DEVICES` | GPU device selection |

## CLI Reference

### Batch Command

```
tts-toolkit batch [OPTIONS]

Options:
  -i, --input-dir PATH    Input directory with text files
  --manifest PATH         JSON manifest file
  -o, --output-dir PATH   Output directory (required)
  -a, --ref-audio PATH    Reference audio file
  -r, --ref-text TEXT     Reference transcript
  -v, --voice NAME        Voice profile name
  --pattern GLOB          File pattern (default: *.txt)
  -w, --workers INT       Parallel workers (default: 1)
  --timeout INT           Timeout per job in seconds
  --report PATH           Save JSON report
  --backend NAME          TTS backend
  --device NAME           Device (cpu, cuda:0, etc.)
```

### Config Command

```
tts-toolkit config {init,show,set}

Subcommands:
  init              Create new config file
    --global, -g    Create global config

  show              Display current config

  set KEY VALUE     Set config value
    --global, -g    Set in global config
```
