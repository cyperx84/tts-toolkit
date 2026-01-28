# Getting Started with TTS Toolkit

TTS Toolkit is an extensible text-to-speech toolkit for creating podcasts, audiobooks, voiceovers, and dialogues.

## Installation

### Basic Installation

```bash
pip install tts-toolkit
```

### With Qwen3-TTS Backend

For voice cloning capabilities:

```bash
pip install tts-toolkit[qwen]
```

This installs the Qwen3-TTS backend with PyTorch support.

### Development Installation

```bash
git clone https://github.com/yourusername/tts-toolkit.git
cd tts-toolkit
pip install -e ".[qwen]"
```

## Quick Start

### 1. Create a Voice Profile

First, you need a reference audio sample (3-10 seconds of clear speech):

```bash
tts-toolkit voice create my-voice \
    --audio path/to/sample.wav \
    --text "Exact transcript of what's spoken in the audio"
```

### 2. Generate Speech

#### Quick TTS (short text)

```bash
tts-toolkit say "Hello, world!" --voice my-voice --output hello.wav
```

#### Voiceover (longer text)

```bash
tts-toolkit voiceover script.txt --voice my-voice --output narration.wav
```

#### Podcast (multiple speakers)

```bash
tts-toolkit podcast episode.txt \
    --host host-voice \
    --guest guest-voice \
    --output episode.wav
```

## Python API

```python
from tts_toolkit import Pipeline
from tts_toolkit.backends import QwenBackend

# Create a pipeline with the Qwen backend
backend = QwenBackend(device="cuda:0")  # or "cpu"
pipeline = Pipeline(backend=backend)

# Generate audio
pipeline.process(
    text="Your text here...",
    ref_audio="reference.wav",
    ref_text="Reference transcript",
    output_path="output.wav",
)
```

## Backends

TTS Toolkit supports multiple TTS backends:

| Backend | Description | Installation |
|---------|-------------|--------------|
| `QwenBackend` | Qwen3-TTS voice cloning | `pip install tts-toolkit[qwen]` |
| `MockBackend` | Testing without GPU | Built-in |

### Using Different Backends

```python
# Qwen backend (requires GPU or CPU with patience)
from tts_toolkit.backends import QwenBackend
backend = QwenBackend(
    model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device="cuda:0",
)

# Mock backend (for testing)
from tts_toolkit.backends import MockBackend
backend = MockBackend()
```

## Features

- **Voice Cloning**: Clone any voice with just 3-10 seconds of audio
- **Multi-Speaker**: Create podcasts and dialogues with multiple voices
- **Audiobooks**: Generate chapter-based audiobooks from Markdown
- **Background Music**: Add intro, outro, and background music
- **Checkpointing**: Resume long generation tasks if interrupted
- **Emotion Control**: Apply emotion presets to generated speech

## Next Steps

- [Voice Profiles](./voice-profiles.md) - Managing voice profiles
- [API Reference](./api-reference.md) - Detailed API documentation
- [Skills Guide](./skills-guide.md) - Using Claude Code skills
