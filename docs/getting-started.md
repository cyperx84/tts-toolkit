# Getting Started with TTS Toolkit

TTS Toolkit is an extensible text-to-speech toolkit for creating podcasts, audiobooks, voiceovers, and dialogues.

## Installation

### Basic Installation

```bash
pip install tts-toolkit
```

### Choose Your Backend

TTS Toolkit supports multiple backends. Choose based on your needs:

```bash
# Voice cloning with emotion control (recommended)
pip install tts-toolkit[qwen]        # Qwen3-TTS - 11 languages, streaming
pip install tts-toolkit[chatterbox]  # Chatterbox - emotion tags, 23 languages

# Fast and lightweight
pip install tts-toolkit[kokoro]      # Kokoro - 82M params, Apache 2.0 license

# Expressive with non-verbal sounds
pip install tts-toolkit[bark]        # Bark - [laughter], [sighs], ♪ singing

# Maximum language support
pip install tts-toolkit[coqui]       # Coqui XTTS - 17 languages, 6s cloning

# API-based (no GPU required)
pip install tts-toolkit[fish-speech] # Fish Speech - cloud API

# Install all backends
pip install tts-toolkit[all-backends]
```

### CosyVoice2 Installation (Manual)

CosyVoice2 requires manual installation from GitHub:

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
pip install -r requirements.txt
```

### Development Installation

```bash
git clone https://github.com/yourusername/tts-toolkit.git
cd tts-toolkit
pip install -e ".[all-backends]"
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

| Backend | Best For | Installation | Voice Cloning |
|---------|----------|--------------|---------------|
| `QwenBackend` | General purpose, streaming | `tts-toolkit[qwen]` | ✅ 3-10s audio |
| `ChatterboxBackend` | Emotional speech, sound effects | `tts-toolkit[chatterbox]` | ✅ 3-10s audio |
| `KokoroBackend` | Fast inference, low resources | `tts-toolkit[kokoro]` | ✅ Short clips |
| `BarkBackend` | Expressive, non-verbal sounds | `tts-toolkit[bark]` | ✅ Voice presets |
| `CoquiXTTSBackend` | Multi-language, streaming | `tts-toolkit[coqui]` | ✅ 6s audio |
| `FishSpeechBackend` | No GPU, API-based | `tts-toolkit[fish-speech]` | ✅ 10-30s audio |
| `CosyVoice2Backend` | Low latency, Chinese dialects | Manual install | ✅ Short clips |
| `MockBackend` | Testing, CI/CD | Built-in | ❌ |

### Using Different Backends

```python
# Qwen backend (recommended for production)
from tts_toolkit.backends import QwenBackend
backend = QwenBackend(
    model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device="cuda:0",
)

# Chatterbox with emotion control
from tts_toolkit.backends import ChatterboxBackend
backend = ChatterboxBackend(device="cuda")
# Use paralinguistic tags: [laugh], [chuckle], [sigh], [gasp]

# Kokoro for fast, lightweight generation
from tts_toolkit.backends import KokoroBackend
backend = KokoroBackend(voice="af_heart")  # American female
# Voices: af_heart, af_bella, am_adam, am_michael, bf_emma, etc.

# Bark for expressive speech
from tts_toolkit.backends import BarkBackend
backend = BarkBackend(device="cuda")
# Use: [laughter], [sighs], [music], [gasps], ♪ for singing

# Coqui XTTS for multilingual
from tts_toolkit.backends import CoquiXTTSBackend
backend = CoquiXTTSBackend(device="cuda")
# Supports: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi

# Fish Speech (API-based, no GPU needed)
from tts_toolkit.backends import FishSpeechBackend
backend = FishSpeechBackend(api_key="your-fish-audio-key")

# CosyVoice2 for ultra-low latency
from tts_toolkit.backends import CosyVoice2Backend
backend = CosyVoice2Backend(model_dir="pretrained_models/CosyVoice2-0.5B")

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
