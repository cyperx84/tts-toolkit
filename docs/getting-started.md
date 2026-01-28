# Getting Started with TTS Toolkit

TTS Toolkit is an extensible text-to-speech toolkit for creating podcasts, audiobooks, voiceovers, and dialogues.

## Installation

TTS Toolkit can be installed using either **UV** (recommended) or **pip**.

### Using UV (Recommended)

[UV](https://docs.astral.sh/uv/) is a fast Python package manager written in Rust that provides 10-100x faster installs. It's especially beneficial for TTS Toolkit due to the large ML dependencies (PyTorch, transformers, etc.).

#### Install UV

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv

# Or via Homebrew (macOS)
brew install uv
```

#### Basic Installation with UV

```bash
# Create a virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install tts-toolkit
```

#### Choose Your Backend with UV

```bash
# Voice cloning with emotion control (recommended)
uv pip install tts-toolkit[qwen]        # Qwen3-TTS - 11 languages, streaming
uv pip install tts-toolkit[chatterbox]  # Chatterbox - emotion tags, 23 languages

# Fast and lightweight
uv pip install tts-toolkit[kokoro]      # Kokoro - 82M params, Apache 2.0 license

# Expressive with non-verbal sounds
uv pip install tts-toolkit[bark]        # Bark - [laughter], [sighs], ♪ singing

# Maximum language support
uv pip install tts-toolkit[coqui]       # Coqui XTTS - 17 languages, 6s cloning

# API-based (no GPU required)
uv pip install tts-toolkit[fish-speech] # Fish Speech - cloud API

# Install all backends
uv pip install tts-toolkit[all-backends]
```

#### Using `uv run` (No Activation Required)

UV can run commands directly without manually activating the virtual environment:

```bash
# Create venv and install in one step
uv venv && uv pip install tts-toolkit[qwen]

# Run TTS Toolkit directly
uv run tts-toolkit --help
uv run tts-toolkit say "Hello, world!" --voice my-voice
```

#### Development Installation with UV

```bash
git clone https://github.com/cyperx84/tts-toolkit.git
cd tts-toolkit

# Create venv and install in development mode
uv venv
uv pip install -e ".[dev,all-backends]"

# Or use uv sync for reproducible installs (if uv.lock exists)
uv sync --all-extras
```

---

### Using pip

#### Basic Installation

```bash
pip install tts-toolkit
```

#### Choose Your Backend

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

#### Development Installation

```bash
git clone https://github.com/cyperx84/tts-toolkit.git
cd tts-toolkit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev,all-backends]"
```

---

### CosyVoice2 Installation (Manual)

CosyVoice2 requires manual installation from GitHub:

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
pip install -r requirements.txt  # Or: uv pip install -r requirements.txt
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
