# TTS Toolkit

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://python.org)
[![Backends](https://img.shields.io/badge/TTS%20Backends-7-orange.svg)](#backends)

**Create professional voiceovers, podcasts, and audiobooks with state-of-the-art voice cloning.**

TTS Toolkit is an extensible text-to-speech framework that makes it easy to generate high-quality speech from text. Clone any voice with just 3-10 seconds of audio, create multi-speaker content, and integrate with your favorite TTS models.

---

## Why TTS Toolkit?

- **Simple CLI** - Generate speech from the command line in seconds
- **Voice Cloning** - Clone any voice with a short audio sample
- **Multiple Backends** - Choose from 7 state-of-the-art TTS engines
- **Production Ready** - Batch processing, checkpointing, and quality metrics
- **Extensible** - Easy to add new backends and formats

## Features

| Feature | Description |
|---------|-------------|
| Voice Cloning | Clone any voice with 3-10 seconds of audio |
| Multi-Speaker | Create podcasts and dialogues with multiple voices |
| Audiobooks | Generate chapter-based audiobooks from Markdown |
| Background Music | Add intro, outro, and background music |
| Checkpointing | Resume long generation tasks if interrupted |
| Batch Processing | Process multiple files in parallel |
| Quality Evaluation | MOS prediction, speaker similarity, WER metrics |
| Config Files | YAML configuration for consistent settings |

## Installation

```bash
# Basic installation
pip install tts-toolkit

# With a specific backend
pip install tts-toolkit[qwen]        # Qwen3-TTS (voice cloning)
pip install tts-toolkit[chatterbox]  # Chatterbox (emotion control)
pip install tts-toolkit[kokoro]      # Kokoro (fast, lightweight)
pip install tts-toolkit[bark]        # Bark (expressive sounds)
pip install tts-toolkit[coqui]       # Coqui XTTS (17 languages)
pip install tts-toolkit[fish-speech] # Fish Speech (API-based)

# All backends
pip install tts-toolkit[all-backends]
```

## Quick Start

### 1. Create a Voice Profile

```bash
tts-toolkit voice create my-voice \
    --audio path/to/sample.wav \
    --text "Exact transcript of the audio"
```

### 2. Generate Speech

```bash
# Quick TTS
tts-toolkit say "Hello, world!" --voice my-voice

# Voiceover from script
tts-toolkit voiceover script.txt --voice my-voice --output narration.wav

# Podcast with multiple speakers
tts-toolkit podcast episode.txt --host host-voice --guest guest-voice --output episode.wav

# Audiobook from Markdown
tts-toolkit audiobook book.md --voice narrator --output-dir ./audiobook
```

### 3. Python API

```python
from tts_toolkit import Pipeline
from tts_toolkit.backends import QwenBackend

# Create pipeline with your preferred backend
pipeline = Pipeline(backend=QwenBackend(device="cuda:0"))

# Generate audio
pipeline.process(
    text="Your text here...",
    ref_audio="reference.wav",
    ref_text="Reference transcript",
    output_path="output.wav",
)
```

## Backends

TTS Toolkit supports 7 production-ready backends, each with unique strengths:

| Backend | Description | Languages | Best For |
|---------|-------------|-----------|----------|
| **Qwen** | Qwen3-TTS | 11 | High-quality voice cloning |
| **Chatterbox** | Resemble AI | 23 | Emotion control `[laugh]`, `[sigh]` |
| **Kokoro** | Lightweight 82M | 5 | Fast generation, Apache 2.0 |
| **Fish Speech** | API-based | 8 | Cloud deployment |
| **Bark** | Suno AI | 9+ | Non-verbal sounds, music |
| **CosyVoice2** | Alibaba | 5+ | Low latency, Chinese dialects |
| **Coqui XTTS** | XTTS v2 | 17 | Multilingual projects |

### Switching Backends

```python
# Qwen - best quality
from tts_toolkit.backends import QwenBackend
backend = QwenBackend(device="cuda:0")

# Chatterbox - emotion control
from tts_toolkit.backends import ChatterboxBackend
backend = ChatterboxBackend(device="cuda")
# Use: "Hello [laugh] that's funny!"

# Kokoro - fast and lightweight
from tts_toolkit.backends import KokoroBackend
backend = KokoroBackend(voice="af_heart")

# Bark - expressive sounds
from tts_toolkit.backends import BarkBackend
backend = BarkBackend(device="cuda")
# Use: "[laughter] Oh wow! [sighs]"

# Coqui XTTS - 17 languages
from tts_toolkit.backends import CoquiXTTSBackend
backend = CoquiXTTSBackend(device="cuda")

# Fish Speech - API-based
from tts_toolkit.backends import FishSpeechBackend
backend = FishSpeechBackend(api_key="your-key")
```

## Script Formats

### Dialogue / Podcast

```
[HOST]: Welcome to the show!
[GUEST]: Thanks for having me.
[HOST]: Let's dive right in.
```

### Audiobook (Markdown)

```markdown
# Chapter 1: The Beginning

Your chapter content here...

# Chapter 2: The Journey

More content...
```

## Format Handlers

| Handler | Use Case |
|---------|----------|
| `VoiceoverHandler` | Video narration |
| `PodcastHandler` | Multi-speaker podcasts |
| `AudiobookHandler` | Chapter-based audiobooks |
| `DialogueHandler` | Two-person conversations |

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Installation and first steps |
| [API Reference](docs/api-reference.md) | Complete API documentation |
| [Voice Profiles](docs/voice-profiles.md) | Managing voice profiles |
| [Skills Guide](docs/skills-guide.md) | Claude Code integration |
| [Evaluation Guide](docs/evaluation.md) | Quality metrics and benchmarking |
| [Advanced Features](docs/advanced-features.md) | Batch processing, config files |

## Examples

See the [examples](examples/) directory:

- `simple_voiceover.py` - Basic voiceover generation
- `podcast_two_speakers.py` - Two-speaker podcast
- `audiobook_from_markdown.py` - Audiobook with chapters
- `dialogue_generation.py` - Two-person dialogue

## Contributing

Contributions are welcome! Whether it's bug fixes, new features, or documentation improvements, we'd love your help.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Setting up your development environment
- Code style and testing
- Adding new TTS backends
- Submitting pull requests

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

TTS Toolkit is built on the shoulders of giants:

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) - High-quality TTS model
- [Chatterbox](https://github.com/resemble-ai/chatterbox) - Emotion-aware TTS by Resemble AI
- [Kokoro](https://github.com/hexgrad/kokoro) - Lightweight, fast TTS
- [Fish Speech](https://fish.audio) - API-based voice cloning
- [Bark](https://github.com/suno-ai/bark) - Expressive speech by Suno AI
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - Streaming TTS by Alibaba
- [Coqui TTS](https://github.com/coqui-ai/TTS) - XTTS v2 model

---

<p align="center">
  <strong>Made for creators, by creators.</strong><br>
  <a href="https://github.com/cyperx84/tts-toolkit/issues">Report Bug</a> ·
  <a href="https://github.com/cyperx84/tts-toolkit/issues">Request Feature</a>
</p>
