# TTS Toolkit

An extensible text-to-speech toolkit for creating podcasts, audiobooks, voiceovers, and dialogues.

## Features

- **Voice Cloning**: Clone any voice with 3-10 seconds of audio
- **Multi-Speaker**: Create podcasts and dialogues with multiple voices
- **Audiobooks**: Generate chapter-based audiobooks from Markdown
- **Background Music**: Add intro, outro, and background music
- **Checkpointing**: Resume long generation tasks if interrupted
- **8 TTS Backends**: Qwen, Chatterbox, Kokoro, Bark, Fish Speech, CosyVoice2, Coqui XTTS
- **Batch Processing**: Process multiple files in parallel
- **Quality Evaluation**: MOS prediction, speaker similarity, WER metrics
- **Config Files**: YAML configuration for consistent settings
- **Claude Code Skills**: Built-in skills for seamless AI integration

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

### Create a Voice Profile

```bash
tts-toolkit voice create my-voice \
    --audio path/to/sample.wav \
    --text "Exact transcript of the audio"
```

### Generate Speech

```bash
# Quick TTS
tts-toolkit say "Hello, world!" --voice my-voice

# Voiceover
tts-toolkit voiceover script.txt --voice my-voice --output narration.wav

# Podcast
tts-toolkit podcast episode.txt --host host-voice --guest guest-voice --output episode.wav

# Audiobook
tts-toolkit audiobook book.md --voice narrator --output-dir ./audiobook
```

## Python API

```python
from tts_toolkit import Pipeline
from tts_toolkit.backends import QwenBackend

# Create pipeline with Qwen backend
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

| Backend | Description | Voice Cloning | Languages | Key Features |
|---------|-------------|---------------|-----------|--------------|
| `QwenBackend` | Qwen3-TTS | ✅ | 11 | Streaming, emotion control |
| `ChatterboxBackend` | Resemble AI Chatterbox | ✅ | 23 | Emotion tags `[laugh]`, `[sigh]` |
| `KokoroBackend` | Kokoro TTS | ✅ | 5 | Fast, 82M params, Apache 2.0 |
| `FishSpeechBackend` | Fish Audio API | ✅ | 8 | API-based, DualAR architecture |
| `BarkBackend` | Suno Bark | ✅ | 9+ | Non-verbal sounds, music |
| `CosyVoice2Backend` | Alibaba CosyVoice2 | ✅ | 5+ | 150ms latency, Chinese dialects |
| `CoquiXTTSBackend` | Coqui XTTS v2 | ✅ | 17 | 6s voice cloning, streaming |
| `MockBackend` | Testing | ❌ | - | No dependencies |

### Using Different Backends

```python
# Qwen backend (production)
from tts_toolkit.backends import QwenBackend
backend = QwenBackend(model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base", device="cuda:0")

# Chatterbox with emotion control
from tts_toolkit.backends import ChatterboxBackend
backend = ChatterboxBackend(device="cuda")
# Supports: [laugh], [chuckle], [sigh], [gasp], [clears throat]

# Kokoro for fast, lightweight TTS
from tts_toolkit.backends import KokoroBackend
backend = KokoroBackend(voice="af_heart")  # American female voice

# Bark for expressive speech with sounds
from tts_toolkit.backends import BarkBackend
backend = BarkBackend(device="cuda")
# Supports: [laughter], [sighs], [music], [gasps], ♪ for singing

# Coqui XTTS for multilingual
from tts_toolkit.backends import CoquiXTTSBackend
backend = CoquiXTTSBackend(device="cuda")

# Fish Speech (API-based)
from tts_toolkit.backends import FishSpeechBackend
backend = FishSpeechBackend(api_key="your-key")

# Use with any format handler
from tts_toolkit.formats import VoiceoverHandler
handler = VoiceoverHandler(backend=backend)
```

## Format Handlers

| Handler | Use Case |
|---------|----------|
| `VoiceoverHandler` | Video narration |
| `PodcastHandler` | Multi-speaker podcasts |
| `AudiobookHandler` | Chapter-based audiobooks |
| `DialogueHandler` | Two-person conversations |

## Script Formats

### Dialogue/Podcast

```
[HOST]: Welcome to the show!
[GUEST]: Thanks for having me.
```

### Audiobook (Markdown)

```markdown
# Chapter 1: The Beginning

Your chapter content here...

# Chapter 2: The Journey

More content...
```

## Claude Code Skills

When using Claude Code in this project, these skills are available:

- `voiceover` - Generate video voiceovers
- `podcast` - Create multi-speaker podcasts
- `audiobook` - Generate audiobooks with chapters
- `dialogue` - Create two-person conversations
- `voice-profile` - Manage voice profiles
- `quick-tts` - Fast single-utterance TTS

## Documentation

- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [Voice Profiles](docs/voice-profiles.md)
- [Skills Guide](docs/skills-guide.md)
- [Evaluation Guide](docs/evaluation.md) - Quality metrics and benchmarking
- [Advanced Features](docs/advanced-features.md) - Batch processing, config files, metadata

## Examples

See the [examples](examples/) directory for complete usage examples:

- `simple_voiceover.py` - Basic voiceover generation
- `podcast_two_speakers.py` - Two-speaker podcast
- `audiobook_from_markdown.py` - Audiobook with chapters
- `dialogue_generation.py` - Two-person dialogue

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for the amazing TTS model
- [Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI for emotion-aware TTS
- [Kokoro](https://github.com/hexgrad/kokoro) for lightweight, fast TTS
- [Fish Speech](https://fish.audio) for API-based voice cloning
- [Bark](https://github.com/suno-ai/bark) by Suno AI for expressive speech
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) by Alibaba for streaming TTS
- [Coqui TTS](https://github.com/coqui-ai/TTS) for the XTTS v2 model
- [Anthropic](https://anthropic.com) for Claude Code integration
