# TTS Toolkit

An extensible text-to-speech toolkit for creating podcasts, audiobooks, voiceovers, and dialogues.

## Features

- **Voice Cloning**: Clone any voice with 3-10 seconds of audio
- **Multi-Speaker**: Create podcasts and dialogues with multiple voices
- **Audiobooks**: Generate chapter-based audiobooks from Markdown
- **Background Music**: Add intro, outro, and background music
- **Checkpointing**: Resume long generation tasks if interrupted
- **Extensible Backends**: Swap TTS engines (Qwen3-TTS, more coming)
- **Claude Code Skills**: Built-in skills for seamless AI integration

## Installation

```bash
# Basic installation
pip install tts-toolkit

# With Qwen3-TTS backend (recommended)
pip install tts-toolkit[qwen]
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

| Backend | Description | Voice Cloning |
|---------|-------------|---------------|
| `QwenBackend` | Qwen3-TTS | Yes |
| `MockBackend` | Testing | No |

### Using Different Backends

```python
# Qwen backend (production)
from tts_toolkit.backends import QwenBackend
backend = QwenBackend(model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base", device="cuda:0")

# Mock backend (testing)
from tts_toolkit.backends import MockBackend
backend = MockBackend()

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
- [Anthropic](https://anthropic.com) for Claude Code integration
