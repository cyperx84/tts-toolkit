# Voice Profiles

Voice profiles allow you to save and reuse voice configurations for consistent TTS generation.

## Creating a Voice Profile

### CLI

```bash
tts-toolkit voice create <name> \
    --audio <reference.wav> \
    --text "Exact transcript of the audio" \
    --description "Optional description"
```

### Python

```python
from tts_toolkit.voices import VoiceRegistry

registry = VoiceRegistry()
profile = registry.create(
    name="narrator",
    reference_audio="my_voice.wav",
    reference_text="This is a sample of my speaking voice.",
    description="Professional narrator voice",
)
```

## Reference Audio Requirements

For best results, your reference audio should:

1. **Length**: 3-10 seconds of continuous speech
2. **Quality**: Clear audio without background noise
3. **Content**: Natural speaking (not singing or whispering)
4. **Format**: WAV, MP3, or other common audio formats

## Reference Text Requirements

The transcript must:

1. **Exactly match** the spoken words
2. Include natural punctuation
3. NOT include:
   - [sound effects]
   - Timestamps
   - Speaker labels
   - Annotations

### Good Example

```
Audio: Clear recording saying "Hello and welcome. Today we'll explore the world of AI."
Text: "Hello and welcome. Today we'll explore the world of AI."
```

### Bad Example

```
Audio: Recording with background music and "um"s
Text: "[Music] Um, hello, uh, welcome to, um, this presentation..."
```

## Using Voice Profiles

Once created, use profiles by name:

```bash
# Voiceover
tts-toolkit voiceover script.txt --voice narrator --output out.wav

# Podcast
tts-toolkit podcast script.txt --host host-voice --guest guest-voice

# Audiobook
tts-toolkit audiobook book.md --voice narrator --output-dir ./audiobook

# Quick TTS
tts-toolkit say "Hello" --voice narrator
```

## Managing Profiles

### List Profiles

```bash
tts-toolkit voice list
```

### Delete Profile

```bash
tts-toolkit voice delete <name>
```

### Export Profile

```bash
tts-toolkit voice export <name> --output ./export_dir
```

This creates a portable directory with:
- `profile.json` - Configuration
- `reference.wav` - Audio file

### Import Profile

```python
from tts_toolkit.voices import VoiceRegistry

registry = VoiceRegistry()
registry.import_profile("./export_dir", name="imported_voice")
```

## Profile Storage

Profiles are stored in `~/.tts_toolkit/voices/`:

```
~/.tts_toolkit/
└── voices/
    ├── registry.json
    ├── narrator/
    │   ├── profile.json
    │   └── reference.wav
    └── podcast-host/
        ├── profile.json
        └── reference.wav
```

## Profile Schema

```json
{
  "id": "uuid",
  "name": "narrator",
  "description": "Professional narrator voice",
  "created_at": "2024-01-28T12:00:00Z",
  "reference_audio": "reference.wav",
  "reference_text": "Sample transcript...",
  "metadata": {
    "gender": "female",
    "age_range": "30-40",
    "accent": "american",
    "tone": "warm",
    "use_cases": ["audiobook", "documentary"]
  },
  "generation_params": {
    "temperature": 0.85,
    "top_k": 50
  },
  "emotion_presets": {
    "neutral": {},
    "happy": {"temperature": 0.95},
    "serious": {"temperature": 0.75}
  }
}
```

## Metadata Fields

| Field | Description | Examples |
|-------|-------------|----------|
| `gender` | Voice gender | male, female, neutral |
| `age_range` | Approximate age | 20-30, 30-40, 40-50 |
| `accent` | Regional accent | american, british, neutral |
| `tone` | Voice character | warm, professional, casual |
| `use_cases` | Recommended uses | audiobook, podcast, tutorial |

## Tips

1. **Consistency**: Record reference audio in the same environment you'll use for content
2. **Multiple Profiles**: Create different profiles for different use cases
3. **Emotion Presets**: Customize generation parameters for different moods
4. **Backup**: Export important profiles for backup and sharing
