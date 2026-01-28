# Voice Profile Skill

Manage voice profiles for TTS generation using TTS Toolkit.

## When to Use

Use this skill when the user wants to:
- Create reusable voice profiles from audio samples
- List or manage saved voice profiles
- Export voice profiles for sharing
- Import voice profiles from others

## Prerequisites

- TTS Toolkit installed
- For creation: Reference audio file (3-10 seconds of clear speech)

## Commands

### Create a Voice Profile

```bash
tts-toolkit voice create <name> \
    --audio <reference_audio.wav> \
    --text "Exact transcript of the audio" \
    --description "Optional description"
```

Example:
```bash
tts-toolkit voice create narrator \
    --audio ~/recordings/my_voice.wav \
    --text "Hello, this is a sample of my speaking voice for cloning purposes." \
    --description "Professional narrator voice"
```

### List Voice Profiles

```bash
tts-toolkit voice list
```

Output:
```
Voice profiles:
  - narrator: Professional narrator voice
  - podcast-host: Casual podcast host voice
  - character-alice: Female character voice
```

### Delete a Voice Profile

```bash
tts-toolkit voice delete <name>
```

### Export a Voice Profile

```bash
tts-toolkit voice export <name> --output <directory>
```

This creates a portable directory with:
- `profile.json` - Profile configuration
- `reference.wav` - Reference audio file

### Python API

```python
from tts_toolkit.voices.registry import VoiceRegistry
from tts_toolkit.voices.profile import VoiceProfile

# Initialize registry
registry = VoiceRegistry()

# Create a new profile
profile = registry.create(
    name="narrator",
    reference_audio="my_voice.wav",
    reference_text="Sample transcript",
    description="Professional narrator",
)

# List all profiles
for name in registry.list():
    print(name)

# Get profile details
profile = registry.get("narrator")
print(f"Name: {profile.name}")
print(f"Description: {profile.description}")
print(f"Audio: {profile.reference_audio}")

# Search profiles
podcasters = registry.search(use_case="podcast")

# Delete a profile
registry.delete("narrator")

# Export profile
registry.export("narrator", "./export_dir")

# Import profile
imported = registry.import_profile("./export_dir", name="new_narrator")
```

## Voice Profile Structure

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

### Profile JSON Schema

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

## Tips for Good Voice Profiles

### Reference Audio Quality

1. **Length**: 3-10 seconds is ideal
2. **Clarity**: Clear speech without background noise
3. **Consistency**: Single speaker, consistent tone
4. **Format**: WAV, MP3, or other standard formats

### Reference Text Accuracy

1. **Exact Match**: Text must exactly match spoken words
2. **Punctuation**: Include natural punctuation
3. **No Extras**: Don't include [sounds] or other annotations

### Good Example

Audio: Clear recording of "Hello and welcome to this presentation. Today we'll be exploring the fascinating world of artificial intelligence."

Text: "Hello and welcome to this presentation. Today we'll be exploring the fascinating world of artificial intelligence."

### Bad Example

Audio: Recording with background music and "um"s
Text: "[Music] Um, hello, welcome to, uh, this presentation..."

## Using Profiles

Once created, use profiles with other TTS Toolkit commands:

```bash
# Voiceover
tts-toolkit voiceover script.txt --voice narrator --output out.wav

# Podcast (multiple profiles)
tts-toolkit podcast script.txt --host podcast-host --guest expert --output episode.wav

# Audiobook
tts-toolkit audiobook book.md --voice narrator --output-dir ./audiobook

# Quick TTS
tts-toolkit say "Hello world" --voice narrator
```

## Metadata Fields

| Field | Description | Values |
|-------|-------------|--------|
| `gender` | Voice gender | male, female, neutral |
| `age_range` | Approximate age | 20-30, 30-40, 40-50, etc. |
| `accent` | Regional accent | american, british, neutral, etc. |
| `tone` | Voice character | warm, professional, casual, etc. |
| `use_cases` | Recommended uses | audiobook, podcast, tutorial, etc. |
