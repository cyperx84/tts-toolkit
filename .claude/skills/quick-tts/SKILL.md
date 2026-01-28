# Quick TTS Skill

Fast single-utterance text-to-speech using TTS Toolkit.

## When to Use

Use this skill when the user wants to:
- Quickly generate speech from short text
- Test voice settings
- Create notification sounds
- Generate single audio clips

## Prerequisites

- TTS Toolkit installed
- Voice profile OR reference audio

## Usage

### Basic Usage

```bash
tts-toolkit say "Hello, world!" \
    --output hello.wav \
    --voice narrator
```

### With Reference Audio

```bash
tts-toolkit say "Hello, world!" \
    --output hello.wav \
    --ref-audio sample.wav \
    --ref-text "Sample transcript"
```

### With Emotion

```bash
tts-toolkit say "I'm so excited!" \
    --output excited.wav \
    --voice narrator \
    --emotion excited
```

### Python API

```python
from tts_toolkit.backends import QwenBackend
from tts_toolkit.voices.emotions import apply_emotion
import soundfile as sf

# Setup with backend
backend = QwenBackend(device="cpu")
backend.load_model()

# Create voice prompt
voice_prompt = backend.create_voice_prompt(
    reference_audio="sample.wav",
    reference_text="Sample transcript"
)

# Generate with emotion
gen_kwargs = apply_emotion({"temperature": 0.9}, "happy")

audio, sr = backend.generate(
    text="Hello, this is a quick test!",
    voice_prompt=voice_prompt,
    language="English",
    **gen_kwargs,
)

sf.write("output.wav", audio, sr)
```

## Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `text` | Text to speak | Yes |
| `--output, -o` | Output file (default: output.wav) | No |
| `--voice, -v` | Voice profile name | No* |
| `--ref-audio, -a` | Reference audio file | No* |
| `--ref-text, -r` | Reference audio transcript | No* |
| `--emotion, -e` | Emotion preset | No |
| `--language, -l` | Language code | No |
| `--backend` | Backend to use (qwen, mock) | No |
| `--model, -m` | Model name | No |
| `--device, -d` | Device (cpu, cuda:0, mps) | No |

*Either `--voice` OR (`--ref-audio` AND `--ref-text`) is required.

## Available Emotions

| Emotion | Description | Best For |
|---------|-------------|----------|
| `neutral` | Natural tone | General use |
| `happy` | Cheerful, upbeat | Positive content |
| `sad` | Melancholic | Somber content |
| `excited` | Energetic | Announcements |
| `serious` | Formal | Professional content |
| `calm` | Relaxed | Meditation, ASMR |
| `angry` | Intense | Drama |
| `whisper` | Quiet | Intimate content |
| `narrator` | Storytelling | Narration |
| `news` | Authoritative | News-style |
| `conversational` | Casual | Chat-style |

## Examples

### Notification Sounds

```bash
# Success
tts-toolkit say "Task completed successfully!" \
    --voice narrator --emotion happy -o success.wav

# Error
tts-toolkit say "An error has occurred." \
    --voice narrator --emotion serious -o error.wav

# Reminder
tts-toolkit say "Don't forget your meeting at 3 PM." \
    --voice narrator -o reminder.wav
```

### Testing Voices

```bash
# Test different emotions
for emotion in neutral happy sad excited serious calm; do
    tts-toolkit say "Testing $emotion emotion" \
        --voice narrator --emotion $emotion -o test_$emotion.wav
done
```

### Multiple Languages

```bash
# English
tts-toolkit say "Hello, how are you?" \
    --voice narrator --language English -o hello_en.wav

# Chinese
tts-toolkit say "你好，你好吗？" \
    --voice narrator --language Chinese -o hello_zh.wav
```

## Tips

1. **Short Text Only**: Best for text under 500 characters
2. **Longer Text**: Use `voiceover` or `audiobook` for longer content
3. **First Run**: Model loading takes time; subsequent calls are faster
4. **Voice Profiles**: Create profiles for frequently used voices

## Performance

- Model loading: ~10-30 seconds (first time only)
- Generation: ~1-5 seconds per sentence (CPU)
- GPU: 5-10x faster generation

## Common Patterns

### Quick Test Script

```bash
#!/bin/bash
# test_voice.sh

TEXT="${1:-Hello, this is a test.}"
VOICE="${2:-narrator}"
OUTPUT="${3:-test.wav}"

tts-toolkit say "$TEXT" --voice "$VOICE" -o "$OUTPUT"
echo "Generated: $OUTPUT"
```

### Batch Generation

```python
from tts_toolkit.backends import QwenBackend
import soundfile as sf

backend = QwenBackend()
backend.load_model()

voice_prompt = backend.create_voice_prompt("sample.wav", "Sample text")

phrases = [
    "Welcome to our application.",
    "Please wait while we process your request.",
    "Thank you for your patience.",
    "Goodbye!",
]

for i, phrase in enumerate(phrases):
    audio, sr = backend.generate(phrase, voice_prompt=voice_prompt, language="English")
    sf.write(f"phrase_{i:02d}.wav", audio, sr)
```
