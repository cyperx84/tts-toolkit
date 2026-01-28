# Dialogue Skill

Generate realistic two-person conversations using TTS Toolkit.

## When to Use

Use this skill when the user wants to:
- Create dialogue audio between two speakers
- Generate interview-style conversations
- Produce dramatic readings with multiple characters
- Convert script-format text to audio

## Prerequisites

- TTS Toolkit installed
- Reference audio for each speaker (3-10 seconds each)
- Script with speaker markup

## Script Formats

### Dia-Style (recommended)
```
[S1]: Hello, how are you today?
[S2]: I'm doing great, thanks for asking!
[S1]: That's wonderful to hear.
```

### Bracket Style
```
[ALICE]: Hello, how are you today?
[BOB]: I'm doing great, thanks for asking!
[ALICE]: That's wonderful to hear.
```

### Markdown Bold
```
**Alice**: Hello, how are you today?
**Bob**: I'm doing great, thanks for asking!
**Alice**: That's wonderful to hear.
```

### Screenplay Style
```
ALICE: Hello, how are you today?
BOB: I'm doing great, thanks for asking!
ALICE: That's wonderful to hear.
```

### With Emotions
```
[S1 (happy)]: I just got promoted!
[S2 (excited)]: That's amazing news! Congratulations!
[S1 (grateful)]: Thank you so much.
```

## Usage

### Basic Dialogue Generation

```bash
tts-toolkit dialogue conversation.txt \
    --output dialogue.wav \
    --s1 speaker1.wav \
    --s1-text "Speaker one sample" \
    --s2 speaker2.wav \
    --s2-text "Speaker two sample"
```

### Using Voice Profiles

```bash
# Create profiles
tts-toolkit voice create alice --audio alice.wav --text "..."
tts-toolkit voice create bob --audio bob.wav --text "..."

# Generate dialogue
tts-toolkit dialogue conversation.txt \
    --output dialogue.wav \
    --s1 alice \
    --s2 bob
```

### Python API

```python
from tts_toolkit.formats.dialogue import DialogueHandler
from tts_toolkit.backends import QwenBackend

backend = QwenBackend(device="cpu")
handler = DialogueHandler(backend=backend)

# Parse and generate
with open("conversation.txt") as f:
    text = f.read()

segments = handler.parse(text)
speakers = handler.detect_speakers(segments)
print(f"Detected speakers: {speakers}")

output = handler.generate(
    segments=segments,
    output_path="dialogue.wav",
    speaker_refs={
        speakers[0]: ("speaker1.wav", "Sample one"),
        speakers[1]: ("speaker2.wav", "Sample two"),
    },
)

print(f"Duration: {output.duration_sec:.1f}s")
```

### With Emotions

```python
from tts_toolkit.formats.dialogue import DialogueHandler
from tts_toolkit.parsers.dialogue_markup import DialogueMarkupParser

parser = DialogueMarkupParser()
segments = parser.parse_with_emotions(text)

handler = DialogueHandler()
output = handler.generate_with_emotions(
    segments=segments,
    emotion_map={
        "S1": "happy",  # Default emotion for S1
        "S2": "neutral",
    },
)
```

## Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `input` | Input dialogue file | Yes |
| `--output, -o` | Output audio file | Yes |
| `--s1` | Speaker 1 voice profile or audio | No* |
| `--s1-text` | Speaker 1 reference transcript | No |
| `--s2` | Speaker 2 voice profile or audio | No* |
| `--s2-text` | Speaker 2 reference transcript | No |
| `--language, -l` | Language code | No |
| `--backend` | Backend to use (qwen, mock) | No |
| `--model, -m` | Model name | No |
| `--device, -d` | Device (cpu, cuda:0, mps) | No |

*At least one speaker reference is needed.

## Timing & Pauses

Default timing (customizable in Python API):
- Speaker change: 300ms pause
- Same speaker continuation: 150ms pause

## Available Emotions

| Emotion | Description |
|---------|-------------|
| `neutral` | Natural speaking tone |
| `happy` | Cheerful, upbeat |
| `sad` | Melancholic, subdued |
| `excited` | Energetic, enthusiastic |
| `serious` | Formal, professional |
| `calm` | Relaxed, soothing |
| `angry` | Intense, frustrated |
| `whisper` | Quiet, intimate |

## Tips

1. **Distinct Voices**: Use clearly different voices for speakers
2. **Natural Dialogue**: Include natural conversation patterns
3. **Emotion Tags**: Use emotions sparingly for key moments
4. **Short Lines**: Keep individual lines reasonable length
5. **Testing**: Preview with short excerpts first

## Example Script

```
[S1]: Good morning! Did you sleep well?

[S2]: Not really, the neighbors were noisy again.

[S1]: Oh no, that's frustrating. Have you talked to them?

[S2]: I tried once, but they didn't seem to care.

[S1]: Maybe you should write them a note?

[S2]: That's actually a good idea. I'll try that.
```
