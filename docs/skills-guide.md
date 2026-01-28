# Claude Code Skills Guide

TTS Toolkit includes Claude Code skills for seamless integration with Claude Code CLI.

## Available Skills

| Skill | Description |
|-------|-------------|
| `voiceover` | Generate video voiceover with voice cloning |
| `podcast` | Generate multi-speaker podcast episodes |
| `audiobook` | Generate audiobooks with chapters |
| `dialogue` | Generate two-person conversations |
| `voice-profile` | Manage voice profiles |
| `quick-tts` | Fast single-utterance TTS |

## Using Skills

When working with Claude Code in the tts-toolkit directory, Claude will automatically detect and use these skills based on context.

### Example Prompts

**Voiceover:**
> "Create a voiceover for this script using my narrator voice profile"

**Podcast:**
> "Generate a podcast episode from this conversation script with two speakers"

**Audiobook:**
> "Convert this markdown document into an audiobook with chapters"

**Dialogue:**
> "Generate audio for this dialogue between Alice and Bob"

**Voice Profile:**
> "Create a new voice profile called 'professional' from this audio sample"

**Quick TTS:**
> "Generate a quick audio clip saying 'Welcome to our app'"

## Skill Locations

Skills are located in `.claude/skills/`:

```
.claude/skills/
├── voiceover/SKILL.md
├── podcast/SKILL.md
├── audiobook/SKILL.md
├── dialogue/SKILL.md
├── voice-profile/SKILL.md
└── quick-tts/SKILL.md
```

## Extending Skills

You can create custom skills by adding new directories to `.claude/skills/`:

```markdown
# My Custom Skill

## When to Use

Describe when Claude should use this skill...

## Usage

### CLI
```bash
tts-toolkit my-custom-command ...
```

### Python
```python
# Custom usage example
```

## Arguments

| Argument | Description |
|----------|-------------|
| `--option` | Description |
```

## Integration with Other Tools

The skills integrate with:

1. **Claude Code CLI**: Automatic skill detection and execution
2. **Voice Registry**: Access saved voice profiles
3. **TTS Backends**: Use configured backends
4. **Audio Mixer**: Background music and effects
5. **Evaluation**: Quality metrics and benchmarking

## Selecting a Backend

Choose the right backend for your use case:

| Backend | Best For | Install |
|---------|----------|---------|
| `QwenBackend` | General production use | `tts-toolkit[qwen]` |
| `ChatterboxBackend` | Emotional content, sound effects | `tts-toolkit[chatterbox]` |
| `KokoroBackend` | Fast generation, low resources | `tts-toolkit[kokoro]` |
| `BarkBackend` | Expressive content with laughs/sighs | `tts-toolkit[bark]` |
| `CoquiXTTSBackend` | Multi-language content | `tts-toolkit[coqui]` |
| `FishSpeechBackend` | Cloud API (no GPU needed) | `tts-toolkit[fish-speech]` |
| `CosyVoice2Backend` | Low-latency streaming, Chinese | Manual install |

### Configuring Backend in Code

```python
from tts_toolkit.backends import ChatterboxBackend
from tts_toolkit.formats import VoiceoverHandler

# Use Chatterbox for emotional content
backend = ChatterboxBackend(device="cuda")
handler = VoiceoverHandler(backend=backend)

# Generate with emotion tags
handler.process(
    input_text="Hi there [chuckle], great to see you!",
    output_path="output.wav",
    ref_audio="speaker.wav",
    ref_text="Reference transcript",
)
```

## Best Practices

1. **Voice Profiles**: Create profiles for frequently used voices
2. **Script Format**: Use standard markup for multi-speaker content
3. **Chunking**: Let the toolkit handle text chunking for long content
4. **GPU**: Use CUDA for faster generation when available
5. **Backend Selection**: Match backend to content type
   - Use Chatterbox/Bark for expressive content
   - Use Kokoro for quick prototyping
   - Use Coqui XTTS for multilingual content

## Troubleshooting

### Skill Not Found

Ensure you're in the tts-toolkit project directory or have it installed.

### Backend Not Available

Install your preferred backend:
```bash
# Choose one or more:
pip install tts-toolkit[qwen]        # Qwen3-TTS
pip install tts-toolkit[chatterbox]  # Chatterbox with emotion
pip install tts-toolkit[kokoro]      # Kokoro (fast, lightweight)
pip install tts-toolkit[bark]        # Bark (expressive)
pip install tts-toolkit[coqui]       # Coqui XTTS (multilingual)
pip install tts-toolkit[fish-speech] # Fish Speech (API)

# Or install all:
pip install tts-toolkit[all-backends]
```

### Voice Profile Missing

Create the profile first:
```bash
tts-toolkit voice create <name> --audio <file> --text "..."
```

### Evaluating Quality

Check generation quality:
```python
from tts_toolkit.evaluation import TTSEvaluator

evaluator = TTSEvaluator()
results = evaluator.evaluate(
    audio_path="output.wav",
    text="Original text",
    reference_audio="speaker.wav",
)
print(f"MOS: {results['utmos']:.2f}")
print(f"Speaker Similarity: {results['speaker_similarity']:.2f}")
```
