# Voiceover Skill

Generate video voiceover audio with voice cloning using TTS Toolkit.

## When to Use

Use this skill when the user wants to:
- Create narration for videos or tutorials
- Generate voiceover from a script
- Clone a voice for video narration
- Convert text to speech with consistent voice

## Prerequisites

- TTS Toolkit installed (`pip install tts-toolkit` or `pip install tts-toolkit[qwen]`)
- Reference audio file (3-10 seconds of clear speech)
- Reference text (exact transcript of reference audio)

## Usage

### Basic Voiceover Generation

```bash
# Using CLI
tts-toolkit voiceover script.txt \
    --output output.wav \
    --ref-audio voice_sample.wav \
    --ref-text "Transcript of the voice sample"

# Using saved voice profile
tts-toolkit voiceover script.txt \
    --output output.wav \
    --voice narrator
```

### With SRT Timing

```bash
tts-toolkit voiceover script.txt \
    --output output.wav \
    --ref-audio voice.wav \
    --ref-text "Sample text" \
    --srt subtitles.srt
```

### Python API

```python
from tts_toolkit.formats.voiceover import VoiceoverHandler
from tts_toolkit.backends import QwenBackend  # or MockBackend

# Setup with Qwen backend
backend = QwenBackend(model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base", device="cpu")
handler = VoiceoverHandler(backend=backend)

# Read script
with open("script.txt", "r") as f:
    text = f.read()

# Generate voiceover
output = handler.process(
    input_text=text,
    output_path="output.wav",
    ref_audio="voice_sample.wav",
    ref_text="Sample transcript",
    language="English",
)

print(f"Generated {output.duration_sec:.1f}s of audio")
```

## Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `input` | Input text file | Yes |
| `--output, -o` | Output audio file | Yes |
| `--voice, -v` | Voice profile name | No* |
| `--ref-audio, -a` | Reference audio file | No* |
| `--ref-text, -r` | Reference audio transcript | No* |
| `--srt` | SRT file for timing sync | No |
| `--language, -l` | Language code (default: Auto) | No |
| `--backend` | Backend to use (qwen, mock) | No |
| `--model, -m` | Model name | No |
| `--device, -d` | Device (cpu, cuda:0, mps) | No |
| `--temperature` | Sampling temperature | No |

*Either `--voice` OR (`--ref-audio` AND `--ref-text`) is required.

## Tips

1. **Reference Audio Quality**: Use 3-10 seconds of clear speech without background noise
2. **Reference Text Accuracy**: The transcript must exactly match the spoken words
3. **Language**: Set explicitly for best results (e.g., `--language English`)
4. **Chunking**: Long texts are automatically split into optimal chunks
5. **GPU Acceleration**: Use `--device cuda:0` for faster generation

## Output

- WAV file at 24kHz sample rate
- Automatic crossfade between chunks for smooth transitions
- Normalized audio to prevent clipping

## Example Workflow

1. Create a voice profile once:
   ```bash
   tts-toolkit voice create narrator \
       --audio my_voice.wav \
       --text "Hello, this is my sample recording"
   ```

2. Generate voiceovers using the saved profile:
   ```bash
   tts-toolkit voiceover script.txt \
       --output video_narration.wav \
       --voice narrator
   ```
