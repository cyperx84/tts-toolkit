# Podcast Skill

Generate multi-speaker podcast audio with TTS Toolkit.

## When to Use

Use this skill when the user wants to:
- Create podcast episodes with multiple speakers
- Generate interview-style audio content
- Produce conversational audio with host and guest voices
- Add intro/outro music to podcast episodes

## Prerequisites

- TTS Toolkit installed
- Reference audio for each speaker (3-10 seconds each)
- Script with speaker markup

## Script Format

Use `[SPEAKER]:` markup in your script:

```
[HOST]: Welcome to the show! Today we're discussing AI.

[GUEST]: Thanks for having me. It's a fascinating topic.

[HOST]: So tell us, what got you interested in this field?

[GUEST]: Well, it started when I was in college...
```

You can also use segment markers:

```
[INTRO]

[HOST]: Welcome back to Tech Talk!

[SEGMENT: Interview]

[HOST]: Let's dive into our main topic.
[GUEST]: I'm excited to share my thoughts.

[OUTRO]
```

## Usage

### Basic Podcast Generation

```bash
tts-toolkit podcast script.txt \
    --output episode.wav \
    --host host_voice.wav \
    --host-text "Host voice sample transcript" \
    --guest guest_voice.wav \
    --guest-text "Guest voice sample transcript"
```

### With Music

```bash
tts-toolkit podcast script.txt \
    --output episode.wav \
    --host host.wav --host-text "..." \
    --guest guest.wav --guest-text "..." \
    --intro intro_music.mp3 \
    --outro outro_music.mp3 \
    --background ambient.mp3 \
    --background-volume -18
```

### Using Saved Voice Profiles

```bash
# First create profiles
tts-toolkit voice create podcast-host --audio host.wav --text "..."
tts-toolkit voice create podcast-guest --audio guest.wav --text "..."

# Then use them
tts-toolkit podcast script.txt \
    --output episode.wav \
    --host podcast-host \
    --guest podcast-guest
```

### Python API

```python
from tts_toolkit.formats.podcast import PodcastHandler
from tts_toolkit.backends import QwenBackend

backend = QwenBackend(device="cpu")
handler = PodcastHandler(
    backend=backend,
    intro_music="intro.mp3",
    outro_music="outro.mp3",
    background_music="ambient.mp3",
    background_volume_db=-18,
)

output = handler.generate_episode(
    script_path="script.txt",
    output_path="episode.wav",
    host_ref=("host.wav", "Host transcript"),
    guest_ref=("guest.wav", "Guest transcript"),
)

print(f"Speakers: {output.metadata['speakers']}")
print(f"Duration: {output.duration_sec / 60:.1f} minutes")
```

## Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `input` | Input script file | Yes |
| `--output, -o` | Output audio file | Yes |
| `--host` | Host voice profile or audio file | No |
| `--host-text` | Host reference transcript | No |
| `--guest` | Guest voice profile or audio file | No |
| `--guest-text` | Guest reference transcript | No |
| `--intro` | Intro music file | No |
| `--outro` | Outro music file | No |
| `--background` | Background music file | No |
| `--background-volume` | Background volume in dB (default: -18) | No |

## Tips

1. **Voice Differentiation**: Use distinct voices for host and guest
2. **Pause Timing**: The system automatically adds pauses between speakers
3. **Music Volume**: -18dB is a good starting point for background music
4. **Crossfades**: Intro/outro automatically crossfade with content
5. **Long Episodes**: Large scripts work well due to automatic chunking

## Output

- Combined WAV file with all speakers and music
- Automatic ducking of background music during speech
- Smooth transitions between speakers

## Example Script

```
[INTRO]

[HOST]: Welcome to AI Weekly, the podcast where we explore the cutting edge of artificial intelligence. I'm your host, Alex Chen.

[HOST]: Today we have a special guest joining us - Dr. Sarah Miller, a leading researcher in natural language processing.

[GUEST]: Thank you, Alex! I'm thrilled to be here.

[SEGMENT: Current Research]

[HOST]: So Sarah, tell us about your latest research.

[GUEST]: We've been working on improving how AI understands context in conversations. It's fascinating work.

[HOST]: That sounds amazing. How does it work?

[GUEST]: Well, the key insight is that context matters at multiple levels...

[OUTRO]
```
