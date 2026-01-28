# API Reference

## Core Components

### Pipeline

The main class for long-form TTS generation with checkpointing.

```python
from tts_toolkit import Pipeline
from tts_toolkit.backends import QwenBackend

pipeline = Pipeline(
    backend=QwenBackend(),
    chunk_min=100,      # Minimum chars per chunk
    chunk_max=300,      # Maximum chars per chunk
    chunk_target=200,   # Target chars per chunk
    crossfade_ms=75,    # Crossfade between chunks
    max_retries=3,      # Retry failed chunks
)

# Process text to audio
pipeline.process(
    text="Your long text...",
    ref_audio="reference.wav",
    ref_text="Reference transcript",
    output_path="output.wav",
    work_dir=None,      # Auto-generated if None
    language="Auto",
    resume=True,        # Resume from checkpoint
    progress_callback=None,
)
```

### TextChunker

Split text into optimal chunks for TTS.

```python
from tts_toolkit.core import TextChunker

chunker = TextChunker(
    min_chars=100,
    max_chars=300,
    target_chars=200,
)

chunks = chunker.chunk("Your long text...")
```

### AudioStitcher

Combine audio segments with crossfade.

```python
from tts_toolkit.core import AudioStitcher

stitcher = AudioStitcher(
    crossfade_ms=75,
    sample_rate=24000,
)

# Stitch numpy arrays
combined = stitcher.stitch([audio1, audio2, audio3])

# Stitch files
stitcher.stitch_files(
    ["chunk1.wav", "chunk2.wav"],
    "output.wav",
    normalize=True,
)
```

## Backends

### TTSBackend (Abstract)

```python
from tts_toolkit.backends.base import TTSBackend, VoicePrompt

class MyBackend(TTSBackend):
    def load_model(self) -> None: ...

    def create_voice_prompt(
        self,
        reference_audio: str,
        reference_text: str,
        **kwargs
    ) -> VoicePrompt: ...

    def generate(
        self,
        text: str,
        voice_prompt: VoicePrompt,
        language: str = "Auto",
        **kwargs
    ) -> Tuple[np.ndarray, int]: ...

    @property
    def sample_rate(self) -> int: ...

    def supports_voice_cloning(self) -> bool: ...
```

### QwenBackend

```python
from tts_toolkit.backends import QwenBackend

backend = QwenBackend(
    model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device="cuda:0",
)

backend.load_model()
voice_prompt = backend.create_voice_prompt("ref.wav", "Transcript")
audio, sr = backend.generate("Hello", voice_prompt)
```

### MockBackend

```python
from tts_toolkit.backends import MockBackend

backend = MockBackend(sample_rate=24000)
audio, sr = backend.generate("Hello", voice_prompt=None)
```

## Format Handlers

### VoiceoverHandler

```python
from tts_toolkit.formats import VoiceoverHandler

handler = VoiceoverHandler(backend=backend)

output = handler.process(
    input_text="Your script...",
    output_path="voiceover.wav",
    ref_audio="voice.wav",
    ref_text="Transcript",
    language="English",
)
```

### DialogueHandler

```python
from tts_toolkit.formats import DialogueHandler

handler = DialogueHandler(
    backend=backend,
    pause_between_speakers_ms=300,
    pause_between_lines_ms=150,
)

segments = handler.parse("[S1]: Hello\n[S2]: Hi there!")
speakers = handler.detect_speakers(segments)

output = handler.generate(
    segments=segments,
    output_path="dialogue.wav",
    speaker_refs={"1": ("s1.wav", "Transcript")},
)
```

### PodcastHandler

```python
from tts_toolkit.formats import PodcastHandler

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
    host_ref=("host.wav", "Transcript"),
    guest_ref=("guest.wav", "Transcript"),
)
```

### AudiobookHandler

```python
from tts_toolkit.formats import AudiobookHandler

handler = AudiobookHandler(
    backend=backend,
    chapter_pause_ms=2000,
    paragraph_pause_ms=500,
)

result = handler.generate_book(
    input_path="book.md",
    output_dir="./audiobook",
    ref_audio="narrator.wav",
    ref_text="Transcript",
    format="markdown",
    combine=True,
)
```

## Voice Management

### VoiceProfile

```python
from tts_toolkit.voices import VoiceProfile

profile = VoiceProfile(
    name="narrator",
    reference_audio="voice.wav",
    reference_text="Transcript",
    description="Professional narrator",
)
```

### VoiceRegistry

```python
from tts_toolkit.voices import VoiceRegistry

registry = VoiceRegistry()

# Create profile
profile = registry.create(
    name="narrator",
    reference_audio="voice.wav",
    reference_text="Transcript",
)

# Get profile
profile = registry.get("narrator")

# List profiles
for name in registry.list():
    print(name)

# Delete profile
registry.delete("narrator")
```

## Emotions

```python
from tts_toolkit.voices.emotions import apply_emotion, EMOTION_PRESETS

# Get available emotions
print(list(EMOTION_PRESETS.keys()))

# Apply emotion to generation kwargs
gen_kwargs = {"temperature": 0.9}
gen_kwargs = apply_emotion(gen_kwargs, "happy")
```

## Data Classes

### Segment

```python
from tts_toolkit.formats.base import Segment

segment = Segment(
    text="Hello, world!",
    speaker_id="S1",
    voice_id=None,
    emotion="happy",
    pause_before_ms=0,
    pause_after_ms=50,
    metadata={},
)
```

### AudioOutput

```python
from tts_toolkit.formats.base import AudioOutput

output = AudioOutput(
    audio=np.array([...]),
    sample_rate=24000,
    segments=[...],
    duration_sec=10.5,
    metadata={"speakers": ["S1", "S2"]},
)
```
