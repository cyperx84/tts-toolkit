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
    def supports_streaming(self) -> bool: ...
    def supports_emotions(self) -> bool: ...
    def get_supported_languages(self) -> List[str]: ...
```

### QwenBackend

Production backend with streaming and emotion support.

```python
from tts_toolkit.backends import QwenBackend

backend = QwenBackend(
    model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device="cuda:0",
)

backend.load_model()
voice_prompt = backend.create_voice_prompt("ref.wav", "Transcript")
audio, sr = backend.generate("Hello", voice_prompt, language="English")
```

### ChatterboxBackend

Emotion-aware TTS with paralinguistic tags.

```python
from tts_toolkit.backends import ChatterboxBackend

backend = ChatterboxBackend(
    model_type="default",  # or "turbo" for faster inference
    device="cuda",
)

backend.load_model()
voice_prompt = backend.create_voice_prompt("ref.wav", "Reference text")

# Generate with emotion tags
audio, sr = backend.generate(
    text="Hi there [chuckle], how are you doing?",
    voice_prompt=voice_prompt,
    exaggeration=0.5,  # 0.0-1.0, emotion intensity
)

# Supported tags: [laugh], [chuckle], [sigh], [gasp], [clears throat]
```

### KokoroBackend

Lightweight, fast backend (82M parameters).

```python
from tts_toolkit.backends import KokoroBackend

backend = KokoroBackend(
    voice="af_heart",  # Built-in voice
    lang_code="a",     # 'a'=American, 'b'=British
    speed=1.0,
)

# Built-in voices (no reference audio needed)
# American: af_heart, af_bella, af_nicole, af_sarah, af_sky
# American Male: am_adam, am_michael
# British: bf_emma, bf_isabella, bm_george, bm_lewis

backend.load_model()
voice_prompt = backend.create_voice_prompt("", "")  # Uses built-in voice
audio, sr = backend.generate("Hello, world!")
```

### FishSpeechBackend

API-based backend (no GPU required).

```python
from tts_toolkit.backends import FishSpeechBackend
import os

backend = FishSpeechBackend(
    api_key=os.environ["FISH_AUDIO_API_KEY"],
    model_id="speech-01-turbo",
)

backend.load_model()

# Option 1: Use reference audio for voice cloning
voice_prompt = backend.create_voice_prompt(
    reference_audio="voice_sample.wav",  # 10-30 seconds ideal
    reference_text="Transcript of the sample",
)

# Option 2: Use pre-uploaded voice model
voice_prompt = backend.create_voice_prompt(
    reference_audio="",
    reference_text="",
    reference_id="your-voice-model-id",
)

audio, sr = backend.generate("Hello from Fish Speech!", voice_prompt)
```

### BarkBackend

Expressive TTS with non-verbal sounds.

```python
from tts_toolkit.backends import BarkBackend

backend = BarkBackend(
    model_size="suno/bark",  # or "suno/bark-small"
    device="cuda",
)

backend.load_model()
voice_prompt = backend.create_voice_prompt("ref.wav", "Reference text")

# Generate with non-verbal sounds
audio, sr = backend.generate(
    text="Oh wow! [laughter] That's amazing [sighs].",
    voice_prompt=voice_prompt,
)

# Supported: [laughter], [laughs], [sighs], [music], [gasps], [clears throat]
# Use ♪ for singing: "♪ La la la ♪"
# Use ... for hesitation: "I... um... okay"
# Use CAPS for emphasis: "I REALLY love it"
```

### CosyVoice2Backend

Ultra-low latency streaming (150ms first packet).

```python
from tts_toolkit.backends import CosyVoice2Backend

backend = CosyVoice2Backend(
    model_dir="pretrained_models/CosyVoice2-0.5B",
    load_jit=False,
    load_trt=False,  # Enable for TensorRT acceleration
)

backend.load_model()
voice_prompt = backend.create_voice_prompt(
    reference_audio="ref_16k.wav",  # Must be 16kHz
    reference_text="Reference transcript",
)

# Standard generation
audio, sr = backend.generate("你好，世界！", voice_prompt)

# Streaming generation
for chunk_audio, chunk_sr in backend.generate_streaming(
    text="Long text here...",
    voice_prompt=voice_prompt,
):
    # Process each chunk as it arrives
    play_audio(chunk_audio)
```

### CoquiXTTSBackend

Multi-language TTS with 6-second voice cloning.

```python
from tts_toolkit.backends import CoquiXTTSBackend

backend = CoquiXTTSBackend(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    device="cuda",
)

backend.load_model()
voice_prompt = backend.create_voice_prompt(
    reference_audio="ref.wav",  # 6+ seconds recommended
    reference_text="",  # Not required for XTTS
)

# Generate in any of 17 languages
audio, sr = backend.generate(
    text="Bonjour, comment allez-vous?",
    voice_prompt=voice_prompt,
    language="fr",  # en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi
)
```

### MockBackend

Testing backend with no dependencies.

```python
from tts_toolkit.backends import MockBackend

backend = MockBackend(
    sample_rate=24000,
    mode="sine",  # "silent" or "sine"
)
audio, sr = backend.generate("Hello", voice_prompt=None)
```

### Backend Comparison

| Backend | Speed | Quality | GPU Required | Voice Cloning | Streaming |
|---------|-------|---------|--------------|---------------|-----------|
| Qwen | Medium | High | Recommended | 3-10s audio | ✅ |
| Chatterbox | Medium | High | Yes | 3-10s audio | ❌ |
| Kokoro | Fast | Good | No | Built-in voices | ❌ |
| Fish Speech | Varies | High | No (API) | 10-30s audio | ✅ |
| Bark | Slow | High | Recommended | Voice presets | ❌ |
| CosyVoice2 | Fast | High | Recommended | Short clips | ✅ |
| Coqui XTTS | Medium | High | Recommended | 6s audio | ✅ |

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
