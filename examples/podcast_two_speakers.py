"""Podcast with two speakers example.

This example demonstrates how to generate a podcast episode with
host and guest speakers using TTS Toolkit.
"""

from tts_toolkit.formats.podcast import PodcastHandler

# Create backend
try:
    from tts_toolkit.backends import QwenBackend
    backend = QwenBackend(device="cpu")
except ImportError:
    from tts_toolkit.backends import MockBackend
    backend = MockBackend()
    print("Using MockBackend (qwen-tts not available)")

# Create podcast handler with optional music
handler = PodcastHandler(
    backend=backend,
    intro_music="assets/intro.mp3",  # Optional
    outro_music="assets/outro.mp3",  # Optional
    background_music=None,
    pause_between_speakers_ms=400,
)

# Podcast script with [SPEAKER]: markup
script = """
[INTRO]

[HOST]: Welcome to Tech Talk, the podcast where we explore the latest in technology!

[HOST]: Today, I'm joined by a special guest. Please introduce yourself!

[GUEST]: Thanks for having me! I'm excited to be here and talk about AI.

[HOST]: So tell us, what's the most exciting development in AI right now?

[GUEST]: Well, I think the progress in voice synthesis is incredible.
Being able to clone voices with just a few seconds of audio is amazing.

[HOST]: Absolutely! And the applications are endless.

[GUEST]: Exactly. From accessibility to content creation, it's transforming everything.

[OUTRO]
"""

# Generate podcast
output = handler.process(
    input_text=script,
    output_path="podcast_episode.wav",
    speaker_refs={
        "HOST": ("host_voice.wav", "Host sample transcript"),
        "GUEST": ("guest_voice.wav", "Guest sample transcript"),
    },
    language="English",
    add_music=False,  # Set True if you have music files
)

print(f"Generated podcast: {output.duration_sec:.1f} seconds")
print(f"Speakers: {output.metadata['speakers']}")
print(f"Saved to: podcast_episode.wav")
