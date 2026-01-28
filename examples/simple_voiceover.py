"""Simple voiceover example.

This example demonstrates how to generate a voiceover from text using
voice cloning with TTS Toolkit.
"""

from tts_toolkit.formats.voiceover import VoiceoverHandler

# For Qwen backend (requires qwen-tts package)
try:
    from tts_toolkit.backends import QwenBackend
    backend = QwenBackend(device="cpu")  # Use "cuda:0" for GPU
except ImportError:
    # Fall back to mock backend for testing without GPU
    from tts_toolkit.backends import MockBackend
    backend = MockBackend()
    print("Using MockBackend (qwen-tts not available)")

# Create voiceover handler
handler = VoiceoverHandler(backend=backend)

# Your script text
script = """
Welcome to TTS Toolkit, an open-source text-to-speech toolkit.

This toolkit allows you to create podcasts, audiobooks, and voiceovers
using state-of-the-art voice cloning technology.

Let's get started with your first voiceover!
"""

# Generate voiceover
output = handler.process(
    input_text=script,
    output_path="voiceover_output.wav",
    ref_audio="path/to/your/reference.wav",  # 3-10 seconds of clear speech
    ref_text="This is the exact transcript of the reference audio.",
    language="English",
)

print(f"Generated {output.duration_sec:.1f} seconds of audio")
print(f"Saved to: voiceover_output.wav")
