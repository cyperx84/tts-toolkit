"""Dialogue generation example.

This example demonstrates how to generate a two-person dialogue
using TTS Toolkit's dialogue handler.
"""

from tts_toolkit.formats.dialogue import DialogueHandler

# Create backend
try:
    from tts_toolkit.backends import QwenBackend
    backend = QwenBackend(device="cpu")
except ImportError:
    from tts_toolkit.backends import MockBackend
    backend = MockBackend()
    print("Using MockBackend (qwen-tts not available)")

# Create dialogue handler
handler = DialogueHandler(
    backend=backend,
    pause_between_speakers_ms=300,  # Pause when speaker changes
    pause_between_lines_ms=150,     # Pause for same speaker continuation
)

# Dialogue script using Dia-style markup
dialogue_script = """
[S1]: Good morning! How did you sleep?

[S2]: Pretty well, thanks. Ready for today's presentation?

[S1]: A bit nervous, to be honest. It's a big audience.

[S2]: You'll do great! You've prepared so well for this.

[S1]: Thanks for the encouragement. Let's grab some coffee first.

[S2]: Perfect idea. The cafe downstairs makes excellent lattes.
"""

# Parse the dialogue to see detected speakers
segments = handler.parse(dialogue_script)
speakers = handler.detect_speakers(segments)
print(f"Detected speakers: {speakers}")
print(f"Total segments: {len(segments)}")

# Generate dialogue audio
output = handler.generate(
    segments=segments,
    output_path="dialogue_output.wav",
    speaker_refs={
        "1": ("speaker1.wav", "Speaker one voice sample"),  # Maps to S1
        "2": ("speaker2.wav", "Speaker two voice sample"),  # Maps to S2
    },
    language="English",
)

print(f"\nGenerated dialogue: {output.duration_sec:.1f} seconds")
print(f"Saved to: dialogue_output.wav")

# Alternative: using named speakers
named_dialogue = """
[ALICE]: Hey Bob, have you tried that new TTS toolkit?

[BOB]: Not yet! Is it good?

[ALICE]: It's amazing. The voice cloning is really impressive.

[BOB]: I'll definitely check it out. Thanks for the tip!
"""

segments = handler.parse(named_dialogue)
speakers = handler.detect_speakers(segments)
print(f"\nNamed dialogue speakers: {speakers}")
