"""Audiobook from Markdown example.

This example demonstrates how to generate an audiobook with chapters
from a Markdown file using TTS Toolkit.
"""

import os
from tts_toolkit.formats.audiobook import AudiobookHandler

# Create backend
try:
    from tts_toolkit.backends import QwenBackend
    backend = QwenBackend(device="cpu")
except ImportError:
    from tts_toolkit.backends import MockBackend
    backend = MockBackend()
    print("Using MockBackend (qwen-tts not available)")

# Create audiobook handler
handler = AudiobookHandler(
    backend=backend,
    chunk_min=100,
    chunk_max=300,
    chunk_target=200,
    chapter_pause_ms=2000,  # 2 seconds between chapters
    paragraph_pause_ms=500,  # 0.5 seconds between paragraphs
)

# Sample Markdown content
markdown_content = """
# Chapter 1: The Beginning

It was a bright and sunny morning when our story begins.
The birds were singing, and the world seemed full of possibility.

Our protagonist, Alex, woke up feeling different today.
Something extraordinary was about to happen.

# Chapter 2: The Discovery

Walking through the old library, Alex stumbled upon a mysterious book.
Its pages were yellowed with age, but the text was clear as day.

"This is it," Alex whispered, "the key to everything."

# Chapter 3: The Journey Begins

With the ancient book in hand, Alex set out on an adventure.
The path ahead was uncertain, but the destination was clear.

The world would never be the same again.
"""

# Save markdown to file
with open("sample_book.md", "w") as f:
    f.write(markdown_content)

# Generate audiobook
result = handler.generate_book(
    input_path="sample_book.md",
    output_dir="./audiobook_output",
    ref_audio="narrator_voice.wav",  # Your narrator's voice sample
    ref_text="This is the narrator's voice sample transcript.",
    format="markdown",
    combine=True,  # Create full audiobook file
    language="English",
)

# Print results
print(f"Generated {result['total_chapters']} chapters")
print(f"Total duration: {result['total_duration_sec'] / 60:.1f} minutes")
print(f"\nChapter details:")
for ch in result['chapters']:
    print(f"  - Chapter {ch['chapter']}: {ch['title']} ({ch['duration_sec']:.0f}s)")

if result['combined_path']:
    print(f"\nCombined audiobook: {result['combined_path']}")

# Clean up
os.remove("sample_book.md")
