# Audiobook Skill

Generate audiobooks with chapter support using TTS Toolkit.

## When to Use

Use this skill when the user wants to:
- Convert a book or long text into an audiobook
- Generate chapter-by-chapter audio from Markdown
- Create narrated versions of documentation
- Produce long-form audio content with consistent voice

## Prerequisites

- TTS Toolkit installed
- Reference audio (3-10 seconds of narrator voice)
- Text or Markdown file with content

## Input Formats

### Markdown (recommended)
Chapters detected from headers:

```markdown
# Chapter 1: The Beginning

It was a dark and stormy night...

## Part One

The adventure begins here.

# Chapter 2: The Journey

Our heroes set out on their quest...
```

### Numbered Chapters
```
Chapter 1: The Beginning

It was a dark and stormy night...

Chapter 2: The Journey

Our heroes set out on their quest...
```

### Plain Text
Treated as a single chapter.

## Usage

### Basic Audiobook Generation

```bash
tts-toolkit audiobook book.md \
    --output-dir ./audiobook \
    --ref-audio narrator.wav \
    --ref-text "Sample narrator speech"
```

### With Voice Profile

```bash
# Create narrator profile
tts-toolkit voice create narrator \
    --audio narrator.wav \
    --text "Sample narrator speech"

# Generate audiobook
tts-toolkit audiobook book.md \
    --output-dir ./audiobook \
    --voice narrator
```

### Different Input Formats

```bash
# Markdown (default)
tts-toolkit audiobook book.md --output-dir ./out --voice narrator

# Numbered chapters
tts-toolkit audiobook book.txt --output-dir ./out --voice narrator --format numbered

# Plain text (single chapter)
tts-toolkit audiobook article.txt --output-dir ./out --voice narrator --format plain
```

### Python API

```python
from tts_toolkit.formats.audiobook import AudiobookHandler
from tts_toolkit.backends import QwenBackend

backend = QwenBackend(device="cpu")
handler = AudiobookHandler(backend=backend)

result = handler.generate_book(
    input_path="book.md",
    output_dir="./audiobook",
    ref_audio="narrator.wav",
    ref_text="Sample narrator speech",
    format="markdown",
    combine=True,  # Create full_audiobook.wav
)

print(f"Generated {result['total_chapters']} chapters")
print(f"Total duration: {result['total_duration_sec'] / 60:.1f} minutes")
for ch in result['chapters']:
    print(f"  - {ch['title']}: {ch['duration_sec']:.0f}s")
```

## Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `input` | Input text/markdown file | Yes |
| `--output-dir, -o` | Output directory | Yes |
| `--voice, -v` | Voice profile name | No* |
| `--ref-audio, -a` | Reference audio file | No* |
| `--ref-text, -r` | Reference audio transcript | No* |
| `--format` | Input format (markdown, plain, numbered) | No |
| `--no-combine` | Don't create combined audiobook file | No |
| `--language, -l` | Language code | No |
| `--backend` | Backend to use (qwen, mock) | No |
| `--model, -m` | Model name | No |
| `--device, -d` | Device (cpu, cuda:0, mps) | No |

*Either `--voice` OR (`--ref-audio` AND `--ref-text`) is required.

## Output Structure

```
audiobook/
├── chapter_01.wav
├── chapter_02.wav
├── chapter_03.wav
└── full_audiobook.wav  (combined, unless --no-combine)
```

## Tips

1. **Voice Consistency**: Use a high-quality reference audio for consistent narration
2. **Chapter Pauses**: 2-second pauses are automatically added between chapters
3. **Paragraph Pauses**: 500ms pauses between paragraphs for natural pacing
4. **Progress Tracking**: Long books support checkpoint/resume
5. **GPU Recommended**: Use `--device cuda:0` for long books

## Performance Estimates

On M4 Mac (CPU):
- Short chapter (1000 words): ~5-10 minutes
- Full book (50,000 words): ~4-6 hours

On GPU (CUDA):
- 5-10x faster than CPU

## Example Workflow

1. Prepare your book as Markdown with chapter headers
2. Create a narrator voice profile
3. Run audiobook generation overnight
4. Review chapter files individually
5. Use the combined file for distribution
