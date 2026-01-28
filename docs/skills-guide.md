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

## Best Practices

1. **Voice Profiles**: Create profiles for frequently used voices
2. **Script Format**: Use standard markup for multi-speaker content
3. **Chunking**: Let the toolkit handle text chunking for long content
4. **GPU**: Use CUDA for faster generation when available

## Troubleshooting

### Skill Not Found

Ensure you're in the tts-toolkit project directory or have it installed.

### Backend Not Available

Install the Qwen backend:
```bash
pip install tts-toolkit[qwen]
```

### Voice Profile Missing

Create the profile first:
```bash
tts-toolkit voice create <name> --audio <file> --text "..."
```
