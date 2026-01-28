# Contributing to TTS Toolkit

Thank you for your interest in contributing to TTS Toolkit! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- (Optional) CUDA-capable GPU for testing TTS backends

### Setting Up the Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/tts-toolkit.git
cd tts-toolkit

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,all-backends]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tts_toolkit --cov-report=html

# Run specific test file
pytest tests/test_backends.py

# Run with verbose output
pytest -v
```

### Code Style

We use `black` for formatting and `ruff` for linting:

```bash
# Format code
black tts_toolkit tests

# Check linting
ruff check tts_toolkit tests

# Fix auto-fixable issues
ruff check --fix tts_toolkit tests
```

## Project Structure

```
tts-toolkit/
├── tts_toolkit/
│   ├── backends/       # TTS engine implementations
│   ├── core/           # Core processing (Pipeline, Chunker, Stitcher)
│   ├── formats/        # Format handlers (Voiceover, Podcast, etc.)
│   ├── voices/         # Voice profiles and emotions
│   ├── evaluation/     # Quality metrics
│   ├── export/         # Audio export (WAV, MP3)
│   └── cli.py          # Command-line interface
├── tests/              # Test suite
├── docs/               # Documentation
└── examples/           # Example scripts
```

## Adding a New TTS Backend

1. Create a new file in `tts_toolkit/backends/`:

```python
# tts_toolkit/backends/my_backend.py
from .base import TTSBackend, VoicePrompt

class MyBackend(TTSBackend):
    def load_model(self) -> None:
        # Load your TTS model
        pass

    def create_voice_prompt(self, reference_audio, reference_text, **kwargs):
        # Create voice prompt from reference
        return VoicePrompt(...)

    def generate(self, text, voice_prompt, language="Auto", **kwargs):
        # Generate audio
        return audio_array, sample_rate

    @property
    def sample_rate(self) -> int:
        return 24000

    def supports_voice_cloning(self) -> bool:
        return True
```

2. Register in `tts_toolkit/backends/__init__.py`:

```python
# Add lazy import
def get_my_backend():
    from .my_backend import MyBackend
    return MyBackend

# Add to registry
AVAILABLE_BACKENDS["my_backend"] = (
    "MyBackend",
    "my-tts-lib",
    "Description of features"
)

# Optional: Direct import if dependencies available
try:
    from .my_backend import MyBackend
    __all__.append("MyBackend")
except ImportError:
    pass
```

3. Add optional dependency to `pyproject.toml`:

```toml
[project.optional-dependencies]
my-backend = ["my-tts-lib>=1.0"]
```

4. Update CLI in `tts_toolkit/cli.py`:

```python
elif backend_name == "my_backend":
    from .backends import MyBackend
    return MyBackend(device=args.device)
```

5. Add tests in `tests/test_backends.py`

6. Update documentation in `docs/api-reference.md`

## Commit Guidelines

We follow conventional commits:

```
feat: Add new TTS backend for XYZ
fix: Handle empty text input in chunker
docs: Update API reference for new backends
test: Add tests for evaluation module
refactor: Simplify audio stitching logic
```

### Commit Message Format

```
<type>: <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding or updating tests
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement
- `chore`: Maintenance tasks

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes with tests
3. Ensure all tests pass: `pytest`
4. Ensure code is formatted: `black tts_toolkit tests`
5. Ensure linting passes: `ruff check tts_toolkit tests`
6. Update documentation if needed
7. Submit a pull request with a clear description

### PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code formatted with `black`
- [ ] Linting passes with `ruff`
- [ ] Commit messages follow conventions
- [ ] PR description explains the changes

## Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Minimal code example
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: Python version, OS, backend being used
6. **Error Messages**: Full traceback if applicable

## Feature Requests

Feature requests are welcome! Please include:

1. **Use Case**: Why is this feature needed?
2. **Proposed Solution**: How would it work?
3. **Alternatives**: Any workarounds you've considered?

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

## Questions?

Open an issue with the "question" label or start a discussion.

Thank you for contributing!
