"""Command-line interface for TTS Toolkit."""

import argparse
import sys
from typing import Optional


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TTS Toolkit - Generate podcasts, audiobooks, voiceovers, and dialogues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Voiceover command
    voiceover_parser = subparsers.add_parser(
        "voiceover",
        help="Generate video voiceover",
        description="Generate voiceover audio from text with voice cloning",
    )
    _add_voiceover_args(voiceover_parser)

    # Podcast command
    podcast_parser = subparsers.add_parser(
        "podcast",
        help="Generate multi-speaker podcast",
        description="Generate podcast audio with multiple speakers",
    )
    _add_podcast_args(podcast_parser)

    # Audiobook command
    audiobook_parser = subparsers.add_parser(
        "audiobook",
        help="Generate audiobook from text",
        description="Generate audiobook with chapters from text or markdown",
    )
    _add_audiobook_args(audiobook_parser)

    # Dialogue command
    dialogue_parser = subparsers.add_parser(
        "dialogue",
        help="Generate two-person conversation",
        description="Generate dialogue audio with two speakers",
    )
    _add_dialogue_args(dialogue_parser)

    # Voice management
    voice_parser = subparsers.add_parser(
        "voice",
        help="Manage voice profiles",
        description="Create, list, and manage voice profiles",
    )
    _add_voice_args(voice_parser)

    # Quick TTS
    say_parser = subparsers.add_parser(
        "say",
        help="Quick text-to-speech",
        description="Quickly generate speech from text",
    )
    _add_say_args(say_parser)

    # Pipeline command (long-form TTS)
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Long-form TTS with chunking and stitching",
        description="Generate long-form TTS with voice cloning, chunking, and stitching",
    )
    _add_pipeline_args(pipeline_parser)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Dispatch to command handler
    try:
        if args.command == "voiceover":
            _run_voiceover(args)
        elif args.command == "podcast":
            _run_podcast(args)
        elif args.command == "audiobook":
            _run_audiobook(args)
        elif args.command == "dialogue":
            _run_dialogue(args)
        elif args.command == "voice":
            _run_voice(args)
        elif args.command == "say":
            _run_say(args)
        elif args.command == "pipeline":
            _run_pipeline(args)
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def _add_common_args(parser: argparse.ArgumentParser):
    """Add common arguments to a parser."""
    parser.add_argument(
        "--model",
        "-m",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Model name or path",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="cpu",
        help="Device: cpu, cuda:0, mps",
    )
    parser.add_argument(
        "--language",
        "-l",
        default="Auto",
        help="Language: Auto, English, Chinese, etc.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--backend",
        default="qwen",
        choices=["qwen", "mock"],
        help="TTS backend to use",
    )


def _create_backend(args):
    """Create TTS backend from args."""
    if args.backend == "mock":
        from .backends import MockBackend
        return MockBackend()
    else:
        try:
            from .backends import QwenBackend
            return QwenBackend(
                model_name=args.model,
                device=args.device,
            )
        except ImportError:
            print("Warning: qwen-tts not available, using MockBackend")
            from .backends import MockBackend
            return MockBackend()


def _add_voiceover_args(parser: argparse.ArgumentParser):
    """Add voiceover-specific arguments."""
    parser.add_argument("input", help="Input text file")
    parser.add_argument("--output", "-o", required=True, help="Output audio file")
    parser.add_argument("--voice", "-v", help="Voice profile name")
    parser.add_argument("--ref-audio", "-a", help="Reference audio file")
    parser.add_argument("--ref-text", "-r", help="Reference audio transcript")
    parser.add_argument("--srt", help="SRT file for timing sync")
    _add_common_args(parser)


def _add_podcast_args(parser: argparse.ArgumentParser):
    """Add podcast-specific arguments."""
    parser.add_argument("input", help="Input script file")
    parser.add_argument("--output", "-o", required=True, help="Output audio file")
    parser.add_argument("--host", help="Host voice profile or ref audio")
    parser.add_argument("--host-text", help="Host reference transcript")
    parser.add_argument("--guest", help="Guest voice profile or ref audio")
    parser.add_argument("--guest-text", help="Guest reference transcript")
    parser.add_argument("--intro", help="Intro music file")
    parser.add_argument("--outro", help="Outro music file")
    parser.add_argument("--background", help="Background music file")
    parser.add_argument("--background-volume", type=float, default=-18, help="Background volume (dB)")
    _add_common_args(parser)


def _add_audiobook_args(parser: argparse.ArgumentParser):
    """Add audiobook-specific arguments."""
    parser.add_argument("input", help="Input text/markdown file")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--voice", "-v", help="Voice profile name")
    parser.add_argument("--ref-audio", "-a", help="Reference audio file")
    parser.add_argument("--ref-text", "-r", help="Reference audio transcript")
    parser.add_argument("--format", choices=["markdown", "plain", "numbered"], default="markdown")
    parser.add_argument("--no-combine", action="store_true", help="Don't create combined file")
    _add_common_args(parser)


def _add_dialogue_args(parser: argparse.ArgumentParser):
    """Add dialogue-specific arguments."""
    parser.add_argument("input", help="Input dialogue file")
    parser.add_argument("--output", "-o", required=True, help="Output audio file")
    parser.add_argument("--s1", help="Speaker 1 voice profile or ref audio")
    parser.add_argument("--s1-text", help="Speaker 1 reference transcript")
    parser.add_argument("--s2", help="Speaker 2 voice profile or ref audio")
    parser.add_argument("--s2-text", help="Speaker 2 reference transcript")
    _add_common_args(parser)


def _add_voice_args(parser: argparse.ArgumentParser):
    """Add voice management arguments."""
    voice_sub = parser.add_subparsers(dest="voice_command")

    # Create voice
    create_parser = voice_sub.add_parser("create", help="Create a voice profile")
    create_parser.add_argument("name", help="Voice profile name")
    create_parser.add_argument("--audio", "-a", required=True, help="Reference audio file")
    create_parser.add_argument("--text", "-t", required=True, help="Reference transcript")
    create_parser.add_argument("--description", "-d", default="", help="Profile description")

    # List voices
    voice_sub.add_parser("list", help="List voice profiles")

    # Delete voice
    delete_parser = voice_sub.add_parser("delete", help="Delete a voice profile")
    delete_parser.add_argument("name", help="Voice profile name")

    # Export voice
    export_parser = voice_sub.add_parser("export", help="Export a voice profile")
    export_parser.add_argument("name", help="Voice profile name")
    export_parser.add_argument("--output", "-o", required=True, help="Output directory")


def _add_say_args(parser: argparse.ArgumentParser):
    """Add quick TTS arguments."""
    parser.add_argument("text", help="Text to speak")
    parser.add_argument("--output", "-o", default="output.wav", help="Output audio file")
    parser.add_argument("--voice", "-v", help="Voice profile name")
    parser.add_argument("--ref-audio", "-a", help="Reference audio file")
    parser.add_argument("--ref-text", "-r", help="Reference audio transcript")
    parser.add_argument("--emotion", "-e", help="Emotion preset")
    _add_common_args(parser)


def _add_pipeline_args(parser: argparse.ArgumentParser):
    """Add pipeline TTS arguments."""
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text-file", "-f", help="Input text file")
    input_group.add_argument("--text", "-t", help="Direct text input")
    parser.add_argument("--ref-audio", "-a", required=True, help="Reference audio file")
    parser.add_argument("--ref-text", "-r", required=True, help="Reference transcript")
    parser.add_argument("--output", "-o", required=True, help="Output audio file")
    parser.add_argument("--work-dir", "-w", help="Working directory")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")
    parser.add_argument("--chunk-min", type=int, default=100)
    parser.add_argument("--chunk-max", type=int, default=300)
    parser.add_argument("--chunk-target", type=int, default=200)
    parser.add_argument("--crossfade-ms", type=int, default=75)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    _add_common_args(parser)


def _run_voiceover(args):
    """Run voiceover generation."""
    from .formats.voiceover import VoiceoverHandler
    from .voices.registry import VoiceRegistry

    print(f"Generating voiceover from: {args.input}")

    # Setup backend
    backend = _create_backend(args)
    handler = VoiceoverHandler(backend=backend)

    # Read input
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    # Get voice reference
    ref_audio = args.ref_audio
    ref_text = args.ref_text

    if args.voice and not ref_audio:
        registry = VoiceRegistry()
        profile = registry.get(args.voice)
        ref_audio = profile.reference_audio
        ref_text = profile.reference_text

    if not ref_audio or not ref_text:
        print("Error: Must provide --ref-audio and --ref-text, or --voice")
        sys.exit(1)

    # Generate
    output = handler.process(
        input_text=text,
        output_path=args.output,
        ref_audio=ref_audio,
        ref_text=ref_text,
        language=args.language,
        temperature=args.temperature,
    )

    print(f"Output saved to: {args.output}")
    print(f"Duration: {output.duration_sec:.1f}s")


def _run_podcast(args):
    """Run podcast generation."""
    from .formats.podcast import PodcastHandler
    from .voices.registry import VoiceRegistry

    print(f"Generating podcast from: {args.input}")

    # Setup backend
    backend = _create_backend(args)
    handler = PodcastHandler(
        backend=backend,
        intro_music=args.intro,
        outro_music=args.outro,
        background_music=args.background,
        background_volume_db=args.background_volume,
    )

    # Build speaker refs
    speaker_refs = {}
    registry = VoiceRegistry()

    # Host voice
    if args.host:
        if args.host_text:
            speaker_refs["HOST"] = (args.host, args.host_text)
        elif args.host in registry:
            profile = registry.get(args.host)
            speaker_refs["HOST"] = (profile.reference_audio, profile.reference_text)

    # Guest voice
    if args.guest:
        if args.guest_text:
            speaker_refs["GUEST"] = (args.guest, args.guest_text)
        elif args.guest in registry:
            profile = registry.get(args.guest)
            speaker_refs["GUEST"] = (profile.reference_audio, profile.reference_text)

    output = handler.generate_episode(
        script_path=args.input,
        output_path=args.output,
        host_ref=speaker_refs.get("HOST"),
        guest_ref=speaker_refs.get("GUEST"),
        language=args.language,
        temperature=args.temperature,
    )

    print(f"Output saved to: {args.output}")
    print(f"Duration: {output.duration_sec:.1f}s")


def _run_audiobook(args):
    """Run audiobook generation."""
    from .formats.audiobook import AudiobookHandler
    from .voices.registry import VoiceRegistry

    print(f"Generating audiobook from: {args.input}")

    # Setup backend
    backend = _create_backend(args)
    handler = AudiobookHandler(backend=backend)

    # Get voice reference
    ref_audio = args.ref_audio
    ref_text = args.ref_text

    if args.voice and not ref_audio:
        registry = VoiceRegistry()
        profile = registry.get(args.voice)
        ref_audio = profile.reference_audio
        ref_text = profile.reference_text

    if not ref_audio or not ref_text:
        print("Error: Must provide --ref-audio and --ref-text, or --voice")
        sys.exit(1)

    result = handler.generate_book(
        input_path=args.input,
        output_dir=args.output_dir,
        ref_audio=ref_audio,
        ref_text=ref_text,
        format=args.format,
        combine=not args.no_combine,
        language=args.language,
        temperature=args.temperature,
    )

    print(f"Generated {result['total_chapters']} chapters")
    print(f"Total duration: {result['total_duration_sec'] / 60:.1f} minutes")
    if result['combined_path']:
        print(f"Combined file: {result['combined_path']}")


def _run_dialogue(args):
    """Run dialogue generation."""
    from .formats.dialogue import DialogueHandler
    from .voices.registry import VoiceRegistry

    print(f"Generating dialogue from: {args.input}")

    # Setup backend
    backend = _create_backend(args)
    handler = DialogueHandler(backend=backend)

    # Read input
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    # Parse segments
    segments = handler.parse(text)
    speakers = handler.detect_speakers(segments)

    # Build speaker refs
    speaker_refs = {}
    registry = VoiceRegistry()

    # Map S1/S2 or first two detected speakers
    speaker_args = [
        (args.s1, args.s1_text, speakers[0] if speakers else "S1"),
        (args.s2, args.s2_text, speakers[1] if len(speakers) > 1 else "S2"),
    ]

    for voice, text_arg, speaker_id in speaker_args:
        if voice:
            if text_arg:
                speaker_refs[speaker_id] = (voice, text_arg)
            elif voice in registry:
                profile = registry.get(voice)
                speaker_refs[speaker_id] = (profile.reference_audio, profile.reference_text)

    output = handler.generate(
        segments=segments,
        output_path=args.output,
        speaker_refs=speaker_refs,
        language=args.language,
        temperature=args.temperature,
    )

    print(f"Output saved to: {args.output}")
    print(f"Duration: {output.duration_sec:.1f}s")
    print(f"Speakers: {', '.join(speakers)}")


def _run_voice(args):
    """Run voice management commands."""
    from .voices.registry import VoiceRegistry

    registry = VoiceRegistry()

    if args.voice_command == "create":
        profile = registry.create(
            name=args.name,
            reference_audio=args.audio,
            reference_text=args.text,
            description=args.description,
        )
        print(f"Created voice profile: {profile.name}")

    elif args.voice_command == "list":
        profiles = registry.list_detailed()
        if not profiles:
            print("No voice profiles found.")
        else:
            print("Voice profiles:")
            for p in profiles:
                print(f"  - {p['name']}: {p['description'] or 'No description'}")

    elif args.voice_command == "delete":
        if registry.delete(args.name):
            print(f"Deleted voice profile: {args.name}")
        else:
            print(f"Voice profile not found: {args.name}")

    elif args.voice_command == "export":
        output = registry.export(args.name, args.output)
        print(f"Exported to: {output}")

    else:
        print("Usage: tts-toolkit voice {create|list|delete|export}")


def _run_say(args):
    """Run quick TTS generation."""
    from .voices.registry import VoiceRegistry
    from .voices.emotions import apply_emotion
    import soundfile as sf

    # Get voice reference
    ref_audio = args.ref_audio
    ref_text = args.ref_text

    if args.voice and not ref_audio:
        registry = VoiceRegistry()
        profile = registry.get(args.voice)
        ref_audio = profile.reference_audio
        ref_text = profile.reference_text

    if not ref_audio or not ref_text:
        print("Error: Must provide --ref-audio and --ref-text, or --voice")
        sys.exit(1)

    # Setup backend
    backend = _create_backend(args)
    backend.load_model()

    # Create voice prompt
    voice_prompt = backend.create_voice_prompt(
        reference_audio=ref_audio,
        reference_text=ref_text,
    )

    # Apply emotion if specified
    gen_kwargs = {"temperature": args.temperature}
    if args.emotion:
        gen_kwargs = apply_emotion(gen_kwargs, args.emotion)

    # Generate
    audio, sr = backend.generate(
        text=args.text,
        voice_prompt=voice_prompt,
        language=args.language,
        **gen_kwargs,
    )

    sf.write(args.output, audio, sr)
    print(f"Output saved to: {args.output}")


def _run_pipeline(args):
    """Run pipeline long-form TTS."""
    from .core.pipeline import Pipeline

    # Get text
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text

    # Create backend
    backend = _create_backend(args)

    # Create pipeline
    pipeline = Pipeline(
        backend=backend,
        chunk_min=args.chunk_min,
        chunk_max=args.chunk_max,
        chunk_target=args.chunk_target,
        crossfade_ms=args.crossfade_ms,
    )

    # Progress display
    try:
        from tqdm import tqdm
        pbar = None

        def progress_callback(current, total, text):
            nonlocal pbar
            if pbar is None:
                pbar = tqdm(total=total, desc="Generating", unit="chunk")
            pbar.n = current
            pbar.refresh()
    except ImportError:
        def progress_callback(current, total, text):
            print(f"Progress: {current}/{total} chunks")

    resume = not args.no_resume

    output_path = pipeline.process(
        text=text,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        output_path=args.output,
        work_dir=args.work_dir,
        language=args.language,
        resume=resume,
        progress_callback=progress_callback,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    print(f"\nDone! Output saved to: {output_path}")


if __name__ == "__main__":
    main()
