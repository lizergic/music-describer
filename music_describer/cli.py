import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="music-describer",
        description="Analyze music and generate natural-language descriptions",
    )
    parser.add_argument("audio_file", help="Path to audio file (mp3, wav, flac)")
    parser.add_argument(
        "--json", action="store_true", help="Output full JSON (analysis + description)"
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Output structured analysis only (no LLM call)",
    )
    parser.add_argument("--output", "-o", help="Save output to file instead of stdout")
    parser.add_argument("--config", help="Path to config.yaml file")
    parser.add_argument(
        "--analyzers",
        help=(
            "Comma-separated subset of analyzers to run "
            "(rhythm,harmony,timbre,structure,energy). Default: all."
        ),
    )

    args = parser.parse_args()

    from music_describer import analyze, describe

    selected = (
        [name.strip() for name in args.analyzers.split(",") if name.strip()]
        if args.analyzers
        else None
    )

    if args.analysis_only:
        result = analyze(args.audio_file, analyzers=selected)
        output = json.dumps(result, indent=2)
    else:
        result = describe(args.audio_file, config_path=args.config, analyzers=selected)
        if args.json:
            output = json.dumps(result, indent=2)
        else:
            output = result["description"]

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Output saved to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
