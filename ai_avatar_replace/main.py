#!/usr/bin/env python3
"""CLI for AI Avatar Video Replacement."""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_avatar_replace.config import AppConfig
from ai_avatar_replace.pipeline import AvatarReplacementPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replace the person in a video with a synthetic AI avatar, "
        "a new background, and a synthetic voice.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_video", type=Path, help="Source video path")
    parser.add_argument("background_image", type=Path, help="New background image")
    parser.add_argument("-o", "--output", type=Path, default=Path("output/result.mp4"))

    avatar_group = parser.add_mutually_exclusive_group(required=True)
    avatar_group.add_argument("--avatar-image", type=Path, help="Reference image of the synthetic character")
    avatar_group.add_argument("--avatar-prompt", type=str, help="Text prompt describing a new synthetic character")

    parser.add_argument("--voice", default="default", help="TTS voice name (engine-dependent)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AppConfig.load(args.config)
    pipeline = AvatarReplacementPipeline(config)

    print(f"Device: {pipeline.device}")
    print(f"Input: {args.input_video}")

    result = pipeline.process(
        input_video=args.input_video,
        output_video=args.output,
        background_image=args.background_image,
        avatar_image=args.avatar_image,
        avatar_prompt=args.avatar_prompt,
        voice=args.voice,
        seed=args.seed,
    )
    print(f"Done — wrote {result.frames_processed} frames to {result.output_path}")


if __name__ == "__main__":
    main()
