#!/usr/bin/env python3
"""Gradio web UI for AI Avatar Video Replacement."""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from ai_avatar_replace.config import AppConfig
from ai_avatar_replace.pipeline import AvatarReplacementPipeline


def run(video_file, background_image, avatar_image, avatar_prompt, voice, seed, progress=gr.Progress()):
    if video_file is None:
        raise gr.Error("Upload a source video.")
    if background_image is None:
        raise gr.Error("Upload or generate a background image.")
    if not avatar_image and not avatar_prompt:
        raise gr.Error("Provide an avatar reference image or a text prompt for the synthetic character.")

    config = AppConfig.load()
    pipeline = AvatarReplacementPipeline(config)

    out_dir = Path(tempfile.mkdtemp(prefix="avr_"))
    output_path = out_dir / "result.mp4"

    progress(0, desc="Starting pipeline…")
    result = pipeline.process(
        input_video=video_file,
        output_video=output_path,
        background_image=background_image,
        avatar_image=avatar_image or None,
        avatar_prompt=avatar_prompt or None,
        voice=voice,
        seed=int(seed),
    )
    progress(1.0, desc="Complete")
    return str(result.output_path), f"Processed {result.frames_processed} frames on {pipeline.device}."


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI Avatar Video Replacement", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # AI Avatar Video Replacement
            Upload a video and this tool will replace the person with a **synthetic AI avatar**,
            swap in a **new background**, and re-voice the audio with a **synthetic voice**.

            The avatar is a generated character, not a copy of a real person — describe one with
            a text prompt, or upload a reference image of a character you already have.

            Outputs are AI-generated media; disclose that when you share them.
            """
        )

        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Source video (motion only is used)")
                background_input = gr.Image(type="filepath", label="New background image")
                with gr.Tab("Describe a character"):
                    avatar_prompt = gr.Textbox(label="Avatar description", lines=2,
                                                placeholder="a friendly animated-style synthetic news anchor")
                with gr.Tab("Upload a character"):
                    avatar_image = gr.Image(type="filepath", label="Synthetic character reference image")
                voice = gr.Textbox(label="Voice name (optional)", value="default")
                seed = gr.Number(label="Seed", value=42, precision=0)
                run_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                video_output = gr.Video(label="Result")
                status = gr.Textbox(label="Status", interactive=False)

        run_btn.click(
            fn=run,
            inputs=[video_input, background_input, avatar_image, avatar_prompt, voice, seed],
            outputs=[video_output, status],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
