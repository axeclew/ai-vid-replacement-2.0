from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ai_avatar_replace.config import AppConfig
from ai_avatar_replace.core.avatar_render import AvatarRenderer, composite_background
from ai_avatar_replace.core.motion import MotionTracker
from ai_avatar_replace.core.temporal import ExponentialSmoother, feather_mask
from ai_avatar_replace.core.video_io import (
    VideoReader,
    extract_audio,
    mux_audio,
    resize_frame,
    write_video,
)
from ai_avatar_replace.core.voice import synthesize_speech, transcribe


@dataclass
class ReplacementResult:
    output_path: Path
    frames_processed: int


class AvatarReplacementPipeline:
    """
    Source video -> track motion & segment person -> render synthetic avatar
    in that pose -> composite onto new background -> replace voice -> mux.
    """

    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig.load()
        self.device = self.config.resolve_device()

    def process(
        self,
        input_video: str | Path,
        output_video: str | Path,
        background_image: str | Path,
        avatar_image: str | Path | None = None,
        avatar_prompt: str | None = None,
        voice: str = "default",
        seed: int = 42,
    ) -> ReplacementResult:
        if avatar_image is None and avatar_prompt is None:
            raise ValueError("Provide either avatar_image or avatar_prompt.")

        input_video = Path(input_video)
        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        silent_output = output_video.with_name(output_video.stem + "_silent.mp4")

        background_rgb = self._load_rgb(background_image)

        renderer = AvatarRenderer(
            device=self.device,
            sd_model=self.config.models.get("sd_inpaint", "runwayml/stable-diffusion-inpainting"),
            controlnet_model=self.config.models.get("controlnet_openpose", "lllyasviel/control_v11p_sd15_openpose"),
            ip_adapter_repo=self.config.models.get("ip_adapter", "h94/IP-Adapter"),
        )
        if avatar_image is not None:
            renderer.set_avatar_from_image(avatar_image)
        else:
            renderer.generate_avatar_from_prompt(avatar_prompt, seed=seed)

        pos_prompt, neg_prompt = AvatarRenderer.default_prompts(avatar_prompt or "a photorealistic synthetic person")

        tracker = MotionTracker()
        mask_smoother = ExponentialSmoother(alpha=self.config.smoothing_alpha)

        with VideoReader(input_video) as reader:
            fps = reader.meta.fps
            frames = list(reader.read_frames())

        processed_frames: list[np.ndarray] = []
        for idx, frame in enumerate(tqdm(frames, desc="Rendering avatar frames")):
            frame, _scale = resize_frame(frame, self.config.max_resolution)
            motion = tracker.process(frame)

            mask = mask_smoother.update(motion.person_mask)
            mask = feather_mask(mask, self.config.mask_feather)
            skeleton = motion.skeleton_image()

            if np.any(mask > 32):
                avatar_frame = renderer.render_frame(
                    frame_rgb=frame,
                    person_mask=mask,
                    skeleton_rgb=skeleton,
                    prompt=pos_prompt,
                    negative_prompt=neg_prompt,
                    steps=self.config.diffusion_steps,
                    seed=seed + idx,
                )
            else:
                avatar_frame = frame

            final = composite_background(avatar_frame, mask, background_rgb)
            processed_frames.append(final)

        tracker.close()
        write_video(processed_frames, silent_output, fps=fps, codec=self.config.output.get("codec", "mp4v"))

        new_audio_path = self._build_voice_track(input_video, voice)
        mux_audio(silent_output, new_audio_path, output_video)
        silent_output.unlink(missing_ok=True)

        return ReplacementResult(output_path=output_video, frames_processed=len(processed_frames))

    def _build_voice_track(self, input_video: Path, voice: str) -> Path:
        tmp_dir = Path(tempfile.mkdtemp(prefix="avr_audio_"))
        original_audio = extract_audio(input_video, tmp_dir / "original.wav")
        text = transcribe(original_audio, model_size=self.config.whisper_model)
        new_audio = synthesize_speech(
            text=text,
            output_path=tmp_dir / "synthetic_voice.wav",
            engine=self.config.tts_engine,
            voice=voice,
        )
        return new_audio

    @staticmethod
    def _load_rgb(path: str | Path) -> np.ndarray:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Cannot read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
