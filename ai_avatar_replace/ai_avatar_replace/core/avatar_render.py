from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


class AvatarRenderer:
    """
    Renders a single fixed synthetic-character reference into the pose/shape
    of each frame, using ControlNet (OpenPose) for motion and IP-Adapter for
    appearance consistency. The character is either a user-supplied image or
    one generated once from a text prompt — never a real person's likeness.
    """

    def __init__(
        self,
        device: str = "cpu",
        sd_model: str = "runwayml/stable-diffusion-inpainting",
        controlnet_model: str = "lllyasviel/control_v11p_sd15_openpose",
        ip_adapter_repo: str = "h94/IP-Adapter",
    ):
        self.device = device
        self._dtype = torch.float16 if device != "cpu" else torch.float32
        self._pipe = None
        self._txt2img_pipe = None
        self._sd_model = sd_model
        self._controlnet_model = controlnet_model
        self._ip_adapter_repo = ip_adapter_repo
        self._ip_adapter_enabled = False
        self.avatar_reference: Image.Image | None = None

    # -- avatar reference -------------------------------------------------

    def set_avatar_from_image(self, path: str | Path) -> None:
        self.avatar_reference = Image.open(path).convert("RGB")

    def generate_avatar_from_prompt(self, prompt: str, seed: int = 0) -> Image.Image:
        """Generate a single reference image for a new synthetic character."""
        if self._txt2img_pipe is None:
            from diffusers import StableDiffusionPipeline

            self._txt2img_pipe = StableDiffusionPipeline.from_pretrained(
                self._sd_model.replace("-inpainting", ""), torch_dtype=self._dtype, safety_checker=None
            )
            self._txt2img_pipe.to(self.device if self.device != "cpu" else "cpu")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        full_prompt = (
            f"{prompt}, full body, neutral standing pose, studio lighting, "
            "photorealistic, highly detailed, plain background"
        )
        image = self._txt2img_pipe(full_prompt, generator=generator, num_inference_steps=30).images[0]
        self.avatar_reference = image
        return image

    # -- pipeline loading ---------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
        from transformers import CLIPVisionModelWithProjection

        controlnet = ControlNetModel.from_pretrained(self._controlnet_model, torch_dtype=self._dtype)
        self._pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            self._sd_model, controlnet=controlnet, torch_dtype=self._dtype, safety_checker=None
        )
        try:
            self._pipe.load_ip_adapter(self._ip_adapter_repo, subfolder="models", weight_name="ip-adapter_sd15.bin")
            self._pipe.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self._ip_adapter_repo, subfolder="models/image_encoder", torch_dtype=self._dtype
            )
            self._pipe.set_ip_adapter_scale(0.8)
            self._ip_adapter_enabled = True
        except Exception:
            self._ip_adapter_enabled = False

        if self.device == "cuda":
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe.to(self.device)

    # -- per-frame rendering --------------------------------------------

    def render_frame(
        self,
        frame_rgb: np.ndarray,
        person_mask: np.ndarray,
        skeleton_rgb: np.ndarray,
        prompt: str,
        negative_prompt: str,
        steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> np.ndarray:
        """Replace the masked person region with the synthetic avatar, posed
        to match `skeleton_rgb` for this frame."""
        if self.avatar_reference is None:
            raise RuntimeError("No avatar reference set. Call set_avatar_from_image() or generate_avatar_from_prompt().")
        self._ensure_loaded()

        image = Image.fromarray(frame_rgb)
        mask_pil = Image.fromarray(person_mask).convert("L")
        control = Image.fromarray(skeleton_rgb)
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None

        kwargs: dict = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_pil,
            control_image=control,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        if self._ip_adapter_enabled:
            kwargs["ip_adapter_image"] = self.avatar_reference

        result = self._pipe(**kwargs).images[0]
        return np.array(result)

    @staticmethod
    def default_prompts(appearance_hint: str) -> tuple[str, str]:
        positive = (
            f"{appearance_hint}, consistent synthetic character, natural skin texture, "
            "realistic lighting, sharp details, cinematic quality, consistent anatomy"
        )
        negative = (
            "blurry, deformed, disfigured, bad anatomy, extra limbs, cartoon, "
            "painting, low quality, watermark, text, duplicate face, different person each frame"
        )
        return positive, negative


def composite_background(
    person_rgb: np.ndarray,
    person_mask: np.ndarray,
    background_rgb: np.ndarray,
) -> np.ndarray:
    """Paste the (avatar-rendered) person onto a new background using the mask."""
    bg = background_rgb
    if bg.shape[:2] != person_rgb.shape[:2]:
        import cv2

        bg = cv2.resize(bg, (person_rgb.shape[1], person_rgb.shape[0]))
    alpha = (person_mask.astype(np.float32) / 255.0)[..., None]
    out = person_rgb.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)
