from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


@dataclass
class AppConfig:
    device: str = "auto"
    max_resolution: int = 1024
    mask_feather: int = 15
    smoothing_alpha: float = 0.6
    diffusion_steps: int = 20
    tts_engine: str = "coqui"          # "coqui" | "system"
    tts_voice: str = "default"
    whisper_model: str = "base"
    models: dict[str, str] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        config_path = path or DEFAULT_CONFIG_PATH
        raw: dict[str, Any] = {}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        return cls(
            device=raw.get("device", "auto"),
            max_resolution=raw.get("max_resolution", 1024),
            mask_feather=raw.get("mask_feather", 15),
            smoothing_alpha=raw.get("smoothing_alpha", 0.6),
            diffusion_steps=raw.get("diffusion_steps", 20),
            tts_engine=raw.get("tts_engine", "coqui"),
            tts_voice=raw.get("tts_voice", "default"),
            whisper_model=raw.get("whisper_model", "base"),
            models=raw.get("models") or {},
            output=raw.get("output") or {},
        )

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
