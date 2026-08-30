from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    frame_count: int


class VideoReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "VideoReader":
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video: {self.path}")
        self.meta = VideoMeta(
            fps=self._cap.get(cv2.CAP_PROP_FPS) or 30.0,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        return self

    def read_frames(self) -> Iterator[np.ndarray]:
        assert self._cap is not None
        while True:
            ok, frame_bgr = self._cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def __exit__(self, *exc) -> None:
        if self._cap is not None:
            self._cap.release()


def resize_frame(frame: np.ndarray, max_edge: int) -> tuple[np.ndarray, float]:
    if max_edge <= 0:
        return frame, 1.0
    h, w = frame.shape[:2]
    scale = max_edge / max(h, w)
    if scale >= 1.0:
        return frame, 1.0
    return cv2.resize(frame, (int(w * scale), int(h * scale))), scale


def write_video(frames: list[np.ndarray], path: str | Path, fps: float, codec: str = "mp4v") -> None:
    if not frames:
        raise ValueError("No frames to write.")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def mux_audio(silent_video: str | Path, audio_source: str | Path, output_path: str | Path) -> None:
    """Combine a silent video with an audio track via ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(silent_video),
        "-i", str(audio_source),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def extract_audio(video_path: str | Path, audio_out: str | Path) -> Path:
    audio_out = Path(audio_out)
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return audio_out
