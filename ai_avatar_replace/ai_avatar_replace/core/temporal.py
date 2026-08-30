from __future__ import annotations

import cv2
import numpy as np


class ExponentialSmoother:
    """Exponential moving average for numpy arrays (masks, keypoints)."""

    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
        self._state: np.ndarray | None = None

    def reset(self) -> None:
        self._state = None

    def update(self, value: np.ndarray) -> np.ndarray:
        if self._state is None:
            self._state = value.astype(np.float32)
            return value
        self._state = self.alpha * value.astype(np.float32) + (1.0 - self.alpha) * self._state
        return self._state.astype(value.dtype)


def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    k = radius * 2 + 1
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
    return np.clip(blurred, 0, 255).astype(np.uint8)
