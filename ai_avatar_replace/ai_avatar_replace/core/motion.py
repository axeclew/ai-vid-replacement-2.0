from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from rembg import remove


@dataclass
class FrameMotion:
    """Pose landmarks + person mask for a single frame."""

    pose_landmarks: np.ndarray | None  # (33, 3) normalized x, y, visibility
    person_mask: np.ndarray            # (H, W) uint8, 255 = person
    image_height: int
    image_width: int

    def pose_pixels(self) -> np.ndarray | None:
        if self.pose_landmarks is None:
            return None
        pts = self.pose_landmarks.copy()
        pts[:, 0] *= self.image_width
        pts[:, 1] *= self.image_height
        return pts

    def bbox(self, margin: float = 0.1) -> tuple[int, int, int, int] | None:
        pts = self.pose_pixels()
        if pts is None:
            return None
        visible = pts[:, 2] > 0.5
        if not np.any(visible):
            return None
        xs, ys = pts[visible, 0], pts[visible, 1]
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        pad_x, pad_y = (x2 - x1) * margin, (y2 - y1) * margin
        return (
            max(0, int(x1 - pad_x)),
            max(0, int(y1 - pad_y)),
            min(self.image_width, int(x2 + pad_x)),
            min(self.image_height, int(y2 + pad_y)),
        )

    def skeleton_image(self) -> np.ndarray:
        """Render an OpenPose-style skeleton for ControlNet guidance."""
        canvas = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        pts = self.pose_pixels()
        if pts is None:
            return canvas
        for i, j in mp.solutions.pose.POSE_CONNECTIONS:
            if pts[i, 2] < 0.5 or pts[j, 2] < 0.5:
                continue
            p1, p2 = (int(pts[i, 0]), int(pts[i, 1])), (int(pts[j, 0]), int(pts[j, 1]))
            cv2.line(canvas, p1, p2, (255, 255, 255), 3, cv2.LINE_AA)
        for x, y, vis in pts:
            if vis < 0.5:
                continue
            cv2.circle(canvas, (int(x), int(y)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        return canvas


class MotionTracker:
    """Tracks body pose and segments the person, frame by frame.

    This only ever extracts *motion* (skeleton) and a *silhouette* (mask)
    from the source video — no facial identity or likeness data is kept or
    reused, since the output identity is a synthetic avatar, not the person
    in the source footage.
    """

    def __init__(self):
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
        )

    def process(self, frame_rgb: np.ndarray) -> FrameMotion:
        h, w = frame_rgb.shape[:2]
        results = self._pose.process(frame_rgb)

        landmarks = None
        if results.pose_landmarks:
            landmarks = np.array(
                [[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark],
                dtype=np.float32,
            )

        mask = self._segment(frame_rgb)
        return FrameMotion(pose_landmarks=landmarks, person_mask=mask, image_height=h, image_width=w)

    @staticmethod
    def _segment(frame_rgb: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(frame_rgb)
        out = remove(pil)
        return np.array(out.split()[-1])

    def close(self) -> None:
        self._pose.close()
