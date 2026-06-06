from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class PlateDetection:
    """Single plate detection in xyxy coordinates."""

    bbox_xyxy: tuple[int, int, int, int]
    confidence: float


def clip_bbox_xyxy(
    bbox_xyxy: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    """Clip a box to frame bounds and drop empty results."""
    if len(frame_shape) < 2:
        return None

    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = bbox_xyxy

    x1 = max(0, min(int(x1), frame_w - 1))
    y1 = max(0, min(int(y1), frame_h - 1))
    x2 = max(0, min(int(x2), frame_w - 1))
    y2 = max(0, min(int(y2), frame_h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def crop_frame_by_bbox(
    frame: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Crop a region safely for vehicle/plate extraction."""
    clipped = clip_bbox_xyxy(bbox_xyxy, frame.shape)
    if clipped is None:
        return None

    x1, y1, x2, y2 = clipped
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return crop


def translate_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    offset_xyxy: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Translate a crop-relative bbox back into full-frame coordinates."""
    offset_x1, offset_y1, _, _ = offset_xyxy
    x1, y1, x2, y2 = bbox_xyxy
    return x1 + offset_x1, y1 + offset_y1, x2 + offset_x1, y2 + offset_y1


def iter_supervision_detections(detections):
    """Yield normalized detection rows from a supervision Detections object."""
    if detections is None:
        return

    xyxy = getattr(detections, "xyxy", None)
    tracker_ids = getattr(detections, "tracker_id", None)
    confidences = getattr(detections, "confidence", None)
    class_ids = getattr(detections, "class_id", None)

    if xyxy is None or tracker_ids is None:
        return

    total = min(len(xyxy), len(tracker_ids))
    for index in range(total):
        tracker_id = tracker_ids[index]
        if tracker_id is None:
            continue

        yield {
            "track_id": int(tracker_id),
            "bbox_xyxy": tuple(int(value) for value in np.array(xyxy[index]).tolist()),
            "confidence": float(confidences[index]) if confidences is not None else 0.0,
            "class_id": int(class_ids[index]) if class_ids is not None else -1,
        }


def select_best_plate_detection(plate_detections) -> PlateDetection | None:
    """Pick the best plate candidate for OCR, usually the highest-confidence box."""
    if not plate_detections:
        return None

    best_detection = max(plate_detections, key=lambda item: item.confidence)
    return best_detection


def draw_text(frame, text: str, origin: tuple[int, int], color=(0, 215, 255)) -> None:
    """Small OpenCV text helper for realtime overlays."""
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
