import numpy as np
import supervision as sv


def convert_to_sv_detections(detections):
    """
    Convert custom detections to supervision detections.
    """

    if not detections:
        return sv.Detections.empty()

    xyxy = []
    confidence = []
    class_id = []

    for det in detections:
        x, y, w, h = det["box"]

        xyxy.append([
            x,
            y,
            x + w,
            y + h
        ])

        confidence.append(det["score"])
        class_id.append(det["class_id"])

    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confidence, dtype=np.float32),
        class_id=np.array(class_id, dtype=np.int32),
    )