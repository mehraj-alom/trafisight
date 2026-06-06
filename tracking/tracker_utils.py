import numpy as np
import supervision as sv


def convert_to_sv_detections(detections):
    """
    Convert a list of detection dictionaries to a supervision Detections object.
     Each detection dictionary should have the following keys:
        - "box": A list or tuple of [x, y, width, height]
        - "score": A float representing the confidence score of the detection
        - "class_id": An integer representing the class ID of the detected object  
    
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