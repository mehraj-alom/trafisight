import cv2
from config.global_config import GlobalConfig
from pipeline.inference_utils import generate_color_map 


def draw_detections(frame, detections, single_color=None, color_map=None, class_names=None):
    """Draw detections on frame.

    Args:
        frame: BGR image to draw on.
        detections: list of {"box": [x,y,w,h], "score": float, "class_id": int}.
        single_color: BGR tuple to use for all boxes (overrides color_map).
        color_map: list or dict mapping class_id -> BGR tuple.
        class_names: optional list of class names for labels. If None, uses GlobalConfig.CLASS_NAMES.
    """
    if class_names is None:
        class_names = GlobalConfig.CLASS_NAMES

    num_classes = len(class_names) if isinstance(class_names, (list, tuple)) else 0
    generated_map = generate_color_map(num_classes) if num_classes > 0 else []

    for det in detections:
        x, y, w, h = det.get("box", [0, 0, 0, 0])
        score = det.get("score", 0.0)
        class_id = int(det.get("class_id", 0))

        if single_color is not None:
            color = single_color
        elif color_map is not None:
            if isinstance(color_map, dict):
                color = color_map.get(class_id, (0, 220, 255))
            elif isinstance(color_map, (list, tuple)) and class_id < len(color_map):
                color = color_map[class_id]
            else:
                color = (0, 220, 255)
        else:
            color = (
                generated_map[class_id]
                if 0 <= class_id < len(generated_map)
                else (0, 220, 255)
            )

        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"cls_{class_id}"
        label = f"{class_name}: {score:.2f}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
