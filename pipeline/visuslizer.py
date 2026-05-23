import cv2
from config.global_config import GlobalConfig




def draw_detections(
    frame,
    detections,
    single_color=None,
    color_map=None,
    class_names=None,
):
    """
    Draw tracked detections from supervision ByteTrack.

    Args:
        frame: BGR frame
        detections: sv.Detections object
        single_color: optional BGR color for all classes
        color_map: dict/list of class colors
        class_names: optional class name list
    """

    if detections is None:
        return frame

    if class_names is None:
        class_names = GlobalConfig.CLASS_NAMES

    for box, tracker_id, conf, class_id in zip(
        detections.xyxy,
        detections.tracker_id,
        detections.confidence,
        detections.class_id,
    ):

        x1, y1, x2, y2 = map(int, box)

        class_id = int(class_id)

        # ---------- COLOR ----------
        if single_color is not None:
            color = single_color

        elif color_map is not None:

            if isinstance(color_map, dict):
                color = color_map.get(
                    class_id,
                    (0, 220, 255)
                )

            elif (
                isinstance(color_map, (list, tuple))
                and class_id < len(color_map)
            ):
                color = color_map[class_id]

            else:
                color = (0, 220, 255)

        else:
            color = (0, 255, 0)

        # ---------- CLASS NAME ----------
        if 0 <= class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"class_{class_id}"

        # ---------- TRACK ID ----------
        tracker_text = (
            f"ID {tracker_id}"
            if tracker_id is not None
            else "ID ?"
        )

        # ---------- LABEL ----------
        label = (
            f"{class_name} | "
            f"{tracker_text} | "
            f"{conf:.2f}"
        )

        # ---------- BOX ----------
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        # ---------- TEXT ----------
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return frame