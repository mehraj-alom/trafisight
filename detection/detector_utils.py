import cv2
import numpy as np
from logger import inference_logger as logger



def preprocess_frame(frame: np.ndarray, input_shape) -> np.ndarray:
    """Preprocess frame for model input.
    Args:
        frame: BGR image as numpy array.
        input_shape: expected model input shape (N,C,H,W).
    Returns: Preprocessed image as numpy array of shape (1,C,H,W).
    """
    height = input_shape[2] if isinstance(input_shape[2], int) and input_shape[2] > 0 else 640
    width = input_shape[3] if isinstance(input_shape[3], int) and input_shape[3] > 0 else 640

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    logger.debug(f"Preprocessed frame to shape: {image.shape}")
    return image


def postprocess_detections(
    raw_output: np.ndarray,
    frame_shape,
    model_input_shape,
    conf_threshold: float = 0.40,
    iou_threshold: float = 0.45
):
    """Postprocess model output to extract detections.
    Args:
    
        raw_output: raw output from model inference.
        frame_shape: original frame shape (H,W,C).
        model_input_shape: model input shape (N,C,H,W).
        conf_threshold: confidence threshold to filter detections.
        iou_threshold: IoU threshold for non-max suppression.
    Returns: List of detections, each as dict with keys "box", "score", "class_id".
    """
    frame_h, frame_w = frame_shape[:2]
    input_h, input_w = model_input_shape

    output = np.squeeze(raw_output)
    if output.ndim == 1:
        return []

    # Typical YOLOv8 ONNX shape is [num_attrs, num_anchors].
    if output.shape[0] <= output.shape[1]:
        predictions = output.T
    else:
        predictions = output

    if predictions.shape[1] < 6:
        return []

    scale_x = frame_w / float(input_w)
    scale_y = frame_h / float(input_h)

    boxes = []
    scores = []
    class_ids = []

    for pred in predictions:
        class_scores = pred[4:]
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])
        if score < conf_threshold:
            continue

        x_center, y_center, width, height = pred[:4]
        x = int((x_center - width / 2) * scale_x)
        y = int((y_center - height / 2) * scale_y)
        w = int(width * scale_x)
        h = int(height * scale_y)

        # Clamp to frame bounds to avoid draw-time issues.
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = max(1, min(w, frame_w - x))
        h = max(1, min(h, frame_h - y))

        boxes.append([x, y, w, h])
        scores.append(score)
        class_ids.append(class_id)

    if not boxes:
        logger.debug("No detections with confidence threshold.")
        return []
    
    logger.info(f"Detections before NMS: {len(boxes)}")
    
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    if indices is None or len(indices) == 0:
        logger.debug("No detections remaining after NMS.")
        return []

    detections = []
    for idx in np.array(indices).flatten():
        detections.append(
            {
                "box": boxes[idx],
                "score": scores[idx],
                "class_id": class_ids[idx],
            }
        )
    logger.info(f"Detections after NMS: {len(detections)}")
    return detections
