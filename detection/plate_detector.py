from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# from config.global_config import GlobalConfig
from logger import inference_logger as logger
from ocr.ocr_config import OCRConfig
from ocr.ocr_utils import PlateDetection
from utils.global_utils import load_onnx_session


class PlateDetector:
    """License plate detector that runs inside a vehicle crop."""

    def __init__(
        self,
        model_path: str = OCRConfig.LICENSE_PLATE_MODEL_PATH,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.enabled = False
        self.session = None
        self.input_name = None
        self.input_shape = None

        try:
            if not Path(model_path).exists():
                logger.warning(f"License plate model not found yet: {model_path}")
                return

            self.session = load_onnx_session(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.enabled = True
            logger.info(f"Loaded plate detector model from: {model_path}")
        except Exception as exc:
            logger.warning(f"Plate detector disabled because model failed to load: {exc}")

    def detect_plates(self, frame) -> list[PlateDetection]:
        """Run plate detection on a vehicle crop and return local crop coordinates."""
        if not self.enabled or self.session is None or self.input_shape is None:
            return []

        try:
            input_tensor = self._preprocess_frame(frame)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            return self._postprocess_outputs(outputs[0], frame.shape, (input_tensor.shape[2], input_tensor.shape[3]))
        except Exception as exc:
            logger.debug(f"Plate detection failed: {exc}")
            return []

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        height = self.input_shape[2] if isinstance(self.input_shape[2], int) and self.input_shape[2] > 0 else 640
        width = self.input_shape[3] if isinstance(self.input_shape[3], int) and self.input_shape[3] > 0 else 640
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return np.expand_dims(image, axis=0)

    def _postprocess_outputs(self,raw_output: np.ndarray,frame_shape,model_input_shape,) -> list[PlateDetection]:
        frame_h, frame_w = frame_shape[:2]
        input_h, input_w = model_input_shape

        output = np.squeeze(raw_output)

        if output.ndim == 1:
            return []

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

        for pred in predictions:

            confidence = float(pred[4])

            if confidence < self.conf_threshold:
                continue

            x_center, y_center, width, height = pred[:4]

            x1 = int((x_center - width / 2) * scale_x)
            y1 = int((y_center - height / 2) * scale_y)
            x2 = int((x_center + width / 2) * scale_x)
            y2 = int((y_center + height / 2) * scale_y)

            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(0, min(x2, frame_w - 1))
            y2 = max(0, min(y2, frame_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append((x1, y1, x2, y2))
            scores.append(confidence)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.conf_threshold,
            self.iou_threshold,
        )

        if indices is None or len(indices) == 0:
            return []

        detections: list[PlateDetection] = []

        for idx in np.array(indices).flatten():

            print(
                "FINAL BOX:",
                boxes[idx],
                "CONF:",
                scores[idx]
            )

            print(
                "Vehicle crop shape:",
                frame_shape
            )

            detections.append(
                PlateDetection(
                    bbox_xyxy=tuple(boxes[idx]),
                    confidence=float(scores[idx]),
                )
            )

        return detections