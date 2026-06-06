from pathlib import Path
import cv2
import numpy as np
from ocr.ocr_utils import draw_text

def generate_color_map(num_classes: int):
    #deterministic color map using HSV -> BGR
    colors = []
    for i in range(num_classes):
        hue = int(180.0 * i / max(1, num_classes))
        color = np.uint8([[[hue, 200, 255]]])
        bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0].tolist()
        colors.append(tuple(int(c) for c in bgr))
    return colors

def draw_ocr_label(frame, box_xyxy, text):
    x1, y1, _, _ = box_xyxy
    draw_text(frame, f"OCR: {text}", (x1, max(40, y1 - 28)))