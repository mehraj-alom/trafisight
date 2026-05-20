from pathlib import Path
import cv2
import numpy as np
def generate_color_map(num_classes: int):
    #deterministic color map using HSV -> BGR
    colors = []
    for i in range(num_classes):
        hue = int(180.0 * i / max(1, num_classes))
        color = np.uint8([[[hue, 200, 255]]])
        bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0].tolist()
        colors.append(tuple(int(c) for c in bgr))
    return colors

