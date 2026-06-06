# test_onnx_output.py

import cv2
import numpy as np
from utils.global_utils import load_onnx_session

MODEL_PATH = "ocr/models/LP_DETECTOR.onnx"
IMAGE_PATH = "test_paddleocr.jpg"

session = load_onnx_session(MODEL_PATH)

input_meta = session.get_inputs()[0]
input_name = input_meta.name

print("\n===== MODEL INFO =====")
print("Input Name :", input_meta.name)
print("Input Shape:", input_meta.shape)
print("Input Type :", input_meta.type)

for out in session.get_outputs():
    print("\nOutput Name :", out.name)
    print("Output Shape:", out.shape)
    print("Output Type :", out.type)

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640, 640))
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

outputs = session.run(None, {input_name: img})

print("\n==============================")
print("NUMBER OF OUTPUTS:", len(outputs))
print("==============================")

output = outputs[0]

print("\nOUTPUT SHAPE:", output.shape)
print("OUTPUT DTYPE:", output.dtype)

print("\n===== MAX VALUES =====")
print("MAX X :", np.max(output[:, :, 0]))
print("MAX Y :", np.max(output[:, :, 1]))
print("MAX W :", np.max(output[:, :, 2]))
print("MAX H :", np.max(output[:, :, 3]))

print("\n===== MIN VALUES =====")
print("MIN X :", np.min(output[:, :, 0]))
print("MIN Y :", np.min(output[:, :, 1]))
print("MIN W :", np.min(output[:, :, 2]))
print("MIN H :", np.min(output[:, :, 3]))

print("\n===== CONFIDENCE COLUMNS =====")
print("COL 4 MIN:", np.min(output[:, :, 4]))
print("COL 4 MAX:", np.max(output[:, :, 4]))

print("COL 5 MIN:", np.min(output[:, :, 5]))
print("COL 5 MAX:", np.max(output[:, :, 5]))

flat = output.reshape(-1, output.shape[-1])

print("\n===== FIRST 20 PREDICTIONS =====")

for idx, row in enumerate(flat[:20]):
    print(f"{idx+1:02d}: {row}")

print("\n===== TOP 20 BY COLUMN 5 =====")

top_idx = np.argsort(flat[:, 5])[::-1][:20]

for rank, idx in enumerate(top_idx, start=1):
    print(
        f"{rank:02d}: "
        f"{flat[idx]}"
    )

print("\n===== TOP 20 BY COLUMN 4 =====")

top_idx = np.argsort(flat[:, 4])[::-1][:20]

for rank, idx in enumerate(top_idx, start=1):
    print(
        f"{rank:02d}: "
        f"{flat[idx]}"
    )