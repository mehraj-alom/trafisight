# # test_plate_ocr_video.py

# import cv2
# import easyocr
# import onnxruntime as ort

# VIDEO_PATH = "video_ind.mp4"
# MODEL_PATH = "ocr/models/LP_DETECTOR.onnx"

# reader = easyocr.Reader(["en"], gpu=False)

# session = ort.InferenceSession(
#     MODEL_PATH,
#     providers=["CPUExecutionProvider"]
# )

# input_name = session.get_inputs()[0].name

# cap = cv2.VideoCapture(VIDEO_PATH)

# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)

# writer = cv2.VideoWriter(
#     "outputs/inference/ocr_debug.mp4",
#     cv2.VideoWriter_fourcc(*"mp4v"),
#     fps,
#     (width, height),
# )

# frame_no = 0

# while True:

#     ret, frame = cap.read()

#     if not ret:
#         break

#     frame_no += 1

#     cv2.putText(
#         frame,
#         f"OCR TEST FRAME {frame_no}",
#         (30, 50),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0, 255, 0),
#         2,
#     )

#     writer.write(frame)

# cap.release()
# writer.release()

# print("Saved outputs/inference/ocr_debug.mp4")


import cv2
import easyocr

IMAGE_PATH = "outputs/inference/ocr_test.jpg"

reader = easyocr.Reader(["en"], gpu=False)

img = cv2.imread(IMAGE_PATH)

results = reader.readtext(img)

print(results)

for item in results:
    bbox, text, conf = item

    print(
        f"Text={text} "
        f"Conf={conf:.2f}"
    )

    pts = [(int(x), int(y)) for x, y in bbox]

    for i in range(4):
        cv2.line(
            img,
            pts[i],
            pts[(i + 1) % 4],
            (0, 255, 0),
            2,
        )
    
        cv2.putText(
            img,
            text,
            pts[0],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imwrite(
            "outputs/inference/ocr_test_check.jpg",
            img,
        )

