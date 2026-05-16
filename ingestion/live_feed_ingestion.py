# Ingestion of live video feed into the system
import cv2
def ingest_live_feed(camera_source=0):
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        yield frame

    cap.release()
    print("Live feed ingestion completed.")