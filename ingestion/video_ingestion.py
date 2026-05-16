# ingestion of video files into the system
import cv2

def ingest_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()
    print("Video ingestion completed.")