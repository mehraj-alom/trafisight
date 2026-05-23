# inference_pipeline  
from pathlib import Path
from datetime import datetime
import cv2
from config.global_config import GlobalConfig
from ingestion.stream_manager import StreamManager

from logger import inference_logger as logger

from detection.detector import Detector

# from pipeline.inference_utils import ( ,
#                                       )


from pipeline.visuslizer import (draw_detections,
                                 )
from tracking.tracker import Tracker


def run_pipeline(
    source_type,
    source,
    model_path: str = GlobalConfig.VEHICLE_MODEL_ONNX,
    output_path: str | None = None,
    draw_single_color: tuple | None = None,
    draw_color_map: dict | list | None = None,
):
    try:
        detector = Detector(model_path)
    except Exception as e:
        logger.error(f"Error initializing Detector with model '{model_path}': {e}")
        return

    try : 
        stream_manager = StreamManager(source_type, source)
        logger.info(f"StreamManager initialized successfully for source: {source}")
    except Exception as e:
        logger.error(f"Error initializing StreamManager: {e}")
        return

    if output_path:
        output_file = Path(output_path)
    else:
        output_dir = GlobalConfig.OUTPUT_PATH
        output_dir.mkdir(parents=True, exist_ok=True)
        source_name = Path(str(source)).stem if isinstance(source, str) else f"source_{source}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{source_name}_inference_{timestamp}.mp4"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    processed_frames = 0
    saved_frames = 0
    latest_detections = []
    

    try:
        for i, frame in enumerate(stream_manager.get_frames()):
            if frame is None:
                logger.info("No more frames to process.")
                break

            if writer is None:
                height, width = frame.shape[:2]
                fps = 20.0
                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_file),
                    fourcc,
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    logger.error(f"Failed to open output video writer: {output_file}")
                    return
                logger.info(f"Saving inference video to: {output_file}")

            frame_to_save = frame.copy()

            if i % 4 == 0:
                raw_detections = detector.detect_vehicle(frame)
                i += 1      
                tracker = Tracker()      
                latest_detections = tracker.update(raw_detections)
            draw_detections(frame_to_save, latest_detections, single_color=draw_single_color, color_map=draw_color_map)

            cv2.putText(
                frame_to_save,
                f"Processed: {processed_frames} | Detections: {len(latest_detections)}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
            frame_to_save,
            f"FPS: {fps:.2f}",
            (12, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            )
            writer.write(frame_to_save)
            saved_frames += 1
            processed_frames += 1
    finally:
        if writer is not None:
            writer.release()

    logger.info(
        f"Inference completed. Processed {processed_frames} frames and saved {saved_frames} frames to {output_file}"
    )
            
 

# for Testing only (- Main will be in the api )
def main():
    video_path = 'video_ind.mp4' 
    run_pipeline('video', video_path)
    
if __name__ == "__main__":    
    main()
        