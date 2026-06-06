# inference_pipeline.py
from pathlib import Path
from datetime import datetime
import time

import cv2

from config.global_config import GlobalConfig
from detection.detector import Detector
from detection.plate_detector import PlateDetector
from ingestion.stream_manager import StreamManager
from logger import inference_logger as logger
from ocr.ocr_config import OCRConfig
from ocr.ocr_core import OCREngine
from ocr.ocr_utils import (
    crop_frame_by_bbox,
    iter_supervision_detections,
    select_best_plate_detection,
)
from pipeline.visuslizer import draw_detections
from tracking.tracker import Tracker
from tracking.tracker_memory import TrackMemory
from pipeline.inference_utils import draw_ocr_label
from mock_db.mock_db import MockDB


mock_db = MockDB()

DETECTION_INTERVAL = 2


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
        logger.info("Vehicle detector initialized.")
    except Exception as exc:
        logger.error(
            f"Failed to initialize detector with model "
            f"'{model_path}': {exc}"
        )
        return

    try:
        stream_manager = StreamManager(source_type, source)
        logger.info(f"Stream initialized: {source}")
    except Exception as exc:
        logger.error(f"Failed to initialize stream: {exc}")
        return

    if output_path:
        output_file = Path(output_path)
    else:
        output_dir = GlobalConfig.OUTPUT_PATH
        output_dir.mkdir(parents=True, exist_ok=True)

        source_name = (
            Path(str(source)).stem
            if isinstance(source, str)
            else f"source_{source}"
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_file = (
            output_dir /
            f"{source_name}_inference_{timestamp}.mp4"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    tracker = Tracker()
    plate_detector = PlateDetector()
    ocr_engine = OCREngine()

    track_memory = TrackMemory(
        cooldown_frames=OCRConfig.OCR_COOLDOWN_FRAMES,
        expiry_frames=OCRConfig.TRACK_EXPIRY_FRAMES,
        max_attempts=OCRConfig.MAX_OCR_ATTEMPTS_PER_TRACK,
    )

    writer = None

    processed_frames = 0
    saved_frames = 0

    latest_detections = []

    fps_display = 0.0
    prev_time = time.time()

    def process_track(frame, track_row, frame_index):

        track_id = track_row["track_id"]
        vehicle_box = track_row["bbox_xyxy"]

        x1, y1, x2, y2 = vehicle_box

        if (x2 - x1) < 80 or (y2 - y1) < 40:
            return

        entry = track_memory.register_track(
            track_id,
            frame_index,
        )

        if entry.ocr_result:
            draw_ocr_label(
                frame,
                vehicle_box,
                entry.ocr_result,
            )
            return

        if not track_memory.should_run_ocr(
            track_id,
            frame_index,
        ):
            return

        vehicle_crop = crop_frame_by_bbox(
            frame,
            vehicle_box,
        )

        if vehicle_crop is None:
            return

        try:
            plate_detections = (
                plate_detector.detect_plates(
                    vehicle_crop
                )
            )
        except Exception as exc:
            logger.warning(
                f"Plate detection failed "
                f"for track {track_id}: {exc}"
            )
            return

        best_plate = select_best_plate_detection(
            plate_detections
        )

        track_memory.mark_ocr_attempt(
            track_id,
            frame_index,
        )

        if best_plate is None:
            return

        plate_crop = crop_frame_by_bbox(
            vehicle_crop,
            best_plate.bbox_xyxy,
        )

        if plate_crop is None:
            return

        try:
            print(f"Running OCR for track {track_id}") # debugg 
            debug = vehicle_crop.copy()
            x1, y1, x2, y2 = best_plate.bbox_xyxy
            cv2.rectangle(
                debug,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2
            )
            import os
            os.makedirs(
                GlobalConfig.LOCAL_DEBUG_PATH,
                exist_ok=True,
            )
            cv2.imwrite(
                f"{GlobalConfig.LOCAL_DEBUG_PATH}/track_{track_id}.jpg",
                debug
            )
            ocr_text, ocr_confidence = (
                ocr_engine.recognize(
                    plate_crop
                )
            )
            print("OCR RESULT:", ocr_text)
            
            if ocr_text and ocr_confidence > 0.2: 
                db_result = mock_db.query_plate(ocr_text)
                print("DB RESULT:", db_result)                                   ### db result 
                
            os.makedirs(
                f"{GlobalConfig.LOCAL_DEBUG_PATH}/plates",
                exist_ok=True,
            )

            cv2.imwrite(
                f"{GlobalConfig.LOCAL_DEBUG_PATH}/plates/"
                f"frame_{frame_index}_track_{track_id}.jpg",      ####### Debugging output
                plate_crop,
            )
        except Exception as exc:
            logger.warning(
                f"OCR failed "
                f"for track {track_id}: {exc}"
            )
            return

        if ocr_text:

            track_memory.update_ocr_result(
                track_id=track_id,
                frame_index=frame_index,
                ocr_result=ocr_text,
                ocr_confidence=ocr_confidence,
            )

            draw_ocr_label(
                frame,
                vehicle_box,
                f"{ocr_text} "
                f"({ocr_confidence:.2f})",
            )

    try:

        for frame_index, frame in enumerate(
            stream_manager.get_frames()
        ):

            if frame is None:
                logger.info(
                    "No more frames available."
                )
                break

            processed_frames += 1

            current_time = time.time()
            fps_display = (
                1.0 /
                max(
                    current_time - prev_time,
                    1e-6,
                )
            )
            prev_time = current_time

            if writer is None:

                height, width = frame.shape[:2]

                source_fps = getattr(
                    stream_manager,
                    "fps",
                    20.0,
                )

                if source_fps <= 0:
                    source_fps = 20.0

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                writer = cv2.VideoWriter(
                    str(output_file),
                    fourcc,
                    source_fps,
                    (width, height),
                )

                if not writer.isOpened():
                    logger.error(
                        f"Failed to create "
                        f"video writer: "
                        f"{output_file}"
                    )
                    return

                logger.info(
                    f"Saving output to "
                    f"{output_file}"
                )

            frame_to_save = frame.copy()

            track_memory.expire_stale_tracks(
                frame_index
            )

            if frame_index % DETECTION_INTERVAL == 0:

                try:
                    raw_detections = (
                        detector.detect_vehicle(
                            frame
                        )
                    )

                    latest_detections = (
                        tracker.update(
                            raw_detections
                        )
                    )

                except Exception as exc:
                    logger.warning(
                        f"Detection failed "
                        f"at frame "
                        f"{frame_index}: {exc}"
                    )

            if latest_detections:

                for track_row in (
                    iter_supervision_detections(
                        latest_detections
                    )
                ):
                    process_track(
                        frame_to_save,
                        track_row,
                        frame_index,
                    )

            draw_detections(
                frame_to_save,
                latest_detections,
                single_color=draw_single_color,
                color_map=draw_color_map,
            )

            cv2.putText(
                frame_to_save,
                f"Processed: "
                f"{processed_frames}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame_to_save,
                f"Detections: "
                f"{len(latest_detections)}",
                (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame_to_save,
                f"FPS: "
                f"{fps_display:.2f}",
                (12, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame_to_save)
            saved_frames += 1

    except KeyboardInterrupt:
        logger.info(
            "Pipeline interrupted "
            "by user."
        )

    finally:

        if writer is not None:
            writer.release()

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    logger.info(
        f"Inference complete. "
        f"Processed={processed_frames}, "
        f"Saved={saved_frames}, "
        f"Output={output_file}"
    )


def main():
    video_path = "local2.mp4"
    run_pipeline(
        source_type="video",
        source=video_path,
    )

if __name__ == "__main__":
    main()