# inference_pipeline
from ingestion.stream_manager import StreamManager
from logger import main_logger as logger
def run_pipeline(source_type, source):
    try : 
        stream_manager = StreamManager(source_type, source)
        logger.info(f"StreamManager initialized successfully for source: {source}")
    except Exception as e:
        logger.error(f"Error initializing StreamManager: {e}")
        return
    for i ,frame in enumerate(stream_manager.get_frames()):
        if (i % 4 == 0):
            if frame is None:
                logger.info("No more frames to process.")
                break
            print("Processing frame...")
            i += 1
            print(f"Processed {i} frames.")
 

# for Testing only (- Main will be in the api )
def main():
    video_path = 'video.mp4' 
    run_pipeline('video', video_path)
    
if __name__ == "__main__":    
    main()
        