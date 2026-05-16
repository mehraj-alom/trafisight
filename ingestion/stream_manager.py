from .live_feed_ingestion import ingest_live_feed
from .video_ingestion import ingest_video

class StreamManager:
    """Manages video streams from various sources (video files, live feeds).
     It provides a unified interface for ingesting video data regardless of the source type.
    Attributes:
        source_type (str): The type of the video source ('video' or 'live').
        source (str or int): The path to the video file or the camera source index."""
    def __init__(self,source_type, source):
        self.source_type = source_type
        self.source = source
    def get_frames(self):
        """Yields video frames from the specified source based on the source type."""
        if self.source_type == 'video':
            return ingest_video(self.source)
        elif self.source_type == 'live':
            return ingest_live_feed(self.source)
        else:
            raise ValueError("Invalid source type. Use 'video' or 'live'.")

            