import supervision as sv
from .tracker_utils import convert_to_sv_detections


tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=60,
    minimum_matching_threshold=0.6,
    frame_rate=20,
)

class Tracker:
    def __init__(self):
        self.tracker = tracker

    def update(self, raw_detections):
        sv_detections = convert_to_sv_detections(raw_detections)

        tracked_detections = self.tracker.update_with_detections(
            sv_detections
        )
        latest_detections = tracked_detections
        
        return latest_detections
