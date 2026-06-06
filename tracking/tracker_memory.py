from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrackMemoryEntry:
    """State for one tracked vehicle and its OCR lifecycle."""
    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    last_ocr_frame: int = -1
    ocr_result: str | None = None
    ocr_confidence: float | None = None
    ocr_attempts: int = 0


class TrackMemory:
    """Frame-based OCR cache with duplicate-prevention and expiration."""
    def __init__(self, cooldown_frames: int, expiry_frames: int, max_attempts: int = 3) -> None:
        self.cooldown_frames = max(1, int(cooldown_frames))
        self.expiry_frames = max(1, int(expiry_frames))
        self.max_attempts = max(1, int(max_attempts))
        self._entries: dict[int, TrackMemoryEntry] = {}

    def register_track(self, track_id: int, frame_index: int) -> TrackMemoryEntry:
        """Create or refresh a track entry.
        If the previous entry aged out, this creates a fresh record so reused IDs
        are treated like a new vehicle.
        """
        track_id = int(track_id)
        entry = self._entries.get(track_id)
        if entry is None or self._outdated(entry, frame_index):
            entry = TrackMemoryEntry(
                track_id=track_id,
                first_seen_frame=frame_index,
                last_seen_frame=frame_index,
            )
            self._entries[track_id] = entry
            return entry
        entry.last_seen_frame = frame_index
        return entry

    def should_run_ocr(self, track_id: int, frame_index: int) -> bool:
        """Return False when OCR should be skipped for a still-active track."""
        entry = self._entries.get(int(track_id))
        if entry is None:
            return True

        if self._outdated(entry, frame_index):
            return True

        # Duplicate prevention: if we already have a valid OCR result, reuse it.
        if entry.ocr_result:
            return False

        # Cooldown prevents OCR retries on consecutive frames for the same track.
        if entry.last_ocr_frame >= 0 and frame_index - entry.last_ocr_frame < self.cooldown_frames:
            return False
        if entry.ocr_attempts >= self.max_attempts:
            return False
        return True
    def mark_ocr_attempt(self, track_id: int, frame_index: int) -> None:
        entry = self._entries.get(int(track_id))
        if entry is None:
            return

        entry.last_ocr_frame = frame_index
        entry.ocr_attempts += 1
    def update_ocr_result(
        self,
        track_id: int,
        frame_index: int,
        ocr_result: str,
        ocr_confidence: float | None = None,
    ) -> None:
        entry = self.register_track(track_id, frame_index)
        text = ocr_result.strip()
        if not text:
            return
        entry.ocr_result = text
        entry.ocr_confidence = float(ocr_confidence) if ocr_confidence is not None else None
        entry.last_ocr_frame = frame_index

    def get_cached_result(self, track_id: int) -> str | None:
        entry = self._entries.get(int(track_id))
        if entry is None:
            return None
        return entry.ocr_result

    def get_last_seen_frame(self, track_id: int) -> int | None:
        entry = self._entries.get(int(track_id))
        if entry is None:
            return None
        return entry.last_seen_frame

    def expire_stale_tracks(self, frame_index: int) -> list[int]:
        expired_track_ids: list[int] = []
        for track_id, entry in list(self._entries.items()):
            if self._outdated(entry, frame_index):
                expired_track_ids.append(track_id)
                del self._entries[track_id]
        return expired_track_ids

    def _outdated(self, entry: TrackMemoryEntry, frame_index: int) -> bool:
        is_outdated = frame_index - entry.last_seen_frame > self.expiry_frames
        return is_outdated
