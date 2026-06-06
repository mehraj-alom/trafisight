from __future__ import annotations


class OCRConfig:
    """OCR and plate-detection settings for the inference pipeline."""

    # Placeholder path for the license plate detector model.
    LICENSE_PLATE_MODEL_PATH = "ocr/models/LP_DETECTOR.onnx"

    # Do not re-run OCR on the same active track within this many frames.
    OCR_COOLDOWN_FRAMES = 300

    # Remove track memory after this many frames without seeing the track again.
    TRACK_EXPIRY_FRAMES = 600

    # Keep OCR retries bounded when detection/OCR is temporarily unreliable.
    MAX_OCR_ATTEMPTS_PER_TRACK = 3
