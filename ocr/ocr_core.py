from __future__ import annotations

from logger import inference_logger as logger

try:
    import easyocr
except Exception:
    easyocr = None

from typing import Any, Tuple


class OCREngine:
    """
    EasyOCR-backed OCR engine.
    """

    def __init__(
        self,
        model_path: str | None = None,
        use_gpu: bool = False,
    ) -> None:

        self.model_path = model_path
        self.use_gpu = use_gpu
        self.enabled = False
        self.reader: Any | None = None

        if easyocr is None:
            logger.warning(
                "EasyOCR is not installed; OCR disabled."
            )
            return

        try:
            try:
                import torch

                has_cuda = torch.cuda.is_available()

            except Exception:
                has_cuda = False

            use_gpu_effective = (
                self.use_gpu and has_cuda
            )

            if self.use_gpu and not has_cuda:
                logger.warning(
                    "CUDA not available. "
                    "Using CPU."
                )

            self.reader = easyocr.Reader(
                ["en"],
                gpu=use_gpu_effective,
            )

            self.enabled = True

            logger.info(
                f"EasyOCR initialized "
                f"(gpu={use_gpu_effective})"
            )

        except Exception:
            logger.exception(
                "Failed to initialize EasyOCR."
            )

            self.reader = None
            self.enabled = False

    def recognize(
        self,
        plate_crop,
    ) -> Tuple[str | None, float | None]:

        if not self.enabled:
            return None, None

        try:

            results = self.reader.readtext(
                plate_crop,
                detail=1,
                paragraph=False,
            )

            if not results:
                return None, None

            texts = []
            confs = []

            for item in results:

                if len(item) < 3:
                    continue

                text = str(item[1]).strip()

                try:
                    conf = float(item[2])
                except Exception:
                    conf = 0.0

                if text:
                    texts.append(text)
                    confs.append(conf)

            if not texts:
                return None, None

            text = " ".join(texts)

            confidence = (
                sum(confs) / len(confs)
                if confs
                else None
            )

            return text, confidence

        except Exception:
            logger.exception(
                "EasyOCR recognition failed."
            )

            return None, None