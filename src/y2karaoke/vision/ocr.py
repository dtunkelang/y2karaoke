"""OCR Engine wrapper for Vision (macOS) and PaddleOCR."""

from __future__ import annotations

import logging
import platform
from typing import Any, List, Dict

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore

from ..exceptions import OCRError

logger = logging.getLogger(__name__)

_OCR_ENGINE = None


class VisionOCR:
    """Wrapper for Apple's Vision framework OCR."""

    def __init__(self) -> None:
        try:
            import Vision
            from Quartz import CIImage, kCIFormatRGBA8
            import objc
        except ImportError as e:
            raise OCRError(
                "Apple Vision dependencies (pyobjc-framework-Vision, etc.) not found."
            ) from e

        self.Vision = Vision
        self.CIImage = CIImage
        self.kCIFormatRGBA8 = kCIFormatRGBA8
        self.objc = objc

    def predict(self, frame_nd: np.ndarray) -> List[Dict[str, Any]]:
        """Run OCR on a numpy image array (BGR)."""
        if cv2 is None or np is None:
            raise OCRError("OpenCV and Numpy are required for OCR.")

        frame_rgba = cv2.cvtColor(frame_nd, cv2.COLOR_BGR2RGBA)
        h, w = frame_rgba.shape[:2]
        bytes_data = frame_rgba.tobytes()
        ci_image = self.CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
            bytes_data, w * 4, (w, h), self.kCIFormatRGBA8, None
        )

        request = self.Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(self.Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)

        handler = self.Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
            ci_image, None
        )
        success, error = handler.performRequests_error_([request], None)

        if not success:
            logger.error(f"Vision OCR request failed: {error}")
            return []

        results = request.results()
        rec_texts = []
        rec_boxes = []
        rec_scores = []

        for result in results:
            candidates = result.topCandidates_(1)
            if not candidates:
                continue
            top_candidate = candidates[0]
            full_text = top_candidate.string()
            words = full_text.split()
            current_pos = 0

            for word_text in words:
                word_start_idx = full_text.find(word_text, current_pos)
                if word_start_idx == -1:
                    continue
                current_pos = word_start_idx + len(word_text)

                try:
                    range_obj = (word_start_idx, len(word_text))
                    box_result_tuple = top_candidate.boundingBoxForRange_error_(
                        range_obj, None
                    )
                    if not box_result_tuple or not box_result_tuple[0]:
                        continue

                    # Attempt to get first char box for finer granularity if needed
                    # (Currently unused but kept for parity with original logic)
                    first_char_range = (word_start_idx, 1)
                    char_box_result = top_candidate.boundingBoxForRange_error_(
                        first_char_range, None
                    )
                    char_box = None
                    if char_box_result and char_box_result[0]:
                        cobs = char_box_result[0]
                        cbbox = cobs.boundingBox()
                        char_box = [
                            cbbox.origin.x * w,
                            (1.0 - cbbox.origin.y - cbbox.size.height) * h,
                            cbbox.size.width * w,
                            cbbox.size.height * h,
                        ]

                    bbox = box_result_tuple[0].boundingBox()
                    # Convert normalized coords (bottom-left origin) to pixel coords (top-left origin)
                    px_x = bbox.origin.x * w
                    px_y = (1.0 - bbox.origin.y - bbox.size.height) * h
                    px_w = bbox.size.width * w
                    px_h = bbox.size.height * h

                    # PaddleOCR format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    box = [
                        [px_x, px_y],
                        [px_x + px_w, px_y],
                        [px_x + px_w, px_y + px_h],
                        [px_x, px_y + px_h],
                    ]

                    rec_texts.append(word_text)
                    rec_boxes.append({"word": box, "first_char": char_box})
                    rec_scores.append(top_candidate.confidence())

                except Exception as e:
                    logger.debug(f"Error parsing bbox for '{word_text}': {e}")
                    continue

        if not rec_texts:
            return []

        # Return list of dicts to match generic interface
        return [
            {"rec_texts": rec_texts, "rec_boxes": rec_boxes, "rec_scores": rec_scores}
        ]


def get_ocr_engine() -> Any:
    """Get the singleton OCR engine instance, initializing if necessary."""
    global _OCR_ENGINE
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE

    # Try Apple Vision on macOS/arm64
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            logger.info("Initializing Apple Vision OCR...")
            _OCR_ENGINE = VisionOCR()
            return _OCR_ENGINE
        except Exception as e:
            logger.warning(f"Failed to initialize Apple Vision OCR: {e}. Falling back.")

    # Fallback to PaddleOCR
    try:
        from paddleocr import PaddleOCR

        logger.info("Initializing PaddleOCR...")
        # lang='en' is standard for typical western karaoke; could be configurable
        _OCR_ENGINE = PaddleOCR(
            use_textline_orientation=True, lang="en", show_log=False
        )
    except ImportError as e:
        msg = "PaddleOCR not found. Please install via `pip install paddlepaddle paddleocr`"
        logger.error(msg)
        raise OCRError(msg) from e

    return _OCR_ENGINE
