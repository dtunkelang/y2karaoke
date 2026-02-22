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


def _box_to_quad(box: Any) -> list[list[float]] | None:
    """Normalize OCR box output to a 4-point polygon."""
    if np is None:
        return None

    try:
        arr = np.array(box, dtype=float)
    except Exception:
        return None

    if arr.ndim == 2 and arr.shape == (4, 2):
        return arr.tolist()

    if arr.ndim == 1 and arr.shape[0] == 4:
        x1, y1, x2, y2 = arr.tolist()
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        return [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ]

    return None


def normalize_ocr_items(items: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize OCR engine output to word-level `rec_texts`/`rec_boxes`/`rec_scores`.

    PaddleOCR v3 returns line-level `rec_*` plus optional token-level
    `text_word`/`text_word_boxes` when `return_word_box=True`. This helper converts
    token-level output into the canonical shape consumed by the visual pipeline.
    """
    token_lines = items.get("text_word")
    token_boxes_lines = items.get("text_word_boxes")
    line_scores = items.get("rec_scores", [])

    if isinstance(token_lines, list) and isinstance(token_boxes_lines, list):
        rec_texts: list[str] = []
        rec_boxes: list[dict[str, Any]] = []
        rec_scores: list[float] = []

        for line_idx, (tokens, token_boxes) in enumerate(
            zip(token_lines, token_boxes_lines)
        ):
            if not isinstance(tokens, list):
                continue
            score = float(line_scores[line_idx]) if line_idx < len(line_scores) else 1.0
            for token, token_box in zip(tokens, token_boxes):
                token_text = str(token).strip()
                if not token_text:
                    continue
                quad = _box_to_quad(token_box)
                if quad is None:
                    continue
                rec_texts.append(token_text)
                rec_boxes.append({"word": quad, "first_char": None})
                rec_scores.append(score)

        if rec_texts:
            return {
                "rec_texts": rec_texts,
                "rec_boxes": rec_boxes,
                "rec_scores": rec_scores,
            }

    rec_texts = items.get("rec_texts", [])
    rec_boxes = items.get("rec_boxes", [])
    rec_scores = items.get("rec_scores", [1.0] * len(rec_texts))
    return {
        "rec_texts": rec_texts,
        "rec_boxes": rec_boxes,
        "rec_scores": rec_scores,
    }


class VisionOCR:
    """Wrapper for Apple's Vision framework OCR."""

    def __init__(self) -> None:
        try:
            import Vision
            from Quartz import CIImage, kCIFormatRGBA8
            import objc
        except ImportError as e:
            raise OCRError(
                "Apple Vision dependencies not found. "
                "Install with: pip install -e '.[vision_macos]'"
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
            # Request up to 5 candidates to resolve visual ambiguities
            candidates = result.topCandidates_(5)
            if not candidates:
                continue

            top_candidate = self._select_best_ocr_candidate(candidates)
            full_text = top_candidate.string()
            words = full_text.split()
            current_pos = 0

            for word_text in words:
                word_start_idx = full_text.find(word_text, current_pos)
                if word_start_idx == -1:
                    continue
                current_pos = word_start_idx + len(word_text)

                try:
                    word_data = self._parse_vision_word_data(
                        top_candidate, word_text, word_start_idx, w, h
                    )
                    if word_data:
                        rec_texts.append(word_text)
                        rec_boxes.append(
                            {"word": word_data["box"], "first_char": word_data["char"]}
                        )
                        rec_scores.append(top_candidate.confidence())
                except Exception as e:
                    logger.debug(f"Error parsing word '{word_text}': {e}")
                    continue

        if not rec_texts:
            return []

        # Return list of dicts to match generic interface
        return [
            {"rec_texts": rec_texts, "rec_boxes": rec_boxes, "rec_scores": rec_scores}
        ]

    def _select_best_ocr_candidate(self, candidates: List[Any]) -> Any:
        """Heuristic: prefer candidates that resolve common visual confusions."""
        chosen = candidates[0]
        top_conf = candidates[0].confidence()
        confusions = [
            ("rn", "m"),
            ("in", "m"),
            ("ni", "m"),
            ("ri", "m"),
            ("li", "h"),
            ("vv", "w"),
            ("cl", "d"),
        ]

        for i in range(1, len(candidates)):
            cand = candidates[i]
            # Only consider if confidence is very close (within 10%)
            if cand.confidence() < top_conf * 0.9:
                break

            cand_str = cand.string().lower()
            top_str = chosen.string().lower()

            changed = False
            for confusion, target in confusions:
                if confusion in top_str and target in cand_str:
                    # Candidate resolved a common confusion
                    chosen = cand
                    changed = True
                    break
            if changed:
                break
        return chosen

    def _parse_vision_word_data(
        self, candidate: Any, text: str, start_idx: int, w: int, h: int
    ) -> Dict[str, Any] | None:
        """Extract bounding box and character data for a specific word range."""
        range_obj = (start_idx, len(text))
        box_result_tuple = candidate.boundingBoxForRange_error_(range_obj, None)
        if not box_result_tuple or not box_result_tuple[0]:
            return None

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

        # First character box for alignment precision
        char_box = None
        char_box_res = candidate.boundingBoxForRange_error_((start_idx, 1), None)
        if char_box_res and char_box_res[0]:
            cobs = char_box_res[0]
            cbbox = cobs.boundingBox()
            char_box = [
                cbbox.origin.x * w,
                (1.0 - cbbox.origin.y - cbbox.size.height) * h,
                cbbox.size.width * w,
                cbbox.size.height * h,
            ]

        return {"box": box, "char": char_box}


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
        # lang='en' is standard for typical western karaoke; could be configurable.
        # Newer PaddleOCR releases may reject `show_log`, so retry without it.
        try:
            _OCR_ENGINE = PaddleOCR(
                use_textline_orientation=True,
                lang="en",
                return_word_box=True,
                show_log=False,
            )
        except Exception as e:
            if "show_log" in str(e):
                logger.warning(
                    "PaddleOCR rejected show_log argument; retrying with compatible kwargs"
                )
                _OCR_ENGINE = PaddleOCR(
                    use_textline_orientation=True, lang="en", return_word_box=True
                )
            elif "return_word_box" in str(e):
                logger.warning(
                    "PaddleOCR rejected return_word_box argument; retrying without it"
                )
                _OCR_ENGINE = PaddleOCR(
                    use_textline_orientation=True, lang="en", show_log=False
                )
            else:
                raise
    except ImportError as e:
        msg = "PaddleOCR not found. Please install via `pip install paddlepaddle paddleocr`"
        logger.error(msg)
        raise OCRError(msg) from e

    return _OCR_ENGINE


def get_ocr_cache_fingerprint() -> str:
    """Return a stable cache fingerprint for the preferred OCR backend."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "vision_macos_arm64_v2"
    return "paddleocr"
