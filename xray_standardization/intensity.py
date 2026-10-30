"""Conservative DICOM and raster intensity handling."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .validation import ValidationError


def _linear_uint8(values: np.ndarray, padding_value: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if padding_value is not None:
        finite &= values != padding_value
    if not finite.any():
        raise ValidationError("No finite non-padding pixels available for intensity export.")
    low, high = float(values[finite].min()), float(values[finite].max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.clip((values - low) * 255.0 / (high - low), 0, 255)
    return np.rint(scaled).astype(np.uint8)


def _presentation_inversion(metadata: dict[str, object]) -> bool:
    photometric = str(metadata.get("PhotometricInterpretation", "")).upper()
    shape = metadata.get("PresentationLUTShape")
    expected = "INVERSE" if photometric == "MONOCHROME1" else "IDENTITY"
    if shape is not None and str(shape).upper() != expected:
        raise ValidationError(
            f"Ambiguous DICOM polarity: {photometric} with Presentation LUT Shape '{shape}'."
        )
    return photometric == "MONOCHROME1"


def render_dicom(dataset, metadata: dict[str, object]) -> tuple[Image.Image, dict[str, object]]:
    """Apply supported DICOM display semantics and return a grayscale image."""
    try:
        from pydicom.pixels import apply_modality_lut, apply_voi_lut
    except ImportError as error:  # pragma: no cover - dependency declaration covers this
        raise RuntimeError("DICOM support requires pydicom.") from error

    pixels = dataset.pixel_array
    rendered = apply_modality_lut(pixels, dataset)
    voi_applied = False
    intent = str(metadata.get("PresentationIntentType", "")).upper()
    has_voi = hasattr(dataset, "VOILUTSequence") or hasattr(dataset, "WindowCenter")
    if intent == "FOR PRESENTATION" and has_voi:
        rendered = apply_voi_lut(rendered, dataset)
        voi_applied = True

    padding = getattr(dataset, "PixelPaddingValue", None)
    image = _linear_uint8(rendered, int(padding) if padding is not None else None)
    inverted = _presentation_inversion(metadata)
    if inverted:
        image = np.uint8(255) - image
    return Image.fromarray(image, mode="L"), {
        "modality_lut_applied": True,
        "voi_lut_applied": voi_applied,
        "presentation_inverted": inverted,
        "uint8_mapping": "linear_minmax_non_padding",
    }


def render_raster(image: Image.Image) -> tuple[Image.Image, dict[str, object]]:
    """Treat PNG/JPEG as already-rendered display pixels; no contrast transform."""
    return image.convert("L"), {
        "modality_lut_applied": False,
        "voi_lut_applied": False,
        "presentation_inverted": False,
        "uint8_mapping": "existing_display_pixels",
    }
