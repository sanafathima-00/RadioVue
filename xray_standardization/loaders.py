"""Input loaders.  Source files are read only and never modified."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .metadata import extract_safe_metadata
from .validation import ValidationError


@dataclass
class LoadedInput:
    kind: str
    image: Image.Image | None
    dataset: Any | None
    metadata: dict[str, Any]


def load_input(path: str | Path) -> LoadedInput:
    source = Path(path)
    if not source.is_file():
        raise ValidationError("Input file does not exist or is not a regular file.")
    if source.suffix.lower() in {".dcm", ".dicom"}:
        try:
            import pydicom
            dataset = pydicom.dcmread(source)
        except Exception as error:
            raise ValidationError("Unable to read DICOM input.") from error
        if "PixelData" not in dataset:
            raise ValidationError("DICOM input has no Pixel Data.")
        return LoadedInput("dicom", None, dataset, extract_safe_metadata(dataset))
    try:
        with Image.open(source) as opened:
            image = opened.copy()
    except (UnidentifiedImageError, OSError) as error:
        raise ValidationError("Unsupported or unreadable raster input; expected PNG or JPEG.") from error
    if source.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValidationError("Raster input must be PNG or JPEG.")
    return LoadedInput("raster", image, None, {"format": source.suffix.lower().lstrip(".")})
