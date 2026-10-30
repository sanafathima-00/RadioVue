"""Safe DICOM metadata extraction.

Only technical, non-identifying attributes are returned.  Never add free-text
fields, UIDs, dates, accession numbers, institution names, or patient fields.
"""

from __future__ import annotations

from typing import Any


SAFE_FIELDS = (
    "Modality",
    "SOPClassUID",
    "ViewPosition",
    "PatientPosition",
    "PhotometricInterpretation",
    "PresentationLUTShape",
    "PresentationIntentType",
    "PixelIntensityRelationship",
    "PixelIntensityRelationshipSign",
    "Rows",
    "Columns",
    "BitsAllocated",
    "BitsStored",
    "PixelRepresentation",
    "SamplesPerPixel",
    "NumberOfFrames",
    "BurnedInAnnotation",
    "LossyImageCompression",
    "KVP",
    "ExposureTime",
    "XRayTubeCurrent",
    "Exposure",
    "DistanceSourceToDetector",
    "DistanceSourceToPatient",
    "ImagerPixelSpacing",
    "PixelSpacing",
    "FieldOfViewShape",
    "FieldOfViewDimensions",
    "FieldOfViewRotation",
    "FieldOfViewHorizontalFlip",
)


def _safe_value(value: Any) -> Any:
    """Convert DICOM scalar/list values to JSON-safe primitives."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "real") and not isinstance(value, (str, bytes)):
        return float(value)
    if isinstance(value, (list, tuple)) or value.__class__.__name__ == "MultiValue":
        return [_safe_value(item) for item in value]
    return str(value)


def _view_code_meaning(dataset: Any) -> str | None:
    sequence = getattr(dataset, "ViewCodeSequence", None)
    if not sequence or len(sequence) != 1:
        return None
    meaning = getattr(sequence[0], "CodeMeaning", None)
    return str(meaning) if meaning else None


def extract_safe_metadata(dataset: Any) -> dict[str, Any]:
    """Extract a strict allow-list of technical metadata only."""
    result: dict[str, Any] = {}
    for name in SAFE_FIELDS:
        if hasattr(dataset, name):
            result[name] = _safe_value(getattr(dataset, name))
    view_meaning = _view_code_meaning(dataset)
    if view_meaning:
        result["ViewCodeMeaning"] = view_meaning
    return result


def metadata_has_identifier_keys(payload: dict[str, Any]) -> bool:
    """Defensive test helper for report privacy assertions."""
    forbidden = {
        "patientname", "patientid", "birthdate", "accessionnumber", "studydate",
        "seriesdate", "institutionname", "referringphysicianname", "sopinstanceuid",
    }
    return any(key.lower() in forbidden for key in payload)
