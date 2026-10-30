"""Projection and DICOM safety gates.  Ambiguity is not guessed away."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ValidationError(ValueError):
    """Raised when input cannot safely enter a PA-only profile."""


SUPPORTED_MODALITIES = {"DX", "CR"}
UNSUPPORTED_PROJECTIONS = {"AP", "LATERAL", "LL", "RL", "RPO", "LPO", "RAO", "LAO", "OBLIQUE"}


@dataclass(frozen=True)
class ProjectionDecision:
    projection: str
    source: str
    verified: bool


def _canonical_projection(value: str | None) -> str | None:
    if not value:
        return None
    token = " ".join(str(value).upper().replace("-", " ").split())
    if token in {"PA", "POSTEROANTERIOR", "POSTERO ANTERIOR"}:
        return "PA"
    if token in {"AP", "ANTEROPOSTERIOR", "ANTERO POSTERIOR"}:
        return "AP"
    if token in {"LL", "RL", "LATERAL", "LEFT LATERAL", "RIGHT LATERAL"}:
        return "LATERAL"
    if any(part in token for part in ("OBLIQUE", "RPO", "LPO", "RAO", "LAO")):
        return "OBLIQUE"
    return None


def resolve_projection(metadata: dict[str, Any], asserted_projection: str | None) -> ProjectionDecision:
    """Resolve only unambiguous metadata; assertions fill absence, never conflict."""
    candidates = {
        candidate
        for candidate in (
            _canonical_projection(metadata.get("ViewPosition")),
            _canonical_projection(metadata.get("ViewCodeMeaning")),
        )
        if candidate
    }
    if len(candidates) > 1:
        raise ValidationError("Conflicting authoritative projection metadata; refusing to guess.")
    metadata_projection = next(iter(candidates), None)
    assertion = _canonical_projection(asserted_projection)

    if metadata_projection:
        if assertion and assertion != metadata_projection:
            raise ValidationError(
                f"Explicit projection assertion '{assertion}' conflicts with metadata '{metadata_projection}'."
            )
        if metadata_projection != "PA":
            raise ValidationError(f"Unsupported projection '{metadata_projection}' for PA-only SV-DRR profile.")
        return ProjectionDecision("PA", "dicom_metadata", True)

    if assertion != "PA":
        raise ValidationError(
            "Projection metadata is missing. Supply an explicit '--projection pa' assertion to continue."
        )
    return ProjectionDecision("PA", "user_assertion", False)


def validate_dicom_metadata(metadata: dict[str, Any]) -> list[str]:
    modality = str(metadata.get("Modality", "")).upper()
    if modality not in SUPPORTED_MODALITIES:
        raise ValidationError(f"Unsupported DICOM modality '{modality or 'missing'}'; only DX and CR are supported.")
    if int(metadata.get("SamplesPerPixel", 1) or 1) != 1:
        raise ValidationError("Only single-sample monochrome DX/CR images are supported.")
    frames = metadata.get("NumberOfFrames")
    if frames is not None and int(frames) != 1:
        raise ValidationError("Multi-frame DICOM images are unsupported by this MVP.")
    photometric = str(metadata.get("PhotometricInterpretation", "")).upper()
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValidationError(f"Unsupported photometric interpretation '{photometric or 'missing'}'.")

    rotation = metadata.get("FieldOfViewRotation")
    if rotation is not None and int(rotation) not in {0, 90, 180, 270}:
        raise ValidationError("Unsupported Field of View Rotation; only 0/90/180/270 are safe.")
    flip = metadata.get("FieldOfViewHorizontalFlip")
    if flip is not None and str(flip).upper() not in {"YES", "NO"}:
        raise ValidationError("Unsupported Field of View Horizontal Flip value.")

    warnings: list[str] = []
    if str(metadata.get("BurnedInAnnotation", "")).upper() == "YES":
        warnings.append("burned_in_annotation_possible_pixel_privacy_risk")
    if str(metadata.get("LossyImageCompression", "")) == "01":
        warnings.append("lossy_compression_reported")
    return warnings
