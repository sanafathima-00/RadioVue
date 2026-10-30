"""End-to-end, non-destructive standardization orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .geometry import export_geometry
from .intensity import render_dicom, render_raster
from .loaders import load_input
from .profiles import get_profile
from .report import source_sha256, write_comparison, write_report
from .validation import ValidationError, resolve_projection, validate_dicom_metadata


class StandardizationError(ValidationError):
    """Public pipeline error."""


def _apply_authoritative_orientation(image, metadata: dict[str, Any]):
    rotation = metadata.get("FieldOfViewRotation")
    flip = metadata.get("FieldOfViewHorizontalFlip")
    operations: list[str] = []
    if rotation is not None:
        rotation = int(rotation)
        if rotation:
            # DICOM describes this as clockwise relative to detector; PIL rotate is CCW.
            image = image.rotate(-rotation, expand=True)
            operations.append(f"rotate_clockwise_{rotation}")
    if str(flip).upper() == "YES":
        from PIL import Image
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        operations.append("horizontal_flip")
    return image, operations


def standardize(
    input_path: str | Path,
    output_dir: str | Path,
    profile_name: str,
    asserted_projection: str | None = None,
) -> dict[str, Any]:
    """Create standardized PNG, JSON report, and comparison image.

    The original input is never copied, altered, or deleted.
    """
    try:
        profile = get_profile(profile_name)
        loaded = load_input(input_path)
        warnings: list[str] = []
        if loaded.kind == "dicom":
            warnings.extend(validate_dicom_metadata(loaded.metadata))
        decision = resolve_projection(loaded.metadata, asserted_projection)

        if loaded.kind == "dicom":
            before, intensity = render_dicom(loaded.dataset, loaded.metadata)
            before, orientation_operations = _apply_authoritative_orientation(before, loaded.metadata)
        else:
            before, intensity = render_raster(loaded.image)
            orientation_operations = []
            warnings.append("raster_input_has_no_authoritative_acquisition_metadata")

        after, geometry = export_geometry(before, profile.target_size, profile.resize_strategy)
    except ValidationError as error:
        raise StandardizationError(str(error)) from error

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = "standardized"
    png_path = destination / f"{stem}.png"
    report_path = destination / f"{stem}.report.json"
    comparison_path = destination / f"{stem}.comparison.png"
    after.save(png_path, format="PNG")
    write_comparison(before, after, comparison_path)

    report = {
        "schema_version": 1,
        "profile": {
            "name": profile.name,
            "target_size": profile.target_size,
            "resize_strategy": profile.resize_strategy,
            "experimental": profile.experimental,
        },
        "input": {
            "kind": loaded.kind,
            "sha256": source_sha256(input_path),
            "source_size": list(before.size),
        },
        "projection": {
            "value": decision.projection,
            "source": decision.source,
            "verified": decision.verified,
        },
        "safe_technical_metadata": loaded.metadata,
        "transformations": {
            "orientation_operations": orientation_operations,
            "intensity": intensity,
            "geometry": geometry,
        },
        "warnings": sorted(set(warnings)),
        "privacy": {
            "patient_metadata_excluded": True,
            "burned_in_annotation_is_pixel_risk": str(loaded.metadata.get("BurnedInAnnotation", "")).upper() == "YES",
        },
        "outputs": {
            "standardized_png": png_path.name,
            "comparison_png": comparison_path.name,
        },
    }
    write_report(report_path, report)
    return {"png_path": png_path, "report_path": report_path, "comparison_path": comparison_path, "report": report}
