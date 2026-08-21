#!/usr/bin/env python3
"""Create a non-destructive, anonymized JPEG working copy of chest X-ray DICOM files."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from PIL import Image, ImageDraw
from pydicom.dataset import Dataset
from pydicom.pixels import apply_modality_lut, apply_voi_lut


@dataclass
class SourceImage:
    """A valid source image whose patient identifier is kept only in memory."""

    path: Path
    dataset: Dataset
    patient_key: str


@dataclass
class RunSummary:
    input_path: Path
    output_path: Path
    report_path: Path
    candidates_found: int = 0
    candidates_selected: int = 0
    dicom_sources: int = 0
    converted_sources: int = 0
    output_images: int = 0
    empty_directories: int = 0
    skipped_directories: int = 0
    patients_detected: int = 0
    failures: list[str] = field(default_factory=list)
    grouping_ambiguities: list[str] = field(default_factory=list)
    dimensions: Counter[str] = field(default_factory=Counter)
    pixel_representations: Counter[str] = field(default_factory=Counter)
    image_formats: Counter[str] = field(default_factory=Counter)
    normalization_methods: Counter[str] = field(default_factory=Counter)
    burned_in_annotation: Counter[str] = field(default_factory=Counter)
    view_metadata_present: int = 0
    source_tree_unchanged: bool | None = None
    organization_policy: str = "DICOM PatientID"
    image_base_name: str = "Image"


def is_im0_candidate(path: Path) -> bool:
    """Accept documented .IM0 files and the extensionless IM0 convention found in this dataset."""
    return path.suffix.casefold() == ".im0" or path.name.casefold() == "im0"


def safe_text(value: Any) -> str:
    """Convert a DICOM value without retaining it beyond the current process."""
    return str(value).strip() if value is not None else ""


def first_number(value: Any) -> float | None:
    """Return the first finite numeric value from a DICOM scalar or multi-value field."""
    values = value if isinstance(value, (list, tuple)) else [value]
    for item in values:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return None


def scan_tree(input_path: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """Return candidate files, all directories, and empty directories without changing the source."""
    candidates: list[Path] = []
    directories: list[Path] = []
    empty_directories: list[Path] = []
    for current, dirnames, filenames in __import__("os").walk(input_path):
        directory = Path(current)
        if directory != input_path:
            directories.append(directory)
            if not dirnames and not filenames:
                empty_directories.append(directory)
        candidates.extend(directory / filename for filename in filenames if is_im0_candidate(directory / filename))
    return sorted(candidates), directories, empty_directories


def tree_fingerprint(input_path: Path) -> str:
    """Fingerprint raw paths, sizes, and modification times without exposing them in reports."""
    digest = hashlib.sha256()
    for path in sorted((item for item in input_path.rglob("*") if item.is_file()), key=lambda item: str(item).casefold()):
        stat = path.stat()
        digest.update(str(path.relative_to(input_path)).encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_metadata(path: Path, summary: RunSummary, require_patient_id: bool) -> SourceImage | None:
    """Read DICOM metadata strictly and reject unusable or ambiguously grouped sources."""
    try:
        dataset = pydicom.dcmread(path, stop_before_pixels=False, force=False)
    except Exception as error:  # pydicom exposes several decoder-specific exception types.
        summary.failures.append(f"Unreadable DICOM source ({type(error).__name__}).")
        return None

    if "PixelData" not in dataset:
        summary.failures.append("DICOM source has no pixel data.")
        return None

    burned_in_annotation = safe_text(dataset.get("BurnedInAnnotation")).upper() or "not declared"
    summary.burned_in_annotation[burned_in_annotation] += 1
    if burned_in_annotation == "YES":
        summary.failures.append("DICOM source declares burned-in annotation and was not converted.")
        return None

    patient_id = safe_text(dataset.get("PatientID"))
    if require_patient_id and not patient_id:
        summary.grouping_ambiguities.append("At least one decodable DICOM source has no PatientID.")
        return None

    rows = dataset.get("Rows")
    columns = dataset.get("Columns")
    if rows is None or columns is None:
        summary.failures.append("DICOM source has no usable image dimensions.")
        return None

    summary.dicom_sources += 1
    summary.dimensions[f"{rows}x{columns}"] += 1
    summary.image_formats[safe_text(dataset.get("PhotometricInterpretation")) or "unspecified"] += 1
    representation = (
        f"bits_allocated={dataset.get('BitsAllocated', 'unknown')}; "
        f"bits_stored={dataset.get('BitsStored', 'unknown')}; "
        f"pixel_representation={dataset.get('PixelRepresentation', 'unknown')}; "
        f"samples_per_pixel={dataset.get('SamplesPerPixel', 'unknown')}"
    )
    summary.pixel_representations[representation] += 1
    if safe_text(dataset.get("ViewPosition")) or safe_text(dataset.get("PatientPosition")):
        summary.view_metadata_present += 1
    group_key = patient_id if require_patient_id else str(path.parent)
    return SourceImage(path=path, dataset=dataset, patient_key=group_key)


def normalize_grayscale(dataset: Dataset, pixels: np.ndarray) -> tuple[np.ndarray, str]:
    """Apply DICOM modality/window information before a documented robust fallback."""
    transformed = np.asarray(apply_modality_lut(pixels, dataset), dtype=np.float64)
    finite = np.isfinite(transformed)
    if not finite.any():
        raise ValueError("Pixel data contains no finite values after modality transformation.")

    center = first_number(dataset.get("WindowCenter"))
    width = first_number(dataset.get("WindowWidth"))
    if center is not None and width is not None and width > 1:
        lower = center - 0.5 - (width - 1) / 2
        upper = center - 0.5 + (width - 1) / 2
        scaled = np.clip((transformed - lower) / (upper - lower), 0.0, 1.0)
        method = "DICOM window center/width"
    elif "VOILUTSequence" in dataset:
        voi = np.asarray(apply_voi_lut(transformed, dataset), dtype=np.float64)
        finite_voi = np.isfinite(voi)
        if not finite_voi.any() or np.ptp(voi[finite_voi]) == 0:
            raise ValueError("VOI LUT produced no usable display range.")
        lower, upper = np.percentile(voi[finite_voi], [0.5, 99.5])
        scaled = np.clip((voi - lower) / (upper - lower), 0.0, 1.0)
        method = "DICOM VOI LUT with robust display bounds"
    else:
        lower, upper = np.percentile(transformed[finite], [0.5, 99.5])
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            raise ValueError("No non-constant display range is available without DICOM window metadata.")
        scaled = np.clip((transformed - lower) / (upper - lower), 0.0, 1.0)
        method = "robust 0.5/99.5 percentile fallback (no DICOM window)"

    monochrome_one = safe_text(dataset.get("PhotometricInterpretation")).upper() == "MONOCHROME1"
    inverse_presentation = safe_text(dataset.get("PresentationLUTShape")).upper() == "INVERSE"
    if monochrome_one ^ inverse_presentation:
        scaled = 1.0 - scaled
    return np.rint(scaled * 255).astype(np.uint8), method


def frames_from_dataset(dataset: Dataset) -> list[np.ndarray]:
    """Decode only supported grayscale or RGB frame layouts without guessing dimensions."""
    pixels = np.asarray(dataset.pixel_array)
    samples = int(dataset.get("SamplesPerPixel", 1))
    if samples == 1 and pixels.ndim == 2:
        return [pixels]
    if samples == 1 and pixels.ndim == 3:
        return [pixels[index] for index in range(pixels.shape[0])]
    if samples == 3 and pixels.ndim == 3 and pixels.shape[-1] == 3:
        return [pixels]
    if samples == 3 and pixels.ndim == 4 and pixels.shape[-1] == 3:
        return [pixels[index] for index in range(pixels.shape[0])]
    raise ValueError(f"Unsupported pixel array shape {pixels.shape} for SamplesPerPixel={samples}.")


def save_source_images(
    source: SourceImage,
    destination: Path,
    image_start: int,
    image_base_name: str,
    summary: RunSummary,
    quality: int,
) -> int:
    """Decode one source and save its frames with generic image names only."""
    frames = frames_from_dataset(source.dataset)
    saved = 0
    for offset, frame in enumerate(frames):
        image_number = image_start + offset
        filename = f"{image_base_name}.jpg" if image_number == 0 else f"{image_base_name}_{image_number:03d}.jpg"
        output_path = destination / filename
        if int(source.dataset.get("SamplesPerPixel", 1)) == 1:
            rendered, method = normalize_grayscale(source.dataset, frame)
            image = Image.fromarray(rendered, mode="L")
        else:
            image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
            method = "RGB pixel data preserved"
        image.save(output_path, format="JPEG", quality=quality, optimize=True)
        with Image.open(output_path) as verified:
            verified.load()
        summary.normalization_methods[method] += 1
        saved += 1
    return saved


def create_contact_sheet(images: list[Path], output_path: Path, maximum: int) -> None:
    """Create a generic preview sheet from converted output images."""
    selected = images[:maximum]
    if not selected:
        return
    thumbnail_size = (240, 240)
    label_height = 24
    columns = min(4, len(selected))
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("L", (columns * thumbnail_size[0], rows * (thumbnail_size[1] + label_height)), color=0)
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(selected):
        with Image.open(path) as image:
            preview = image.convert("L")
            preview.thumbnail(thumbnail_size)
            x = (index % columns) * thumbnail_size[0] + (thumbnail_size[0] - preview.width) // 2
            y = (index // columns) * (thumbnail_size[1] + label_height) + (thumbnail_size[1] - preview.height) // 2
            sheet.paste(preview, (x, y))
        draw.text((index % columns * thumbnail_size[0] + 4, (index // columns) * (thumbnail_size[1] + label_height) + thumbnail_size[1] + 4), f"Sample {index:02d}", fill=255)
    sheet.save(output_path, format="JPEG", quality=90, optimize=True)


def markdown_counter(counter: Counter[str]) -> str:
    if not counter:
        return "- None recorded."
    return "\n".join(f"- `{key}`: {value}" for key, value in sorted(counter.items()))


def write_report(summary: RunSummary, processing_limit: int | None, stopped_for_grouping: bool) -> None:
    """Write an anonymized report with no source paths or DICOM identifier values."""
    summary.report_path.parent.mkdir(parents=True, exist_ok=True)
    failure_lines = "\n".join(f"- {message}" for message in summary.failures) or "- None."
    ambiguity_lines = "\n".join(f"- {message}" for message in summary.grouping_ambiguities) or "- None."
    status = "STOPPED: patient grouping could not be established safely." if stopped_for_grouping else "Completed without modifying the raw dataset."
    limit_text = str(processing_limit) if processing_limit is not None else "all detected candidates"
    content = f"""# RadioVue X-ray Dataset Cleanup Report

Generated: {datetime.now(timezone.utc).isoformat()}  
Status: {status}  
Input was read only; patient identifiers and source paths are intentionally omitted.

## Counts

- Candidate IM0-style source files detected: {summary.candidates_found}
- Candidate source files selected for this run: {summary.candidates_selected} (limit: {limit_text})
- Decodable DICOM source files: {summary.dicom_sources}
- Anonymized output units detected: {summary.patients_detected}
- Organization policy: {summary.organization_policy}
- Output image base name: `{summary.image_base_name}`
- Source files successfully converted: {summary.converted_sources}
- Output JPEG images: {summary.output_images}
- Source files failed: {len(summary.failures)}
- Empty source directories: {summary.empty_directories}
- Source directories without a usable image among this run's selected sources: {summary.skipped_directories}
- Sources with directly present view metadata: {summary.view_metadata_present} (reported only; the output image name is caller-supplied)
- Raw source tree unchanged during this run: {"yes" if summary.source_tree_unchanged else "no"}

## Image Dimensions

{markdown_counter(summary.dimensions)}

## Image Format and Pixel Representation

### Photometric interpretation

{markdown_counter(summary.image_formats)}

### Pixel representation

{markdown_counter(summary.pixel_representations)}

### JPEG rendering methods

{markdown_counter(summary.normalization_methods)}

### Burned-in annotation declaration

{markdown_counter(summary.burned_in_annotation)}

## Conversion Failures or Potential Corruption

{failure_lines}

## Grouping Ambiguities

{ambiguity_lines}

## Source Directory Notes

Empty source directory names are omitted to avoid exposing potentially identifying path components. No empty output directories are created.

## Reproducibility

- Decoder: pydicom {pydicom.__version__}
- Array processing: numpy {np.__version__}
- JPEG writer: Pillow {Image.__version__}
- Conversion policy: DICOM modality transform and window information are used when available; only images without usable window information use the documented robust 0.5/99.5 percentile fallback.
- View policy: the default `Image` name does not infer a view; a caller-supplied name such as `PA` is an explicit workspace label, not a DICOM-derived classification.
- Privacy policy: sources declaring `BurnedInAnnotation=YES` are reported and not converted; this is metadata-based and is not an OCR inspection of image pixels.
"""
    summary.report_path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Read-only raw dataset directory.")
    parser.add_argument("--output", required=True, type=Path, help="New or empty output directory for JPEG files.")
    parser.add_argument("--report", type=Path, default=Path("reports/dataset_cleanup_report.md"), help="Anonymized Markdown report path.")
    parser.add_argument("--limit", type=int, help="Maximum candidate source files to process; use for sample runs.")
    parser.add_argument("--preview-count", type=int, default=12, help="Maximum images in the contact sheet.")
    parser.add_argument("--jpeg-quality", type=int, default=95, choices=range(1, 101), metavar="1..100")
    parser.add_argument(
        "--group-by-source-folder",
        action="store_true",
        help="Treat each folder containing an IM0 source as one anonymized output unit instead of grouping by DICOM PatientID.",
    )
    parser.add_argument(
        "--image-name",
        default="Image",
        help="Explicit output image base name, such as PA; this does not infer a view from metadata.",
    )
    arguments = parser.parse_args()

    if not arguments.input.is_dir():
        parser.error("--input must be an existing directory.")
    if arguments.limit is not None and arguments.limit < 1:
        parser.error("--limit must be at least 1.")
    if not arguments.image_name or any(character in arguments.image_name for character in '\\/<>:"|?*'):
        parser.error("--image-name must be a non-empty filename base without path characters.")
    if arguments.output.exists() and any(arguments.output.iterdir()):
        parser.error("--output already exists and is not empty; use a new empty output directory.")

    summary = RunSummary(input_path=arguments.input, output_path=arguments.output, report_path=arguments.report)
    if arguments.group_by_source_folder:
        summary.organization_policy = "each source folder containing an IM0 file becomes one workspace output unit"
    summary.image_base_name = arguments.image_name
    source_before = tree_fingerprint(arguments.input)

    def finish_report(stopped_for_grouping: bool) -> bool:
        summary.source_tree_unchanged = source_before == tree_fingerprint(arguments.input)
        write_report(summary, arguments.limit, stopped_for_grouping=stopped_for_grouping)
        return bool(summary.source_tree_unchanged)

    candidates, directories, empty_directories = scan_tree(arguments.input)
    summary.candidates_found = len(candidates)
    summary.empty_directories = len(empty_directories)
    selected = candidates[: arguments.limit] if arguments.limit is not None else candidates
    summary.candidates_selected = len(selected)
    if not selected:
        summary.failures.append("No .IM0 or extensionless IM0 source files were found.")
        summary.skipped_directories = len(directories)
        unchanged = finish_report(stopped_for_grouping=False)
        print("No IM0-style source files found; report written and no output created.")
        return 2 if unchanged else 3

    sources = [
        loaded
        for path in selected
        if (loaded := load_metadata(path, summary, require_patient_id=not arguments.group_by_source_folder)) is not None
    ]
    grouped: dict[str, list[SourceImage]] = defaultdict(list)
    for source in sources:
        grouped[source.patient_key].append(source)
    summary.patients_detected = len(grouped)

    usable_ancestors: set[Path] = set()
    for source in sources:
        parent = source.path.parent
        while parent != arguments.input and arguments.input in parent.parents:
            usable_ancestors.add(parent)
            parent = parent.parent
    summary.skipped_directories = sum(directory not in usable_ancestors for directory in directories)

    if summary.grouping_ambiguities:
        unchanged = finish_report(stopped_for_grouping=True)
        print("Patient grouping ambiguity detected; report written and no output created.")
        return 2 if unchanged else 3
    if not grouped:
        summary.failures.append("No safely groupable DICOM images were available for conversion.")
        unchanged = finish_report(stopped_for_grouping=False)
        print("No safely groupable DICOM images found; report written and no output created.")
        return 2 if unchanged else 3

    arguments.output.mkdir(parents=True, exist_ok=True)
    converted_paths: list[Path] = []
    for patient_index, patient_key in enumerate(sorted(grouped, key=lambda key: hashlib.sha256(key.encode("utf-8")).hexdigest())):
        patient_directory = arguments.output / f"Patient_{patient_index:03d}"
        patient_directory.mkdir()
        image_index = 0
        for source in sorted(grouped[patient_key], key=lambda item: str(item.path).casefold()):
            try:
                saved = save_source_images(
                    source,
                    patient_directory,
                    image_index,
                    arguments.image_name,
                    summary,
                    arguments.jpeg_quality,
                )
            except Exception as error:
                summary.failures.append(f"DICOM pixel conversion failed ({type(error).__name__}).")
                continue
            image_index += saved
            summary.converted_sources += 1
            summary.output_images += saved
        patient_images = sorted(patient_directory.glob("*.jpg"))
        if not patient_images:
            patient_directory.rmdir()
        else:
            converted_paths.extend(patient_images)

    create_contact_sheet(converted_paths, arguments.output / "preview_contact_sheet.jpg", arguments.preview_count)
    unchanged = finish_report(stopped_for_grouping=False)
    print(f"Converted {summary.converted_sources} source files into {summary.output_images} JPEG images for {summary.patients_detected} anonymized output folders.")
    print(f"Report: {summary.report_path}")
    if not unchanged:
        print("Raw source tree changed during the run; inspect the report before continuing.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
