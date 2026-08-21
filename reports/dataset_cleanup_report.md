# RadioVue X-ray Dataset Cleanup Report

Generated: 2026-08-20T12:35:06.627825+00:00  
Status: Completed without modifying the raw dataset.  
Input was read only; patient identifiers and source paths are intentionally omitted.

## Counts

- Candidate IM0-style source files detected: 273
- Candidate source files selected for this run: 273 (limit: all detected candidates)
- Decodable DICOM source files: 273
- Anonymized output units detected: 273
- Organization policy: each source folder containing an IM0 file becomes one workspace output unit
- Output image base name: `PA`
- Source files successfully converted: 273
- Output JPEG images: 273
- Source files failed: 0
- Empty source directories: 0
- Source directories without a usable image among this run's selected sources: 0
- Sources with directly present view metadata: 0 (reported only; the output image name is caller-supplied)
- Raw source tree unchanged during this run: yes

## Image Dimensions

- `1670x2010`: 2
- `1760x2140`: 235
- `2010x1670`: 7
- `2140x1760`: 29

## Image Format and Pixel Representation

### Photometric interpretation

- `MONOCHROME1`: 273

### Pixel representation

- `bits_allocated=16; bits_stored=10; pixel_representation=0; samples_per_pixel=1`: 273

### JPEG rendering methods

- `DICOM window center/width`: 273

### Burned-in annotation declaration

- `not declared`: 273

## Conversion Failures or Potential Corruption

- None.

## Grouping Ambiguities

- None.

## Source Directory Notes

Empty source directory names are omitted to avoid exposing potentially identifying path components. No empty output directories are created.

## Reproducibility

- Decoder: pydicom 3.0.2
- Array processing: numpy 2.2.6
- JPEG writer: Pillow 12.0.0
- Conversion policy: DICOM modality transform and window information are used when available; only images without usable window information use the documented robust 0.5/99.5 percentile fallback.
- View policy: the default `Image` name does not infer a view; a caller-supplied name such as `PA` is an explicit workspace label, not a DICOM-derived classification.
- Privacy policy: sources declaring `BurnedInAnnotation=YES` are reported and not converted; this is metadata-based and is not an OCR inspection of image pixels.
