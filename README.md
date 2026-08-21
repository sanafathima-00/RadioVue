# RadioVue

RadioVue is an experimental medical-imaging research project for complementary chest X-ray view synthesis, multi-view X-ray-to-CT reconstruction, and CT-to-X-ray consistency evaluation.

## Repository Layout

- `Parallax/` contains the complementary X-ray view-synthesis research repository.
- `Volumea/` is reserved for multi-view X-ray-to-CT reconstruction work.
- `Recon/` is reserved for CT-to-X-ray consistency evaluation work.

The root `.gitignore` excludes all other working files by default, including datasets, generated images, reports, and agent-local configuration, while retaining these three module directories and the root documentation files.

## Safe X-ray Working-Copy Preparation

`scripts/convert_im0_to_jpeg.py` creates a separate, anonymized JPEG working copy from DICOM files named `.IM0` (any case) or the extensionless `IM0` convention found in the current raw dataset. It never writes to, renames, moves, or deletes raw files.

Install the required libraries in the active Python environment:

```powershell
python -m pip install pydicom numpy Pillow
```

Run a small, separate sample first:

```powershell
python scripts/convert_im0_to_jpeg.py `
  --input "C:\Users\Admin\Downloads\Xrays\XRAY Data" `
  --output "RadioVue_XRays_sample" `
  --limit 5
```

After verifying the output images, contact sheet, patient grouping, and report, use a **new empty output directory** for a full conversion:

```powershell
python scripts/convert_im0_to_jpeg.py `
  --input "C:\Users\Admin\Downloads\Xrays\XRAY Data" `
  --output "RadioVue_XRays"
```

### Flat workspace layout by source folder

When each IM0-containing folder is intentionally a separate workspace slot for one input view and its future generated companion view, use the source-folder mode. This is an explicit organizational choice, not patient grouping from DICOM metadata, and `PA` is an explicit default label rather than an inferred classification.

```powershell
python scripts/convert_im0_to_jpeg.py `
  --input "C:\Users\Admin\Downloads\Xrays\XRAY Data" `
  --output "Parallax\X-ray data" `
  --group-by-source-folder `
  --image-name PA
```

This creates a flat structure such as `Patient_000/PA.jpg`, with one anonymized folder for every source folder that contains a successfully converted IM0 image. If a source folder ever contains multiple images, later images become `PA_001.jpg`, `PA_002.jpg`, and so on.

The script writes anonymized `Patient_###/Image_###.jpg` files by default and `preview_contact_sheet.jpg`; it does not infer PA, AP, or Lateral labels. It creates [reports/dataset_cleanup_report.md](reports/dataset_cleanup_report.md) with counts, anonymized failure information, rendering details, and grouping ambiguity status. A missing DICOM `PatientID` stops default patient grouping before output is created, rather than inventing patient groups, and a source declaring `BurnedInAnnotation=YES` is reported instead of converted.

For display conversion, the script applies DICOM modality transforms and numeric window center/width when present, handles `MONOCHROME1` polarity, and only uses a documented robust percentile fallback when no usable DICOM window is available. JPEG is a display-oriented working copy and is not a replacement for source DICOM data.
