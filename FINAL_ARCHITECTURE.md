# Final architecture: single chest X-ray → complementary synthetic view → Volumea CT

**Status (2026-10-30):** implemented and CPU-tested. **The GPU stages have not been executed** (see EXPERIMENT_RESULTS.md). Research software, not for clinical use.

## Flow

```text
                ┌────────────────────────────────────────────────────────────┐
 input X-ray ──►│ 1 load_xray: DICOM display semantics (xray_standardization) │── ORIGINAL preserved byte-for-byte + SHA-256
 (DCM/PNG/JPG)  └────────────────────────────────────────────────────────────┘
                         │
                ┌────────▼─────────────────────────────────────────────────────┐
                │ 2 view evidence fusion  (view_classification.fuse_view_evidence)│
                │   DICOM ViewPosition ▸ declared ▸ TorchXRayVision frontal/lateral│
                │   ▸ MedGemma (weak). PA vs AP is never inferred from pixels alone│
                └────────┬─────────────────────────────────────────────────────┘
                ┌────────▼─────────────────────────────┐  ┌──────────────────────────────┐
                │ 3 MedGemma 1.5 4B (isolated venv)     │  │ PSPNet anatomy masks (frontal)│
                │   semantic JSON + anatomy boxes       │  │   + box-vs-mask IoU           │
                └────────┬─────────────────────────────┘  └──────────────┬───────────────┘
                         └───────────────► anatomical_extraction ◄───────┘
                ┌────────▼─────────────────────────────────────────────────────┐
                │ 4 plan_synthesis: PA→LAT | AP→LAT (experimental) |             │
                │   LAT→FRONTAL (OOD, off) | UNKNOWN/OBLIQUE → stop               │
                └────────┬─────────────────────────────────────────────────────┘
                ┌────────▼─────────────────────────────────────────────────────┐
                │ 5 Parallax SV-DRR DiT × N seeds @ calibrated azimuth          │── SYNTHETIC_*.png (+display banner,
                │   → output transform to display orientation                   │   PNG tags, provenance JSON),
                │   → QC gate: sample agreement, not-a-copy, dynamic range      │   uncertainty std map
                │     (row-profile advisory)                                    │
                └────────┬─────────────────────────────────────────────────────┘
                   QC fail ──► STOP, report, nothing passed to Volumea
                ┌────────▼─────────────────────────────────────────────────────┐
                │ 6 Volumea adapter: slot0 = frontal (pose −0.5π),              │
                │   slot1 = lateral (pose 0); documented preprocessing mode;    │
                │   calibrated dihedral transforms; refuses bad provenance      │
                └────────┬─────────────────────────────────────────────────────┘
                ┌────────▼────────────────────────┐
                │ 7 DVG-Diffusion (unchanged)      │── SYNTHETIC_ct_volume (128³, normalised, not HU)
                │   sample_from_xray, seed 42      │   + previews + ct_provenance.json
                └─────────────────────────────────┘
```

## Components

| Module | Responsibility | Runs on |
|---|---|---|
| `radiovue_pipeline/pipeline.py` | `run_pipeline(input_xray, config, components, declared_view)`. Lazy GPU loading, stage-wise memory release | CPU orchestration |
| `io_utils.py` | DICOM/PNG/JPEG loading via `xray_standardization` (allow-listed metadata only) | CPU |
| `provenance.py` | Original preservation, SYNTHETIC labelling (filename + PNG tags + banner + sidecar) | CPU |
| `view_classification.py` | Evidence fusion. TorchXRayVision `xinario.ViewModel` (Apache-2.0) | CPU/GPU |
| `anatomy_extraction.py` | MedGemma prompts, parsing, coordinate rejection, box mapping, schema | GPU |
| `medgemma_worker.py` | MedGemma in an isolated interpreter (`transformers>=4.50`) | GPU |
| `segmentation.py` | ChestX-Det PSPNet masks and pixel measurements (frontal only) | CPU/GPU |
| `view_synthesis.py` | Parallax wrapper, route policy, `infer.py` pose convention, seeds | GPU |
| `consistency.py` | QC gate | CPU |
| `volumea_adapter.py` | DVG ordering, preprocessing, transforms, refusal, reconstruction, saving | GPU |
| `drr.py`, `calibration.py`, `validation.py` | CT→DRR (3 renderers), orientation calibration, Tests A/B/C | CPU/GPU |
| `metrics.py` | PSNR/SSIM/NCC/LPIPS, 3D metrics, tissue-class Dice, patient specificity | CPU |
| `embeddings.py` | MedSigLIP / Parallax-CLIP / DenseNet probes | GPU/CPU |
| `colab_setup.py` | Pinned installs, checkpoint staging + fingerprint check, projector build, HF access check | Colab |

## `run_pipeline` output

```json
{
  "status": "completed | stopped_* | failed_* | rejected_input",
  "run_dir": "OUTPUT_ROOT/run_<UTC>",
  "input_view": "PA | AP | LATERAL | FRONTAL_UNSPECIFIED | UNKNOWN",
  "view_decision": {"view": "", "family": "", "confidence": "", "sources": {}, "warnings": []},
  "anatomical_extraction": {"schema_version": "radiovue.anatomy.v1", "...": "see MEDGEMMA_SETUP.md"},
  "original_image": {"preserved_file": "", "sha256": "", "rendered_grayscale": "", "origin": "ORIGINAL"},
  "generated_view": {"path": "SYNTHETIC_lateral_seed0.png", "display_path": "..._display.png", "origin": "SYNTHETIC"},
  "generated_view_type": "LATERAL | FRONTAL_SYNTHETIC_DRR_LIKE | null",
  "view_synthesis_metadata": {"route": "", "in_distribution": false, "azimuth_deg": 90, "seeds": [], "uncertainty_map": ""},
  "ct_reconstruction": {"npy": "", "nifti": "", "preview_axis0": "", "provenance": ""},
  "validation_results": {"synthesis_qc": {"passed": true, "checks": {}}, "ct_volume_stats": {}},
  "warnings": [],
  "disclaimer": "Research output only ..."
}
```

Every run folder holds `config.json`, `result.json`, `anatomical_extraction.json`, `original/`, `synthetic/`, `segmentation/` and `ct/`.

## Safety invariants (enforced in code and unit-tested)

1. AP is never relabelled PA. A declaration may not override DICOM, and conflicts give `UNKNOWN`.
2. Every generated image carries the SYNTHETIC label in its filename, PNG text, sidecar, and the banner on its display copy.
3. Volumea is never given a synthetic view that failed QC, two synthetic views, or a view from a disallowed route.
4. The original upload is preserved byte-identical and hashed. The original filename is not recorded.
5. Only allow-listed technical DICOM metadata is stored.
6. Uncalibrated geometry produces explicit warnings on every run.
