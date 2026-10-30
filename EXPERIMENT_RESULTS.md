# Experiment results: single-X-ray pipeline (2026-10-30)

Every entry states whether it was **EXECUTED**, a **UNIT TEST** (simulated inputs), or **NOT EXECUTED** (with the blocker). No GPU model output exists from this task.

## Executed (local CPU, Intel i7-8650U, Python 3.10.9, torch 2.14.1+cpu, torchxrayvision 1.5.7)

### E1. Pixel view classification on public CC0 radiographs: EXECUTED

Model: TorchXRayVision `xinario.ViewModel` (ResNet-50, Apache-2.0). Input is padded to square to keep all anatomy. Results are in `validation_data/results/local_cpu_2026-10-30/view_and_segmentation_results.json`.

| Image | P(Frontal) | P(Lateral) | Fused decision | Time |
|---|---:|---:|---|---:|
| CC0 PA | 0.99899 | 0.00101 | `FRONTAL_UNSPECIFIED` (low): PNG has no view metadata, and PA vs AP is not inferred from pixels | 0.21 s |
| CC0 lateral | 0.00010 | 0.99990 | `LATERAL` (medium) | 0.22 s |

n = 2 images from one subject. This shows the component works; it is **not** an accuracy estimate.

### E2. Anatomical segmentation of the CC0 PA: EXECUTED

TorchXRayVision ChestX-Det PSPNet, 14 structures, 29 s on CPU. The overlay was visually checked: lungs, heart, clavicles, scapulae, trachea, spine and diaphragm all lie in anatomically plausible positions. Pixel-space derived values: mask-width cardiothoracic ratio 0.371, lung apex row 143, base row 1876 (of 1956). These are model outputs, not measurements against ground truth, and they are not in mm.

### E3. Row-profile QC metric on a real same-subject PA/lateral pair: EXECUTED → negative result

| Comparison | r (fixed frame) | r (scale 0.6–1.6 + shift ±15% search) |
|---|---:|---:|
| PA vs true lateral | 0.451 | 0.655 |
| PA vs vertically flipped lateral (wrong) | 0.046 | **0.868** |
| PA vs uniform noise | 0.074 | 0.142 |

Conclusion: on real radiographs with different fields of view the metric is not discriminative. It was made advisory. Details: `research/failed-experiments/2026-10-30-row-profile-qc-real-radiographs.md`.

### E4. Conventional CNN embedding probe: EXECUTED

TorchXRayVision DenseNet-121 (`densenet121-res224-all`), globally pooled, 1024-d. Cosine similarity:

| | PA | lateral | PA mirrored | PA upside-down | noise |
|---|---:|---:|---:|---:|---:|
| PA | 1.000 | 0.799 | **0.971** | 0.845 | 0.274 |

The pooled CNN embedding barely separates a PA from its mirror image, so it carries little laterality or geometry information. One subject, so this is descriptive only. MedSigLIP and Parallax-CLIP comparisons are in notebook 02 (not executed).

### E5. End-to-end dry run on the CC0 PA with real component loading: EXECUTED

`run_pipeline` with `Components.load()` on the CPU laptop. The classifier and segmenter loaded and ran. MedGemma (`transformers` absent), Parallax (`kornia` absent / no CUDA) and Volumea ("requires a CUDA GPU runtime") were recorded as unavailable. Status `stopped_synthesizer_unavailable`: nothing was generated and nothing was passed to Volumea. The original was preserved byte-identical with SHA-256. Run folder: `validation_data/results/local_cpu_2026-10-30/run_pipeline_dryrun/`.

### E6. LIDC contamination check: EXECUTED (data analysis)

All 16 SV-DRR validation patients (`Parallax/eval_patients.json`) are in the X2CT-GAN **training** split (`github.com/kylekma/X2CT` `3DGAN/data/train.txt`). No LIDC patient is held out by both models. LIDC therefore cannot give an uncontaminated end-to-end test. The X2CT split is *assumed* to be DVG's, because DVG uses X2CT preprocessing; DVG does not publish its own split. Files: `validation_data/lidc_split/`.

### E7. Local data inventory: EXECUTED (technical DICOM fields only)

There are 12 chest CT studies (10 patients) with thin 1.25 mm series and one frontal localizer each. No lateral radiographs. No PA/lateral pairs exist in `Data/X-rays` (IDs do not overlap). Details: `PROJECT_AUDIT.md` §5.

## Unit tests: 48 passed (`tests/`, 30 s)

38 pipeline tests and 10 existing standardization tests. They cover:

- parallel-beam row-sum invariance;
- QC behaviour (copy rejection, sample disagreement, advisory row-profile);
- dihedral calibration recovering hidden transforms;
- field-of-view fit;
- metrics (identity cases, generic-generator detection by patient specificity);
- MedGemma parsing (thinking trace, fences, outermost JSON), coordinate rejection, box mapping through padding, the official prompt format;
- view fusion (7 cases, including AP never relabelled PA);
- route policy, synthetic labelling, DVG preprocessing modes, adapter refusals;
- five end-to-end control-flow runs with fake models (PA completes; inconsistent synthesis refused; lateral stops; AP labelled experimental; lazy GPU failure reported).

The tests use **simulated** model outputs. They verify control flow and safety, not model quality.

A real bug was caught by these tests and fixed: JSON extraction picked the inner `{...}` instead of the outer `[...]` list in unfenced MedGemma answers.

## NOT EXECUTED (blockers)

| Experiment | Notebook | Blocker |
|---|---|---|
| MedGemma 1.5 4B inference (example JSON, VRAM, timing, reproducibility) | 01 | No authenticated Colab session for the agent. Model gated (HAI-DEF acceptance + `HF_TOKEN`) |
| MedSigLIP embeddings / zero-shot | 02 | Same |
| Parallax lateral synthesis | 03, 04 | No GPU session |
| Geometry calibration (lateral sign, DVG orientations, field of view) | 03 | No GPU session. Needs authorised CTs |
| Tests A / B / C (CT metrics, PA vs AP sensitivity, patient specificity) | 03 | No GPU session. Needs authorised non-LIDC CTs |
| End-to-end on the CC0 PA, then on your X-ray | 04 | No GPU session. **Your X-ray: intentionally not used** |

The prior Colab results that this task relies on (DVG view-order ablation, DVG/Parallax runs on a T4) are documented in `PROJECT_AUDIT.md`. They were not re-executed.
