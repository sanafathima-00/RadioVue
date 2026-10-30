# Validation report: single-X-ray → synthetic view → Volumea

**Date:** 2026-10-30 · **Overall status:** the validation **protocol and code are complete and unit-tested**, but **no model-quality validation has been executed**. There is no GPU session and no authorised paired data in Colab. No claim is made that the pipeline improves CT consistency.

## 1. What is validated today

| Claim | Evidence type | Result |
|---|---|---|
| Control flow, refusals, provenance | Unit tests (48, simulated models) | Pass |
| AP is never relabelled PA; conflicts → UNKNOWN | Unit tests | Pass |
| Synthetic images are always labelled; originals preserved byte-identical | Unit tests + executed dry run | Pass |
| Frontal/lateral classifier works on real radiographs | Executed, n = 2, one subject | Correct, p ≥ 0.999 |
| Anatomical segmentation produces plausible masks | Executed, n = 1, visual check | Plausible |
| Row-profile consistency discriminates real pairs | Executed, n = 1 pair | **Fails**, so made advisory |
| Pooled CNN embeddings encode laterality | Executed, n = 1 | Weak (mirror cosine 0.971) |

## 2. Planned tests (notebook 03, ready to run)

Test arms, each changing **one variable** relative to its neighbour. All are scored against the same CT ground truth:

| Arm | Frontal (slot 0) | Lateral (slot 1) | Answers |
|---|---|---|---|
| **A_native** ("existing input configuration") | DVG-projector DRR | DVG-projector DRR | DVG ceiling in its own domain |
| **C_svdrr** ("actual complementary view") | SV-DRR PA DRR | true SV-DRR lateral | cost of DRR style / field-of-view mismatch (vs A) |
| **B_synth** ("generated complementary view") | SV-DRR PA DRR | Parallax lateral | **cost of synthesis** (vs C) |
| **B_ap** | AP-like DRR (SID 100 cm) | Parallax lateral from AP | **PA vs AP sensitivity** (vs B) |
| **C_ap** | AP-like DRR | true lateral | AP frontal effect with a true lateral |

Metrics:

- **2D (synthetic vs true lateral):** PSNR, SSIM, NCC, LPIPS (AlexNet), seed spread. **Patient specificity**: top-1 identification of the correct patient among N, against chance 1/N. This tests "patient-specific vs generic".
- **3D (vs CT):** MAE, RMSE, PSNR, 3D SSIM, NCC, and Dice of lung-like (0.04–0.25) and bone-like (≥ 0.6) voxel classes in DVG units. The tissue Dice is a coarse anatomical-consistency proxy, not a segmentation.
- **QC calibration:** the row-profile distribution for PA→synthetic and PA→true lateral, compared with PA→other patients' laterals.
- **Geometry calibration:** dihedral transforms and field-of-view scale for every renderer and slot. Margins are reported. Low margins mean "inconclusive".

Interpretation rules, fixed in advance:

- SSIM/PSNR improvements alone are **not** evidence of anatomical correctness.
- **B ≈ C** together with patient-specificity well above chance supports the pipeline.
- **B ≈ A-with-a-random-lateral**, or patient-specificity near chance, indicates generic laterals. That would be logged as a failed experiment.
- **B_ap clearly worse than B_synth** confirms AP sensitivity, and would motivate AP-geometry fine-tuning (NEXT_STEPS.md).

## 3. Data constraints

- **LIDC-IDRI is contaminated** for joint evaluation: all SV-DRR validation patients are in X2CT/DVG training.
- **Local CTs (12 studies, 10 patients)** are clean for both models, but contain identifiable data. **Uploading them requires your explicit authorisation.** The agent did not upload or process their pixels.
- **Real-radiograph validation** has one public CC0 PA/lateral pair, with no CT. It can support a qualitative and 2D comparison of the synthetic lateral vs the real lateral (notebook 04 + manual comparison), but not CT metrics.

## 4. Failure cases already identified (by design or by evidence)

| Case | Behaviour |
|---|---|
| Lateral input | Stops (`stopped_no_supported_synthesis`). No validated generator exists |
| PNG/JPEG without a declared view | `FRONTAL_UNSPECIFIED`, low confidence, warning. Processed as frontal |
| Declared view conflicts with DICOM or pixels | `UNKNOWN` → stop |
| Synthetic view copies the input / disagrees across seeds / blank | QC refusal. Volumea not run |
| Missing GPU components | Status `stopped_*` with the exact load error |
| Different FOV between real views | Row-profile metric unreliable (E3) |

## 5. Pending clinical evaluation

None is planned or appropriate at this stage. Outputs are research artefacts. Any clinical study would need ethics approval, a prospective protocol, and radiologist reading. That is outside this task.
