# Experiment results: single-X-ray pipeline (2026-10-30)

Every entry states whether it was **EXECUTED**, a **UNIT TEST** (simulated inputs), or **NOT EXECUTED** (with the blocker).

## Executed on GPU (Colab T4, driven by Claude; raw numbers in `validation_data/results/colab_t4_2026-10-30/`)

Data: three CC0 subjects with **real same-patient PA + lateral** radiographs (Wikimedia Commons). No CT exists for them, so results compare against the real lateral, or against the CT reconstructed from the real lateral. They do **not** compare against a ground-truth CT.

### G1. Parallax PA → lateral: runs; geometry correct; patient specificity NOT demonstrated

- **Runs** on the T4: DiT-fb-512, 30 steps, **6.3 s/image**, peak VRAM 2.7 GB. It needed a shim fix: diffusers resolves `ModelMixin` from the `svdrr_transformer_2d` module namespace.
- **Geometry:** +90° and −90° give mirror-image laterals, as expected. Outputs are not copies of the input (NCC with the input PA ≤ 0.14). Seed-to-seed variation is small (mean pixel std 0.02–0.04).
- **Appearance:** DRR-style laterals with a thorax shape consistent with the PA. **Subject B (normal 21-year-old) shows hallucinated bright irregular structures in the mid-chest that are absent from her real lateral.**
- **Patient specificity:** in the 3×3 comparison of generated vs real laterals (flip/scale/shift-searched NCC), **top-1 identification = 1/3 = chance** at both azimuths. Every generated lateral matched subject A's real lateral best, and generated laterals of different patients resemble each other (NCC 0.62–0.83). n = 3, and the real laterals differ strongly in field of view and processing, so this test is weak. It provides **no evidence** of patient-specific anatomy.

### G2. Parallax lateral → frontal: does not work (as predicted from its training)

The outputs are not frontal-like. Left-right symmetry is 0.12–0.50, vs 0.86–0.92 for real PAs of subjects A and B. They stay correlated with the input lateral (NCC up to 0.57), and match the correct patient's PA no better than other patients' PAs (NCC 0.31–0.44). **Lateral input therefore remains unsupported.**

### G3. Volumea (DVG-Diffusion) with correct two-view input: runs; orientation-critical

- **Runs** on the T4: checkpoints from the public HF repo `xing0916/DVG-Diffusion`, XraySyn CUDA projector compiled on Python 3.13, **27.5 s/volume**, peak VRAM 9.1 GB. It needed a runtime `torch.load(weights_only=False)` patch for the official checkpoints (PyTorch ≥ 2.6), and a scikit-video stub. **Volumea source is unchanged.**
- **Subject A**, slot 0 = real PA. The reference CT came from the real lateral and is visually a plausible chest CT: lungs, heart, spine and mediastinum.

| Slot 1 | NCC vs reference CT | 3D SSIM | MAE |
|---|---:|---:|---:|
| Parallax lateral **−90°** | **0.55** | 0.35 | 0.117 |
| Parallax lateral +90° | 0.25 | 0.18 | 0.179 |
| Real lateral, **mirrored** | 0.20 | 0.20 | 0.181 |

- DVG is highly sensitive to lateral orientation, so the default azimuth is now **−90°** (`PipelineConfig.lateral_azimuth_deg`). With it, the CT from a generated lateral moderately agrees with the CT from the real lateral, and also looks like a plausible thorax.
- **Accuracy against a ground-truth CT is unmeasured.** That needs the paired dataset.

### G5. Patient-specificity investigation (Parallax conditioning): EXECUTED, n = 3

Raw numbers: `validation_data/results/colab_t4_2026-10-30/patient_specificity_PS1.json`. Two seeds per arm. Specificity is measured against each subject's real lateral; margin = own-patient score minus the mean over the other patients.

| Conditioning variant | Outcome |
|---|---|
| Guidance 1.5 / 6 / 9 (default 3) | No consistent gain. Guidance 9 raised diversity, but edge margins went negative (−0.08 to −0.11). |
| Aspect-preserving pad (vs stock stretch) | Worse on every measure. |
| **DRR-domain bridging** (pose-0 re-rendering of the PA, then the lateral) | Best metrics (edge margin +0.08 / +0.11; intensity top-1 up to 2/3). Subject A's downstream CT agreed better with the real-lateral CT (NCC 0.65 vs 0.55, MAE 0.096 vs 0.117, both seeds). **But it badly corrupted subject B.** It ships as an **experimental option, off by default** (`bridge_to_drr_domain`). |

**Finding used in production: pose-0 self-reconstruction fidelity.** Parallax re-rendering its own input at pose 0 separates the hallucinating subject B (edge NCC **0.13**, seed spread 0.050) from A and C (0.71 / 0.75, seed spread 0.018 / 0.028). It is now a **gating** QC check (threshold 0.4, provisional, n = 3). Inputs outside the generator's domain are therefore refused before Volumea rather than turned into a hallucinated CT. **Patient specificity of the generated lateral itself is still not demonstrated.**

### G4. MedGemma: NOT EXECUTED

`HF_TOKEN` did not yet exist in Colab Secrets (`SecretNotFoundError`). MedGemma is therefore optional in the app.

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
| MedGemma 1.5 4B inference (example JSON, VRAM, timing, reproducibility) | 01 | `HF_TOKEN` secret not yet created (HAI-DEF acceptance needed) |
| MedSigLIP embeddings / zero-shot | 02 | Same |
| Geometry calibration and Tests A / B / C against a **ground-truth CT** | 03 | Needs your paired dataset (you declined the public TCIA download) |
| AP → lateral with real AP inputs | 03 / app | No real AP radiographs available yet |
| App (`05_RadioVue_App.ipynb`) end to end in Colab | 05 | Needs the branch pushed so Colab can clone `radiovue_pipeline` |
| **Your X-ray** | 05 | Intentionally not used |

The prior Colab results that this task relies on (DVG view-order ablation, DVG/Parallax runs on a T4) are documented in `PROJECT_AUDIT.md`. They were not re-executed.
