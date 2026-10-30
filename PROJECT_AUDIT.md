# RadioVue Project Audit — Parallax, Volumea, and the single-X-ray pipeline

**Date:** 2026-10-30
**Scope:** Phase 1 of the single-X-ray → complementary view → CT research pipeline.
**Status:** Inspection only. No core model code, checkpoint, or original notebook was modified in this phase.

Evidence labels used below:

- **[CODE]**: verified by reading the code in this repository (file:line cited).
- **[COLAB]**: verified by reading the saved cells/outputs of the Colab notebook (not re-executed).
- **[UPSTREAM]**: upstream paper, model card, or repository documentation.
- **[PRIOR]**: earlier RadioVue research records under `research/`. Not re-verified in this phase.
- **[INFERENCE]**: my reasoning from the evidence above. Not executed.
- **[UNKNOWN]**: could not be established.

---

## 1. Workspace inventory

| Item | Location | Notes |
|---|---|---|
| Parallax (SV-DRR) | `Parallax/` | Upstream SV-DRR (MICCAI 2025) research code, vendored. Local `models/DiT-fb-512/` + 3.2 GB zip (git-ignored). |
| Volumea (DVG-Diffusion) | `Volumea/` | Upstream DVG-Diffusion (IEEE TIP 2026), vendored. **Contains XraySyn's differentiable DRR projector** at `Volumea/xraysyn/`. Checkpoints listed in `reports/voluma_baseline` are git-ignored. |
| X-Recon | `Recon/X-Recon/` | Separate orthogonal-view CT reconstruction (PyTorch 1.10). Not part of this task's critical path. |
| Standardization MVP | `xray_standardization/`, `tests/test_xray_standardization.py` | DICOM/PNG → 256/512 PNG for SV-DRR, PA-only gate, privacy-safe report. Reused by this pipeline. |
| Local DVG notebook copy | `DVG_Diffusion_Colab_Inference.ipynb` | Contains a real patient's name in markdown and paths. **Privacy issue — see §8.** |
| Prior research | `research/notes`, `research/decisions`, `research/failed-experiments`, `reports/voluma_baseline` | Strongly relevant; summarized in §6. |
| Local data | `Data/`, `Parallax/X-ray data/` | Clinical data; folder names contain real patient names. Inventory in §5 uses anonymized indices only. |

### Google Colab notebooks

| Notebook | Link | Access during audit |
|---|---|---|
| Volumea | https://colab.research.google.com/drive/1fpg9j0td195X0m7OFnw9Meptt9Ye701d | **Readable** (link-shared), saved cells and outputs inspected. Not executed. |
| Parallax | https://colab.research.google.com/drive/1H3KPkFAMdQpAPndgn1S7jXo9Z1cGTRNL | **Not readable.** It redirects to Google sign-in in the agent browser. Audited from local code plus `research/decisions/2026-10-13-colab-projection-routing.md`, which documents its cells. |

**Execution blocker [FACT]:** the agent has no authenticated Google session. It cannot attach a Colab GPU runtime or run any cell. MedGemma and MedSigLIP are also **gated** Hugging Face models: they need a Hugging Face account that has accepted the Health AI Developer Foundations terms, plus an `HF_TOKEN` stored in Colab Secrets. Both access steps belong to the user.

### Local hardware [FACT]

- Intel Core i7-8650U, 15.9 GB RAM, Intel UHD 620. **No NVIDIA GPU, no CUDA.**
- System Python 3.10.9 has no PyTorch. A CPU-only venv (`.venv/`, git-ignored) was created for lightweight tests only.
- Conclusion: DVG, SV-DRR, MedGemma and MedSigLIP inference must run on a Colab GPU.

---

## 2. Parallax (SV-DRR)

### Architecture [CODE][UPSTREAM]

- View-conditioned latent diffusion transformer: a PixArt-Σ-derived DiT (`svdrr_transformer_2d.py`, `transformer.py`) with 8 latent in/out channels.
- The conditioning image enters twice: as a VAE latent concatenated with noise, and as a frozen CLIP image embedding concatenated with a pose embedding through `cc_projection` (`Parallax/pipeline.py:363-500`).
- Pose embedding = `[deg2rad(Δpolar), sin(Δazimuth), cos(Δazimuth), Δradius]` (`Parallax/pipeline.py:453-485`).
- Scheduler, VAE and CLIP image encoder are shared components. Checkpoints come at 256, 512 and 1024 (`xiechun-tsukuba/svdrr-dit-fb-{256,512,1024}` on Hugging Face).

### Training distribution (critical) [CODE]

- Trained on **CT-derived DRRs** of LIDC-IDRI, rendered with DiffDRR (`scripts/create_drr_img.py`): `orientation="PA"`, SDD 1800 mm, 0.7 mm pixels, 512², ZXY Euler angles, translation `[0,-1650,0]`.
- `dataset.py:340-361`, "Sampling method 2": **the condition is always the 0° PA view.** Targets are sampled from azimuths −90…+90° in 5° steps.
- **Consequence:** pretrained Parallax has learned **PA → other views only.**
  - PA → lateral (±90°) is in-distribution geometrically. It is still out-of-domain photometrically for real radiographs.
  - AP → lateral is out-of-distribution (AP magnification and patient position differ).
  - **Lateral → frontal is out-of-distribution.** No released model was trained to condition on a lateral view.

### Input/output contract [CODE] (also `SVDRR_INPUT_CONTRACT.md`)

| Property | Value |
|---|---|
| Input format | Any PIL-readable image (`diffusers.utils.load_image`). DICOM must first go through `xray_standardization`. |
| Preprocessing | `resize((S,S))` (non-aspect-preserving, PIL default) → `convert("L")` (`infer.py:120-121`). Pipeline converts to RGB, maps to [-1,1], resizes to 224 for CLIP. |
| Size | `--image_size` 256/512/1024, matching the checkpoint. H and W divisible by 8. |
| Pose | `--simple_pose` gives azimuths −90…+90 (step 5, excluding 0). `query_pose = [0,0,0] - [0, az, 0]` unless `--flip_pose`. README: "polar angle increases downward, azimuth increases rotating to the left". |
| Inference | fp16, CUDA, xFormers optional, 30 steps, guidance 3.0. **No seed exposed in CLI.** |
| Output | One 8-bit PNG per pose. No uncertainty, no provenance metadata. |
| GPU | 256 model ran on a Colab **T4** [COLAB]. The 512 model is recommended on ≥16 GB with fp16 [INFERENCE]. |
| Dependencies | diffusers 0.30.3, transformers 4.46.2, accelerate 1.1.1, kornia 0.7.1. Colab needed a `transformers` force-reinstall and a shim `pipeline_svdrr_DiT.py` [COLAB]. |
| Validation | None in-repo for real radiographs. `EVALUATION_REPORT.md` states that no SV-DRR comparison has been run. |

### Reusable components

- `SvdrrDiTPipeline` is the **only pretrained, pose-conditioned, patient-conditioned novel-view generator available in the project.** It is reused as the view-synthesis engine.
- `scripts/create_drr_img.py` defines the exact DRR geometry Parallax was trained on. It is reused to make in-domain paired DRR validation data.
- `xray_standardization` + the 2026-10-13 routing decision provide a PA/AP/lateral routing policy and a privacy-safe report. Both are reused.
- Not present in Parallax: segmentation, landmark detection, a view classifier, or a VLM.

### Limitations

- Pixel-level PA vs AP vs lateral detection is absent.
- Square resize distorts anatomy for non-square inputs.
- Domain shift exists between real radiographs and DRRs.
- The left vs right lateral sign convention of ±90° relative to clinical laterality is **[UNKNOWN]** and must be calibrated (§4).

---

## 3. Volumea (DVG-Diffusion)

### Architecture [CODE]

1. Each input X-ray is min-max normalized and **back-projected** into a 128³ volume with XraySyn's CUDA DRR projector, at a fixed pose. `add_bp_from_real_xray(xray, 2)` sums the two volumes (`Volumea/ddpm/diffusion.py:820-850`).
2. The sum is encoded by `VPGE_0_90_VQGAN` into an 8×32×32×32 latent.
3. The coarse 3D U-Net (`NewView_Coarse_Model.pt`, dim 64) predicts a coarse CT latent. CT-VQGAN decodes it.
4. The coarse CT is forward-projected at the **45° intermediate pose (−0.25π)**, back-projected, and encoded with `VPGE_45_VQGAN`.
5. The fine 3D U-Net (`DVG_Diffusion.pt` EMA, dim 128, **24-channel input** = 8 + 8 + 8 noisy) runs 100 cosine-schedule steps → CT latent → CT-VQGAN decode.
6. Output is clipped to [0,1], values < 0.04 → 0, and > 0.87 → 1 (`sample_from_xray`).

### Input contract for real X-rays (`FineStage.sample_from_xray`) [CODE]

| Question | Answer |
|---|---|
| Requires two views? | **Yes.** `add_bp_from_real_xray(xray, 2)` is hard-coded. The released checkpoints are the biplanar configuration only [PRIOR, 2026-09-14 note]. Single-view input is not supported. |
| View order | `xray[0]` → pose **−0.5π ("正面" = front)**, `xray[1]` → pose **0 ("右侧面" = right side)** (`diffusion.py:356-363,643-663`). **Order is not interchangeable.** Swapping it changed the volume by MAE 0.137 and Pearson r 0.469 [COLAB, Experiment 1B]. |
| Accepts AP? | **[UNKNOWN] / not validated.** The projector's frontal pose is a synthetic DRR at source-to-detector 1200 (projector units). Training conditioning came only from DRRs of the ground-truth CT (`FineStage.forward` → `add_bp(ct,2)`). The geometry is neither a PA nor an AP clinical acquisition. The paper text calls its frontal view "AP" [PRIOR], but this is a DRR label, not acquisition physics. AP is therefore *accepted by the tensors* but *unvalidated*. |
| Same patient? | Implicitly yes. The two back-projections are summed into one volume. Two different patients' views would yield an incoherent condition. |
| Geometry | Cone-beam projector, 128² detector, pixel size 1.0, 128³ volume, `source_to_detector_distance=1200`, 6-DoF T matrix with angles ×π (`diffusion.py:617-622,924-964`). Laterality/mirroring of the frontal pose relative to clinical display is **[UNKNOWN]** and must be calibrated (§4). |
| Image size | Each view is `(1,1,128,128)` float in [0,1]. The list/stack is indexed `[0]`, `[1]`. |
| Internal normalization | Per-view **min-max** inside `xray2bp_from_real` (so global intensity scaling is removed). No registration. |
| Preprocessing in code | `SingleDataGenerator_real`: grayscale → BOX-halving + bicubic + center crop to 160 → center crop 128 → `ToTensor` (`dataset/lidc.py`). This crop **discards the outer ~20%** of the field of view. |
| Preprocessing used in Colab | `ImageOps.contain` to 128 (LANCZOS) + black pad to square [COLAB]. **This differs from the repository loader.** Neither has been validated against the projector's training field of view. |
| Output | `(1,1,128,128,128)` float [0,1], normalized, **not HU**, unit spacing, axis 0 is "axial" by repository convention; `val()` additionally flips axis 0 before saving. |
| GPU | Ran on Colab **T4** (16 GB) with Torch 2.11 + CUDA 12.8/13.0 [COLAB][PRIOR]. Checkpoints ≈ 3.8 GB total. |
| Dependencies | The custom CUDA extension `drr_projector_function` must be compiled (`pip install --no-build-isolation .../drr_projector`), then torch shared libraries preloaded with `ctypes` [COLAB]. Also hydra/omegaconf/monai/pytorch-lightning/einops/rotary-embedding-torch/ema-pytorch/SimpleITK. NumPy was pinned to 2.2.6 to keep torchvision working [COLAB]. |
| Checkpoint loading | `load_state_dict(ckpt["ema"])`, then the coarse checkpoint's matching keys are overlaid (`Trainer.load_coarse` semantics) [COLAB]. |
| Validation procedure | `train/val_ddpm.py` → `Trainer.val()` on LIDC HDF5 (`ct`, `xray1`, `xray2`). It saves volumes and GIFs; **it computes no metrics** [CODE]. No LIDC HDF5 exists on Drive, so the synthetic-DRR baseline is **blocked** [COLAB, Experiment 2]. |

### Executed history (from Colab outputs and prior records)

- Experiment 1A/1B: one real PA + real lateral JPEG pair → finite 128³ volume. View order matters. Quality is far below diagnostic CT [PRIOR failed-experiment 2026-08-27].
- Parallax 256 was run in the Volumea notebook on a real PA at poses −90 and +90. Outputs were saved to Drive [COLAB]. Which pose corresponds to DVG's "right side" was **not** established.
- Earlier single-view DVG orchestration (coarse-only, 45°/90° re-projection of the coarse CT) exists in the local notebook copy. It is out-of-distribution by design.

### Integration constraints

- Removing the internal 45° intermediate-view feedback would change the 24-channel input and invalidate the checkpoint [PRIOR].
- Fine-tuning on real paired radiographs needs `FineStage.forward` changes, because training never reads `xray1`/`xray2` [PRIOR].
- **Do not modify Volumea.** All integration happens in an external adapter that calls `sample_from_xray` with a correctly ordered, correctly oriented pair.

---

## 4. Geometry conventions that must be calibrated before validation

These are **not** established by any code or prior run. The pipeline treats them as explicit configuration with a calibration procedure, rather than guessing:

1. **Parallax lateral sign:** whether azimuth +90° or −90° renders a *left* lateral (left side to detector, the clinical default) for a PA input.
2. **DVG pose 0 laterality:** whether DVG's pose 0 ("right side") image matches a left lateral, a right lateral, or a mirrored one, in display orientation.
3. **DVG frontal mirroring:** whether DVG's pose −0.5π DRR is displayed with the patient's right on the viewer's left (radiological convention), or mirrored.
4. **Field of view:** the physical extent seen by DVG's 128² detector compared with a clinical 35×43 cm radiograph.

Calibration method (implemented in `notebooks/03_paired_drr_validation.ipynb`): render DVG-projector DRRs of a CT with a known left-sided marker and compare them with the clinical display convention. On a GPU this is a few seconds. It has not been executed.

---

## 5. Local data inventory (anonymized, technical fields only)

| Set | Content [FACT] | Usable for |
|---|---|---|
| `Data/CTs/<10 patients>/…` (12 studies) | **Chest CT** DICOM, 120 kVp. Thin 1.25 mm series (205–257 slices, 512², 0.76–0.93 mm pixels) plus 5 mm series. **One frontal CT localizer (topogram) per study** (660×888). **No lateral radiographs.** | CT → DRR (PA/AP/lateral) paired, geometrically consistent validation data, and Test C ground truth (DRR lateral) plus CT ground truth for volumetric metrics. **Requires explicit authorization to upload to Colab/Drive.** |
| `Data/CTs/<case 00> PNG/` | The same case as PNG slices (257 and 65), one PA radiograph, a report, and an earlier DVG output. | Real PA + CT from the same patient, but no real lateral. |
| `Data/X-rays/PA` (491), `Data/X-rays/Lateral` (27) | Separate PA and lateral PNG sets. **Filename IDs do not overlap** (no PA/lateral pairs by ID). `Generated_Lateral/` is empty. | View-classifier sanity checks only (with authorization). |
| `Parallax/X-ray data/Patient_000…273` | 274 `PA.jpg` | PA-only. |
| `validation_data/public_cc0/` (added in this task) | One CC0 PA + one CC0 lateral radiograph of the **same subject, same session** (Wikimedia Commons, user Stillwaterising, 2010; SHA-256 recorded in `validation_data/public_cc0/PROVENANCE.md`). | Public Test A/C reference pair. **No CT** for this subject. |

**Answer to "do existing CT volumes have corresponding radiographs?"** No acquired PA/lateral radiograph pairs exist for the local CTs. Case 00 has a PA radiograph only, and every study has a frontal localizer. The CTs can therefore provide **DRR-based** paired views, and **not** real-radiograph pairs.

---

## 6. Prior findings that constrain this task

- DVG's released checkpoints implement **only the biplanar path**. Single-view CT requires a second view from elsewhere [PRIOR 2026-09-14].
- The DVG real-X-ray baseline produced a blurry 128³ volume that is not comparable to a diagnostic CT [PRIOR 2026-08-27].
- AP may go through Parallax only as a labelled experimental route. It must **never be relabelled PA** [PRIOR decision 2026-10-13].
- The 512-resolution SV-DRR comparison was blocked by Colab runtime allocation [PRIOR EVALUATION_REPORT].

---

## 7. Integration opportunities

| Opportunity | Component reused | Gap filled by this task |
|---|---|---|
| View routing | DICOM `ViewPosition` handling in `xray_standardization` | Pixel-based frontal/lateral classifier (TorchXRayVision `xinario.ViewModel`, Apache-2.0), plus MedGemma/MedSigLIP zero-shot PA/AP evidence with explicit confidence. |
| Anatomical extraction | None | MedGemma 1.5 4B structured extraction plus its trained CXR anatomy bounding boxes. Precise masks come from TorchXRayVision PSPNet (frontal only). |
| View synthesis | Parallax `SvdrrDiTPipeline` | Seeded multi-sample generation, so uncertainty becomes per-pixel variance. Pose policy, synthetic labelling, consistency QC. |
| Volumea interface | `FineStage.sample_from_xray` (unchanged) | External adapter with ordering, laterality/mirroring config, two documented preprocessing modes, refusal on failed QC, and provenance. |
| Paired data | Parallax DRR geometry + local CTs (if authorized) or public LIDC | DRR-based PA/AP/lateral triplets for Tests A/B/C with ground truth. |

---

## 8. Identified risks and hygiene issues

1. **PHI in repository files.** `DVG_Diffusion_Colab_Inference.ipynb` (untracked) contains a patient's full name in markdown and paths. `Data/` folder names are patient names. Neither is committed (`Data/` is untracked). **Recommendation:** rename data folders to anonymized IDs, scrub the notebook, and never `git add` `Data/`. This audit made no changes to these files.
2. **`Volumea.tar` (3.8 GB) is untracked in the repository root.** Add it to `.gitignore` before any `git add -A`.
3. Two inconsistent DVG preprocessing paths exist (repo crop vs Colab contain-pad). Results from them are not comparable. The adapter exposes both and records which one was used.
4. Parallax and DVG run on different PyTorch stacks in the same Colab runtime, and needed module purges and reinstalls [COLAB]. The integrated notebook installs Parallax dependencies first, then DVG, and frees GPU memory between stages.
5. The official MedGemma localization notebook's padding cell multiplies a `uint8` array by 255 (`pad_image_to_square(image) * 255`). That wraps in NumPy. The pipeline implements padding without that step.

---

## 9. Roadmap (executed in this task unless stated)

1. **Phase 2.** MedGemma 1.5 4B extraction module + notebook + MedSigLIP embedding notebook. Inference is blocked on GPU/HF access, so the code is unit-tested locally on the parsing and schema layer.
2. **Phase 3.** View-synthesis research note. Selected architecture: **Parallax SV-DRR as generator + anatomy/physics QC gate + multi-seed uncertainty**. Wrapper code and notebook.
3. **Phase 4.** Volumea adapter + paired-DRR validation notebook (Tests A/B/C) + metric code (2D/3D). Unit tests run locally. GPU validation is pending access.
4. **Phase 5.** `run_pipeline(input_xray)` + an upload-and-run Colab notebook + handoff documents.
