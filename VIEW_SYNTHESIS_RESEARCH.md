# View synthesis research: one chest X-ray → complementary view

**Date:** 2026-10-30 · **Status:** research + selected architecture. Generation on GPU has not been executed in this task (see EXPERIMENT_RESULTS.md).

Labels: **[PAPER]** peer-reviewed or preprint text, read directly · **[REPO]** official code or model card · **[CODE]** this repository · **[EXEC]** executed in this task · **[HYP]** hypothesis · **[INF]** inference.

---

## 1. The problem, stated precisely

A radiograph is a 2D line-integral projection. Every voxel's depth along the beam is collapsed. A lateral view therefore contains information that **no** frontal image determines uniquely. Any generated lateral is a *conditional estimate*: patient-specific where the frontal constrains it (e.g. cranio-caudal positions, thoracic height, diaphragm level), and prior-driven where it does not (e.g. antero-posterior depth of structures, retrocardiac anatomy). Two consequences shape the whole design:

1. "Looks like a lateral radiograph" ≠ "is this patient's lateral anatomy". Evaluation must test **patient specificity**, e.g. whether a synthetic lateral is closer to its own patient's ground truth than to other patients'. Realism alone is not evidence.
2. Uncertainty is intrinsic and must be shown. The pipeline draws several seeded samples, and publishes their per-pixel spread as an uncertainty map.

One physical constraint *is* shared between views [INF, derived from line-integral geometry]. For parallel-beam projections that rotate about the cranio-caudal axis, the sum along every detector row equals the total attenuation of that axial slab, independent of the angle. This was implemented as a consistency check (`consistency.row_profile_consistency`) and **tested**. It holds for DRRs [EXEC, unit test, error < 2%]. It does **not** reliably discriminate on real, independently acquired radiographs [EXEC: the true lateral scored r = 0.65, a vertically flipped lateral r = 0.87]. It is therefore advisory only. See `research/failed-experiments/2026-10-30-row-profile-qc-real-radiographs.md`.

---

## 2. XraySyn (Peng et al., AAAI 2021, arXiv:2012.02407)

| Aspect | Finding |
|---|---|
| Architecture [PAPER] | Two stages. **3D PriorNet (3DPN)** estimates a coarse 3D CT from one radiograph, using a differentiable back-projector/forward-projector pair inside **CT2Xray** (a modified, differentiable DeepDRR with bone/tissue material decomposition). **2D ReﬁneNet (2DRN)** refines the re-projected novel view, adversarially, toward real-radiograph appearance and keeps the material decomposition. |
| Training data [PAPER] | 3DPN uses 780 LIDC-IDRI CTs (700 train / 10 eval / 70 test), cropped and reshaped to 128³. 2DRN uses 3,000 manually selected healthy TBX11K radiographs (256²), **unpaired**. |
| View range [PAPER] | Training poses are sampled in **−18°…+18° azimuth and elevation**. On real radiographs, testing assumed a canonical frontal input and **−9°…+9° azimuth** outputs. |
| Inference [PAPER/REPO] | Single radiograph in. Estimated volume plus novel views, bone and tissue out. No CT needed at inference. |
| Code/weights [REPO] | `github.com/cpeng93/XraySyn`, MIT licence. Weights on a SharePoint link. Environment: Ubuntu 16.04, Python 3.7, CUDA 10.0, PyTorch 1.3.1. |
| In this repo [CODE] | XraySyn's **CUDA DRR projector is already vendored** in `Volumea/xraysyn/`, and DVG-Diffusion uses it for forward and back-projection. |
| Stated limitations [PAPER] | Resolution is limited by 3DPN memory. 3DPN views are blurry. Real-radiograph evaluation has no ground truth (FID only). |

**Applicability to RadioVue.**

- **PA → lateral (90°): not supported.** The method was trained and validated only within ±18° (tested on real images at ±9°). A 90° rotation is far outside its training distribution, and reporting it as XraySyn output would be fabrication-by-extrapolation.
- **Reusable methodology:** the 3DPN idea (an explicit 3D estimate plus differentiable projection) is essentially what DVG-Diffusion does with two views. XraySyn's projector is reused already.
- **Colab reproduction:** possible in principle, but the PyTorch 1.3 / CUDA 10 stack and external SharePoint weights make it costly. It would not answer our task. **Not pursued.**

---

## 3. Approaches compared

| | A. CT-informed DRR generation | B. Conditional diffusion view synthesis (SV-DRR / Parallax) | C. Anatomy-aware image-to-image | D. VLM-guided synthesis | E. Hybrid (embeddings + masks + generator) |
|---|---|---|---|---|---|
| What it is | Render views from a CT of the same patient | DiT conditioned on input image (VAE + CLIP) + relative pose; trained on LIDC DRRs | Encoder-decoder conditioned on input + masks | Generator conditioned on VLM structured text | Generator conditioned on image, masks, embeddings |
| Patient-specific? | **Exact** (if the patient has a CT) | Yes, via image conditioning; strength unmeasured | Yes, in principle | Weak: text is a lossy bottleneck | Yes, in principle |
| PA→lateral | Exact | **In pose distribution** (trained ±90° from PA) | Needs training | Needs training | Needs training |
| AP→lateral | Exact (AP geometry renderable) | Out of distribution (trained on PA only) | Needs AP data | Needs training | Needs training |
| Lateral→frontal | Exact | **Out of distribution** (condition was always the PA view) | Needs training | Needs training | Needs training |
| Data needed | A CT of *this* patient (unavailable at inference) | None to use. Fine-tuning needs CT→DRR pairs | Paired views | Paired views + text | Paired views |
| GPU | Light | ~4–6 GB fp16 at 512 [INF]; ran on T4 at 256 [COLAB] | Train: large | Train: large | Train: large |
| Training complexity | None | None (pretrained) | High | Very high | High |
| Available today | Validation data generator only | **Yes. Weights public, code in repo** | No | No | No |

Notes:

- **A** cannot be the inference path: if the patient had a CT, we would not need the pipeline. It is the right tool for producing **paired, geometrically exact validation data** (PA, AP-like, lateral DRRs plus the CT ground truth). It is implemented as `drr.py`, and the `validation.py` / notebook 03 design uses it.
- **D** is the hypothesis that motivated this task. The evidence does not support it as a *generator conditioning* today:
  - (i) no pretrained model conditions on structured anatomical text;
  - (ii) MedGemma's anatomy output is coarse (boxes, words), lossy relative to pixels, and its confidence is uncalibrated [REPO model card: CXR anatomy localization IoU 38.0 on Chest ImaGenome];
  - (iii) training such conditioning needs paired lateral data that RadioVue lacks.

  VLM output is used instead for what it does well: **view evidence, anatomical description, QC and provenance**.
- **MedSigLIP** embeddings could replace the CLIP conditioning signal in SV-DRR. That requires retraining `cc_projection` and the DiT cross-attention [CODE: `Parallax/pipeline.py:453-500`]. Probe experiment: notebook 02. A local CNN baseline [EXEC] showed the conventional pooled CXR embedding barely separates a PA from its mirror image (cosine 0.971). Global embeddings are therefore weak carriers of geometry. This motivates keeping spatial (latent-image) conditioning.

---

## 4. Selected architecture and justification

**Parallax (SV-DRR) is the generator. It is wrapped with anatomy-aware gating, multi-seed uncertainty, explicit pose calibration, and provenance.** The VLM and segmentation feed routing, QC and documentation, not pixel generation.

```text
input X-ray ─► load/render (DICOM-safe) ─► view evidence fusion
                                            (DICOM ▸ declaration ▸ TorchXRayVision frontal/lateral ▸ MedGemma [weak])
            ─► MedGemma 1.5 structured extraction (semantic JSON + official anatomy boxes, frontal only)
            ─► PSPNet anatomical masks (frontal only)  ─► box-vs-mask agreement
            ─► route policy: PA→lateral (in-distribution) | AP→lateral (experimental, labelled) |
                             lateral→frontal (OOD, disabled by default) | unknown/oblique (stop)
            ─► Parallax DiT ×N seeds at the calibrated azimuth ─► per-pixel std (uncertainty)
            ─► QC gate (sample agreement, not-a-copy, dynamic range; row-profile advisory)
            ─► SYNTHETIC labelling + provenance ─► Volumea adapter (or refusal)
```

Why this, and not something new:

1. It is the **only available pretrained, pose-conditioned, image-conditioned** chest radiograph view-synthesis model, and it is already in the project.
2. It was trained on exactly the PA→±90° geometry Case A needs [CODE `Parallax/dataset.py:340-361`].
3. It conditions on the full-resolution latent of the input. That is the strongest available channel for patient specificity, much stronger than any text or global embedding.
4. Everything else in the task (classification, VLM extraction, segmentation, QC, adapter) can be validated without retraining the generator.
5. The design keeps a clean upgrade path: fine-tune SV-DRR on local-CT DRRs, AP-geometry DRRs, or lateral-conditioned pairs (§6) without changing the pipeline interface.

What it does **not** solve, recorded explicitly:

- AP and lateral inputs are out of distribution for the pretrained generator.
- Real radiographs are out of domain (DRR training).
- The generated lateral cannot be verified for a single real patient without a real lateral, or a CT, of that patient.

---

## 5. Implementation

| Item | Location |
|---|---|
| Synthesis wrapper (pose convention of `infer.py`, seeds, route policy) | `radiovue_pipeline/view_synthesis.py` |
| QC gate | `radiovue_pipeline/consistency.py` |
| DRR generation (NumPy parallel, SV-DRR DiffDRR geometry, DVG projector) | `radiovue_pipeline/drr.py` |
| Calibration (dihedral matching to DICOM-oriented references, FOV fit) | `radiovue_pipeline/calibration.py`, `validation.py` |
| Synthesis + calibration + Tests A/B/C notebook | `notebooks/03_Paired_DRR_Validation.ipynb` |
| End-to-end notebook | `notebooks/04_RadioVue_End_to_End.ipynb` |

## 6. Data requirements (to go beyond the pretrained model)

| Goal | Minimum data | Notes |
|---|---|---|
| Validate PA→lateral (no training) | ≥ 8–20 chest CTs not in LIDC (**LIDC is contaminated**: all 16 SV-DRR validation patients are in the X2CT/DVG training split, `validation_data/lidc_split/`) | The 12 local CT studies (10 patients) are suitable **if authorised** |
| Validate on real radiographs | Same-session PA + lateral pairs; ideally plus a CT within days | Public CC0 pair (1 subject, no CT) included |
| Fine-tune for AP inputs | CT → AP-geometry DRRs (short SID, anterior source) + lateral DRRs; ~hundreds of CTs | Generated with `SvdrrDrrRenderer(orientation="AP", sdd=1000)` |
| Lateral → frontal | CT → lateral-conditioned training pairs (sampling with condition ≠ 0°) | Retraining SV-DRR's sampling (method 1 in `dataset.py`) |
| Domain adaptation to real radiographs | Unpaired real radiographs (XraySyn's 2DRN idea) or paired real PA/lateral | Needs a GPU budget beyond Colab free tier |

## 7. GPU requirements (estimates unless marked)

| Stage | VRAM | Notes |
|---|---|---|
| MedGemma 1.5 4B bf16 | ~9–11 GB [INF: 4.3 B params × 2 bytes + activations] | L4/A100 native bf16. T4 lacks native bf16 (slower, *unverified*). Isolated subprocess frees memory on exit |
| Parallax DiT-fb-512 fp16 | ~4–6 GB [INF] | 256 model ran on T4 [COLAB] |
| DVG (two 3D U-Nets + 3 VQ-GANs) | ran on T4 16 GB [COLAB / PRIOR] | 100 sampling steps |
| End-to-end | ≥ 16 GB with `free_gpu_between_stages=True`; **L4 24 GB recommended** | Stages run sequentially |

## 8. Explicit limitations

- No generated view in this task has been produced or inspected yet (GPU blocked; see EXPERIMENT_RESULTS.md).
- The pretrained generator has never been evaluated on real radiographs against a real lateral.
- QC thresholds are provisional. Row-profile consistency is advisory.
- The lateral azimuth sign, the generator output orientation, and the DVG slot transforms are **uncalibrated** until notebook 03 runs.
- DVG reconstructs 128³ in normalised units (not HU). Its frontal pose is a DRR geometry, neither a clinical PA nor an AP.
