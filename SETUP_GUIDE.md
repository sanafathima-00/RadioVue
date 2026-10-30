# Setup guide: running the single-X-ray research pipeline

## 0. What you need

| Requirement | Why | Status on 2026-10-30 |
|---|---|---|
| Google account with Colab, **GPU runtime** (L4 recommended; T4 minimum) | All model inference | You have it. The agent has no session |
| Drive folder `MyDrive/RadioVue/Volumea/models/DVG-Diffusion/` with the 5 DVG checkpoints | Volumea | Present (seen in the Volumea Colab) |
| Hugging Face account that has **accepted HAI-DEF terms** for `google/medgemma-1.5-4b-it` and `google/medsiglip-448`, plus a read token | MedGemma / MedSigLIP are gated | **You must do this** |
| Colab Secret `HF_TOKEN` (key icon → add → enable notebook access) | Token is never printed | **You must do this** |
| The new code on Colab: **either** push the branch to GitHub **or** upload the code bundle to Drive | Notebooks import `radiovue_pipeline` | **Pending your choice** |

## 1. Getting the code into Colab (choose one)

**Option A: push to GitHub** (recommended, versioned)

Commit and push the branch with `radiovue_pipeline/`, `notebooks/`, `tools/`, `validation_data/lidc_split/` and the two Parallax shim files (`Parallax/pipeline_svdrr_DiT.py`, `Parallax/svdrr_transformer_2d.py`). Then set `REPO_BRANCH` in the notebook form. Open the notebook directly:

`https://colab.research.google.com/github/sanafathima-00/RadioVue/blob/claude/single-xray-vlm-pipeline/notebooks/04_RadioVue_End_to_End.ipynb`

**Option B: Drive bundle** (no push)

```bash
python tools/make_colab_bundle.py
```

Upload `dist/radiovue_colab_bundle.zip` to `MyDrive/RadioVue/code/radiovue_colab_bundle.zip`. Then upload the notebook files from `notebooks/` to Drive and open them with Colab. The bundle is built from an **allow-list**: code and docs only, with no data, weights or images (~0.5 MB), and it includes `BUNDLE_MANIFEST.json` with file hashes.

## 2. Notebooks (run in this order)

| # | Notebook | Purpose | Est. time |
|---|---|---|---|
| 01 | `notebooks/01_MedGemma_Anatomical_Extraction.ipynb` | MedGemma on public CC0 PA + lateral. Produces the first **real** example JSON, VRAM and timing | ~15 min |
| 02 | `notebooks/02_MedSigLIP_Embedding_Experiment.ipynb` | MedSigLIP vs Parallax-CLIP vs CNN probes | ~10 min |
| 03 | `notebooks/03_Paired_DRR_Validation.ipynb` | **Calibration** (lateral sign, orientations, DVG volume orientation) + Tests A/B/C on authorised CTs | ~1–2 h for 8 CTs |
| 04 | `notebooks/04_RadioVue_End_to_End.ipynb` | Upload one X-ray → full pipeline | ~15 min setup + ~5 min per image |

Each notebook: set the runtime to a GPU, run **Runtime → Run all**, and fill in the form in cell 1 only if your paths differ.

For notebook 03, copy authorised CT DICOM series folders to `MyDrive/RadioVue/validation_ct/<any_name>/`, tick `DATA_AUTHORISED`, and run. It writes `outputs/single_xray_pipeline/calibration/geometry_calibration.json`, which notebook 04 loads automatically. **Do not use LIDC-IDRI for this:** it overlaps the models' training data (see `validation_data/lidc_split/`).

## 3. Where to upload your X-ray

Notebook 04, **cell 6 "Upload ONE chest X-ray"**:

1. Optionally set `DECLARED_VIEW` to PA, AP or LATERAL **if you know it from the acquisition**. Otherwise leave it as `auto`. DICOM `ViewPosition` takes precedence.
2. A file picker appears. Choose one DICOM, PNG or JPEG. Remove burned-in identifiers first.
3. Cell 7 runs `run_pipeline`. Cell 8 is the interactive CT slice viewer.

Cell 5 first runs the public CC0 PA, so you can confirm the installation works before your image is used.

## 4. What you should expect

- **PA input:** the original, 4 SYNTHETIC lateral samples (seed 0 is shown), an uncertainty map, the QC result, anatomy JSON with boxes and masks, and, if QC passes, a SYNTHETIC 128³ CT. Values are normalised (not HU) with unit spacing.
- **AP input:** the same, with the route labelled **EXPERIMENTAL** and the view kept as AP.
- **Lateral input:** stops after anatomy extraction, with status `stopped_no_supported_synthesis`. This is by design: no validated lateral→frontal generator exists.
- **PNG/JPEG without a declared view:** `FRONTAL_UNSPECIFIED` with a warning. Declare the view if known.
- **Without notebook 03 calibration:** every run warns that the lateral sign and the DVG orientation are UNCALIBRATED. Treat the CT orientation as unverified.

Outputs go to `MyDrive/RadioVue/outputs/single_xray_pipeline/run_<UTC timestamp>/`.

## 5. Local (CPU) development

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
.venv/Scripts/python -m pip install -r requirements-pipeline.txt
```

```bash
.venv/Scripts/python -m pytest tests -q
```

```bash
python notebooks/build_notebooks.py
```

The last command regenerates the notebooks after you edit `notebooks/build_notebooks.py`.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `PermissionError: Cannot access gated model` | Accept the terms on the model page, then check that the `HF_TOKEN` secret is enabled for this notebook |
| `import torchvision` fails after installs | `install_runtime` re-pins `numpy==2.2.6` (the fix seen in the Volumea Colab). Then restart the runtime and run all again |
| DRR projector build fails | Confirm a GPU runtime; it compiles with the active torch/CUDA. See the Volumea Colab cells this mirrors |
| CUDA OOM on T4 | Keep `free_gpu_between_stages=True`; use L4; reduce `synthesis_seeds` to 2 |
| `stopped_qc_refusal` | Expected safety behaviour. Inspect `synthetic/` and the QC block; do not bypass |
