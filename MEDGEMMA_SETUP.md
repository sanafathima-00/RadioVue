# MedGemma 1.5 4B and MedSigLIP: setup, hardware, and extraction design

## Models

| | MedGemma 1.5 4B | MedSigLIP-448 |
|---|---|---|
| Hugging Face ID | `google/medgemma-1.5-4b-it` (released 2026-01-13; greedy default since 2026-01-23) | `google/medsiglip-448` |
| Type | Gemma 3 decoder + SigLIP medical image encoder; image-text-to-text | Two-tower image/text encoder (~400 M per tower, ~0.9 B total) |
| Image input | Normalised to 896×896, 256 tokens per image | 448×448 |
| CXR-relevant capability | Description, VQA, **anatomy bounding boxes** (Chest ImaGenome; reported IoU 3.1 → 38.0 vs v1). The card says multi-image use was not evaluated, and the model is prompt-sensitive | Embeddings, zero-shot classification, retrieval. No generation |
| Licence | Health AI Developer Foundations (HAI-DEF) terms. **Gated**: log in and accept on the model page | Same |
| Intended use | Developer starting point. **Not** for direct clinical use; outputs need independent verification | Same |
| Software | `transformers >= 4.50` (Gemma 3) | SigLIP support in transformers |

Sources: model cards on huggingface.co (read 2026-10-30) and Google's official notebook `google-health/medgemma/notebooks/cxr_anatomy_localization_with_hugging_face.ipynb`.

**Vertex AI / Model Garden.** MedGemma is also deployable through Vertex AI Model Garden (official `quick_start_with_model_garden.ipynb`). That needs a GCP project with billing, and it was not set up. Colab with an `HF_TOKEN` is the lowest-cost route.

## Hardware and memory

- **Recommended:** Colab **L4 (24 GB)** or A100. Native bf16.
- **T4 (16 GB):** bf16 is not native on Turing GPUs. The model should still load (≈ 8.6 GB weights), but expect slower generation. **Not verified in this task.**
- **Estimated peak VRAM:** ~9–11 GB for one image. Notebook 01 measures and saves it (`environment.json`).
- **CPU inference** is intentionally refused by `MedGemmaClient`.
- **Version conflict:** MedGemma needs transformers ≥ 4.50. The working Parallax Colab stack pins 4.46.2. The end-to-end notebook runs MedGemma in a `--system-site-packages` venv subprocess (`radiovue_pipeline/medgemma_worker.py`). That venv reuses Colab's torch, and GPU memory is freed when the subprocess exits.

## Access steps (the user must do these; the agent cannot)

1. Sign in at huggingface.co and click **"Agree and access repository"** on both model pages.
2. Create a **read** token (Settings → Access Tokens).
3. In Colab, open the key icon (**Secrets**), add `HF_TOKEN`, and enable notebook access. The notebooks never print it.
4. `colab_setup.check_gated_access()` fails with these instructions if access is missing.

## Extraction design (`radiovue_pipeline/anatomy_extraction.py`)

Three prompts, all greedy, each SHA-256 hashed into provenance:

1. **View prompt.** Returns JSON `{view, confidence, evidence[], orientation_notes[]}`, and `UNKNOWN` is allowed. This is weak evidence only: view fusion ranks it below DICOM, a declaration, and the pixel classifier.
2. **Semantic prompt.** Coordinates and measurements are forbidden. Returns the structures, landmarks, relationships, findings, image quality, uncertainties and limitations. A sanitiser moves any coordinate-like keys to `rejected_spatial_claims`, where they are never used.
3. **Localization prompt.** Reproduced from Google's official notebook. Boxes come as `[y0,x0,y1,x1]` in 0–1000 on a square-padded image, and are mapped back to original pixels. Only valid boxes (`x0<x1`, `y0<y1`) are kept. **Frontal only**, because Chest ImaGenome is frontal. Our padding avoids a `uint8 × 255` overflow present in the official cell.

**Schema `radiovue.anatomy.v1`** (adapted from the task's initial schema):

```json
{
  "schema_version": "radiovue.anatomy.v1",
  "image_metadata": {"view": "PA|AP|LATERAL|UNKNOWN", "confidence": "low|medium|high", "confidence_calibrated": false,
                     "evidence": [], "orientation_notes": []},
  "anatomical_structures": {"left_lung": {"visible": true, "description": "", "confidence": ""}, "...": {}},
  "anatomical_landmarks": [], "spatial_relationships": [], "visible_findings": [], "image_quality": {},
  "localization": {"source": "medgemma-1.5 bounding boxes", "note": "Approximate. Not masks.", "items": [
      {"query": "right lung", "parse_ok": true, "boxes": [{"box_2d_norm1000_yxyx": [], "box_xyxy_px_original": []}]}]},
  "segmentation": {"model": "torchxrayvision chestx_det.PSPNet", "measurements_px": {}, "masks_path": ""},
  "box_vs_mask_iou": [],
  "uncertainties": [], "limitations": [], "rejected_spatial_claims": [],
  "provenance": {"model_id": "", "model_revision": "", "decoding": "greedy", "prompt_sha": {}, "software": {}}
}
```

Changes from the initial schema, and why:

- Confidence is a category flagged *uncalibrated*, not a number. A VLM's self-report is not a probability.
- Precise spatial data lives only in `localization` (approximate boxes) and `segmentation` (pixel masks from a dedicated model).
- `box_vs_mask_iou` makes box quality measurable rather than assumed. It is cross-model agreement, not ground truth.
- `rejected_spatial_claims` keeps an audit trail of invented coordinates.

## Execution status

- **Executed locally (CPU):** schema assembly, JSON parsing, box mapping, and coordinate rejection, through unit tests with simulated responses (`tests/test_radiovue_pipeline.py`).
- **Not executed:** actual MedGemma or MedSigLIP inference. **Blockers:** (1) no authenticated Colab session is available to the agent; (2) both models are gated behind the user's Hugging Face acceptance. **No example JSON from real MedGemma inference exists yet.** Notebook 01 produces it (`OUTPUT_ROOT/medgemma_examples/*_extraction.json`) once access is granted.
