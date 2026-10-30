# Next steps

## A. Needed from you (blocking GPU execution)

1. **Hugging Face:** accept the HAI-DEF terms on `google/medgemma-1.5-4b-it` and `google/medsiglip-448`. Create a read token and add it as the Colab Secret `HF_TOKEN`.
2. **Code delivery:** approve a commit + push of the new branch, **or** upload `dist/radiovue_colab_bundle.zip` to `MyDrive/RadioVue/code/` (SETUP_GUIDE §1).
3. **Validation data:** decide whether the 12 local CT studies may be uploaded to your Drive for notebook 03. Remove direct identifiers first; the folder names are patient names. Alternatively, download a non-LIDC public chest CT set (e.g. TCIA NSCLC-Radiomics).
4. **Colab execution:** run notebooks 01 → 03 → 04 (cell 5 demo only), or give the agent an authenticated browser session, e.g. Claude in Chrome with your Google login, so it can drive Colab.

## B. Then, in order

| # | Step | Success criterion | If it fails |
|---|---|---|---|
| 1 | Notebook 01 | Parseable JSON on both CC0 images. Box-vs-mask IoU reported. Greedy reruns identical | Record in the failed-experiment log. Adjust prompts (the model is prompt-sensitive) |
| 2 | Notebook 03 calibration | Consistent lateral azimuth and transforms across cases, with margins > 0.05 | Inspect the renders. Do not run 04 on your image until it is resolved |
| 3 | Notebook 03 Tests A/B/C | Patient-specificity top-1 ≫ chance. B_synth close to C_svdrr | Log as failed. Consider fine-tuning (C below) |
| 4 | Re-gate QC | Make row-profile gating only if it separates true vs mismatched pairs on DRRs | Keep it advisory |
| 5 | Notebook 04 on the CC0 PA | `completed`. Compare the synthetic lateral with the real CC0 lateral (2D, qualitative) | Check the QC block and warnings |
| 6 | **Your X-ray** in notebook 04, cell 6 | — | — |

## C. Research directions (evidence-gated)

- **AP-geometry fine-tuning** of SV-DRR on CT → AP-like DRRs (`SvdrrDrrRenderer(orientation="AP", sdd=1000)`), if B_ap is clearly worse than B_synth.
- **Lateral-conditioned training** (SV-DRR sampling with a condition other than 0°) to make Case C supportable.
- **MedSigLIP conditioning:** replace or augment the CLIP embedding. This needs retraining `cc_projection`, and is only worth it if notebook 02 shows better anatomical sensitivity.
- **DVG field-of-view harmonisation:** choose between `repo_center_crop` and `contain_pad` (or a fitted scale) using the notebook 03 FOV fit.
- **Real-radiograph domain gap:** an XraySyn-style unpaired refinement stage, or paired real PA/lateral data.
- **Housekeeping:**
  - rename `Data/` folders to anonymised IDs;
  - scrub the patient name from the untracked `DVG_Diffusion_Colab_Inference.ipynb`;
  - commit the two Parallax shim files the published checkpoints need.
