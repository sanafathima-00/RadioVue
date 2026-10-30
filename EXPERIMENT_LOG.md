# Running experiment log: single-X-ray pipeline

Append-only. Times are UTC dates. "EXEC" means actually run, "TEST" means a unit test with simulated inputs, "NOT RUN" means a blocker applies.

| Date | ID | What | Type | Outcome | Evidence |
|---|---|---|---|---|---|
| 2026-10-30 | L0 | Workspace and data inventory (technical DICOM tags only) | EXEC | 12 CT studies / 10 patients, frontal localizers, no laterals. No PA/lateral ID pairs | PROJECT_AUDIT.md §5 |
| 2026-10-30 | L1 | Read the Volumea Colab (link-shared) | EXEC (read) | Contract and history captured. Not re-run | PROJECT_AUDIT.md §3 |
| 2026-10-30 | L2 | Open the Parallax Colab | NOT RUN | Google sign-in required for the agent | PROJECT_AUDIT.md §1 |
| 2026-10-30 | L3 | LIDC split overlap analysis | EXEC | 16/16 SV-DRR validation patients are in X2CT training | validation_data/lidc_split/ |
| 2026-10-30 | L4 | XraySyn paper review | EXEC (read) | Trained ±18°, tested ±9°. Unsuitable for 90° | VIEW_SYNTHESIS_RESEARCH.md §2 |
| 2026-10-30 | L5 | Frontal/lateral classifier, CC0 pair | EXEC | Correct on both (p ≥ 0.999) | results/local_cpu_2026-10-30 |
| 2026-10-30 | L6 | PSPNet segmentation, CC0 PA | EXEC | 14 plausible structures, 29 s CPU | same |
| 2026-10-30 | L7 | Row-profile QC on a real pair | EXEC | Not discriminative, so made advisory | research/failed-experiments/2026-10-30-row-profile-qc-real-radiographs.md |
| 2026-10-30 | L8 | DenseNet embedding probe | EXEC | PA~mirror cosine 0.971 | results/.../xrv_densenet_embeddings.json |
| 2026-10-30 | L9 | `run_pipeline` dry run (CPU) | EXEC | Stopped safely at the synthesizer, with exact blockers recorded | results/.../run_pipeline_dryrun |
| 2026-10-30 | L10 | Unit test suite | TEST | 48/48 pass. One JSON-parsing bug found and fixed | tests/ |
| 2026-10-30 | L11 | MedGemma / MedSigLIP / Parallax / DVG on GPU | NOT RUN | No Colab session. Models gated | EXPERIMENT_RESULTS.md |
