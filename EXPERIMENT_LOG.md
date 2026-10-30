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
| 2026-10-30 | L11 | Parallax PA→lateral on Colab T4, 3 CC0 subjects | EXEC | Runs, 6.3 s/img. Patient specificity at chance. Hallucination in subject B | results/colab_t4_2026-10-30 |
| 2026-10-30 | L12 | Parallax lateral→frontal | EXEC | Not frontal-like. Lateral input stays unsupported | same |
| 2026-10-30 | L13 | Volumea/DVG, real PA + {real, ±90 generated, mirrored real} lateral | EXEC | Runs, 27.5 s/vol. −90° agrees with the real-lateral CT (NCC 0.55). Orientation-critical | same |
| 2026-10-30 | L14 | MedGemma | NOT RUN | `HF_TOKEN` secret absent | — |
