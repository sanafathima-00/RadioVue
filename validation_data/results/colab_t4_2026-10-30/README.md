# Colab T4 GPU results — 2026-10-30

Executed by Claude in the Colab notebook https://colab.research.google.com/drive/13n088v32FElJHQIG1IUjF7Mtg7OdEGhA
(Tesla T4 15 GB, driver 580.82, Python 3.13, diffusers 0.30.3, transformers 4.46.2; code from the public `main`
branch of this repository plus in-notebook glue identical in behaviour to `radiovue_pipeline`).

Inputs: three CC0 subjects with real same-patient PA + lateral radiographs (Wikimedia Commons; checksums in
`validation_data/public_cc0/`): subjA (2010 pair, "Stillwaterising"), subjB (21-year-old, normal, M. Häggström),
subjC (88-year-old, influenza + H. influenzae, M. Häggström). No CT exists for these subjects.

| File | Experiment |
|---|---|
| `parallax_pa_to_lateral_patient_specificity.json` | Parallax DiT-fb-512 PA→lateral, az ±90, seeds 0-3; generated vs real laterals (3×3, alignment-searched NCC) |
| `parallax_lateral_to_frontal.json` | Parallax lateral→frontal (out of distribution), az ±90 |
| `dvg_subjA_arm_comparison.json` | Volumea/DVG with real PA in slot 0 and four slot-1 laterals; compared with the real-lateral CT |
