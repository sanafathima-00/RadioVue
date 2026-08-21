# Parallax source naming and pretrained-model layout

- **Decision:** Use concise, conventional filenames for the Parallax entry points while retaining the upstream model artifact directory and Python class names required by the pretrained pipeline.
- **Date:** 2026-08-20

## Context

The cloned SV-DRR implementation used source filenames such as `test_svdrr_DiT.py` and `pipeline_svdrr_DiT.py`. RadioVue needs clearer local entry points before moving execution to Google Colab. The official 512-pixel pretrained pipeline is published as a structured Diffusers-style repository.

## Options considered

1. Keep all upstream source filenames unchanged.
2. Rename source files and also rename model artifacts/classes.
3. Rename only user-facing source entry points, update imports and documentation, and preserve compatible model artifacts/classes.

## Chosen approach

Rename the source files to `infer.py`, `train.py`, `pipeline.py`, and `transformer.py`. Update every in-repository import and README command that referred to the old paths. Download the official `xiechun-tsukuba/svdrr-dit-fb-512` pipeline unchanged to `Parallax/models/DiT-fb-512/`.

## Why chosen

The new source names make the local workflow easier to navigate. The model artifacts and classes form the serialization/import contract used by the published pretrained pipeline, so retaining them avoids an unnecessary compatibility risk.

## Evidence and reasoning

The official repository's model downloader maps the 512 checkpoint to `models/DiT-fb-512`. The completed download contains the expected pipeline configuration and component weights for the transformer, VAE, image encoder, scheduler, and CC projection. Python syntax compilation passed after the import updates, and no obsolete source-name references remain.

## Trade-offs

The renamed source files diverge from upstream filenames, so future upstream patches may need manual path reconciliation. Keeping the model directory as `DiT-fb-512` is less stylistically uniform, but it communicates the official checkpoint identity and preserves compatibility.

## Reconsideration criteria

Reconsider this layout if an upstream update depends on the original source filenames, if a future RadioVue training workflow defines a distinct checkpoint registry, or if Colab integration shows a path-based assumption that cannot be updated safely.
