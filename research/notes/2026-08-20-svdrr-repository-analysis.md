# SV-DRR Repository Analysis

## Scope and Evidence Type

This note records observed behavior from the locally cloned SV-DRR repository at commit `1d4d59b`; claims in its README about the paper, pretrained weights, and performance are repository documentation, not RadioVue validation.

## Relevance

- **Module:** Parallax.
- **Task:** One X-ray view plus a requested relative pose produces one or more synthesized novel X-ray views.
- **Status:** Candidate pretrained implementation; not yet run or validated on RadioVue data.

## Verified Repository Behavior

- `test_svdrr_DiT.py` loads a `SvdrrDiTPipeline` with float16, moves it to CUDA, enables xFormers attention, VAE tiling, and attention slicing, then runs 30 denoising steps with guidance scale 3.0.
- The pipeline encodes the input image twice: as a VAE latent concatenated with diffusion noise and as a frozen CLIP image embedding concatenated with a relative-pose embedding.
- The custom `SvdrrTransformer2DModel` is a PixArt-derived 28-layer latent-space DiT with eight input/output channels; the VAE and CLIP image encoder are frozen during training.
- The supplied inference script accepts one image through `--image_path`, square-resizes it to `--image_size`, converts it to grayscale, and can generate a sweep of azimuth poses from -90 to 90 degrees in 5-degree increments.
- Dataset inference is not compatible with RadioVue's current `Patient_###/PA.jpg` folders because it expects the repository's LIDC-IDRI DRR layout, including `patients.json`, `camera_views.json`, and `0000.png`.
- `scripts/download_models.py` defines Hugging Face repositories for 256, 512, and 1024 models and assembles each local pipeline with shared VAE, scheduler, and image-encoder directories plus model-specific transformer and projection directories.
- `train_svdrr_DiT.py` trains only against the custom LIDC-IDRI DRR layout produced from CT volumes; `scripts/create_drr_img.py` is the separate DiffDRR-based CT-to-DRR generator and is not required for pretrained inference.

## Data and Pose Assumptions

The README describes training on synthetic DRRs from LIDC-IDRI CTs, with a standard PA conditioning view and pose coordinates from a spherical camera-view file. The source code uses the relative condition/target pose convention and documents an optional pose flip because its coordinate convention is unintuitive.

## Limitations and Risks

- The current RadioVue inputs are real CR chest X-rays, while the repository training inputs are synthetic DRRs; the domain shift is untested.
- The inference script distorts non-square X-rays by resizing directly to a square, which may affect anatomy and pose synthesis.
- No direct DICOM `ViewPosition` was available in the prepared data, and the repository does not calibrate a real PA-to-lateral acquisition geometry.
- The inference script negates the shared pose array inside its per-image loop when `--flip_pose` is absent, so multi-image dataset inference can alternate pose signs; this does not affect a one-image invocation.
- The declared environment pins Python 3.11.9, PyTorch 2.5.1 with CUDA 12.4, Diffusers 0.30.3, Transformers 4.46.2, Accelerate 1.1.1, and xFormers 0.0.28.post3; Colab compatibility has not yet been tested.
- The large vendored `diffusion/` tree includes utilities with undeclared optional dependencies such as MMCV and timm, while `create_drr_img.py` additionally requires DiffDRR and matplotlib.
- Generated images are research outputs and not clinical ground truth.

## Recommended First Colab Experiment

Run a single, manually reviewed `PA.jpg` through the pretrained 256 or 512 pipeline with one explicitly chosen pose and a fixed random seed, saving outputs into that same workspace folder. Record the exact model revision, pose, preprocessing policy, seed, runtime, and visual/anatomical review outcome before sweeping poses or changing the repository.
