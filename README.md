# RadioVue

**Experimental whole-chest reconstruction from limited X-ray views.**

RadioVue investigates whether complementary X-ray synthesis, multi-view CT reconstruction, and projection consistency can form a useful research pipeline. Its intended anatomical scope includes lungs, ribs, heart, and surrounding soft tissues.

The repository brings together research implementations under three conceptual modules. Their presence does not establish a validated end-to-end system or diagnostic accuracy.

## Research pipeline

```text
Chest X-ray → Parallax: complementary synthetic view
    → Voluma: two-view 3D reconstruction
    → Recon: CT-to-X-ray projection and consistency evaluation
    → Compare with reference data and reconsider the model
```

The conceptual module is **Voluma**; its repository directory is **Volumea/**.

## Repository modules

| Module | Implementation and status |
| --- | --- |
| [Parallax](Parallax/README.md) | SV-DRR view-conditioned diffusion research code. `infer.py` accepts an image or dataset, model/output directories, image size, and pose selection. Uses CUDA, half precision, and xformers. |
| [Volumea](Volumea/README.md) | DVG-Diffusion: CT/conditioning VQ-GANs, coarse intermediate-view prediction, fine diffusion, LIDC preprocessing, and DRR utilities. Checkpoints must be supplied separately. |
| Recon | Workspace for projection consistency and reconstruction comparisons. Local X-Recon code is not included in the currently published module snapshot. |

The modules have different upstream environments. There is no shared installer, unified CLI, deployed application, or validated automatic chain between them.

## Interpretation limits

- A complementary generated view is **synthetic**, not an acquired lateral radiograph. Interchangeability requires a validated experiment.
- Digitally rotating a frontal image does not create a lateral acquisition.
- DVG's released training/validation path derives conditioning views from reference CT with its projector. Real radiographs introduce geometry, intensity, preprocessing, and domain differences.
- The DVG configuration uses **128 × 128 inputs**, and LIDC preprocessing resizes CT to **128³**. Larger exported images or interpolated slices do not establish higher native reconstruction resolution.
- Visually plausible anatomy does not establish patient-specific accuracy, calibrated Hounsfield units, or clinical suitability.

This is research software. Generated volumes and projections require appropriate reference data and quantitative evaluation before stronger claims are made.

## Getting started

```powershell
git clone https://github.com/sanafathima-00/RadioVue.git
Set-Location RadioVue
```

Start with one module and its documentation. Use separate environments and review upstream dependency requirements. Verify CUDA/model compatibility in the chosen environment.

### Parallax

Read [the module README](Parallax/README.md) and `Parallax/environment.yaml`. Obtain the appropriate pretrained SV-DRR components separately. `--model_path` must point to the extracted directory with configuration and required components, not an archive or an empty parent directory.

After preparing a compatible CUDA environment and model directory:

```powershell
Set-Location Parallax
python infer.py --model_path models/DiT-fb-512 --image_path example.png --log_dir outputs --image_size 512 --simple_pose
```

This generates a series of pose-conditioned views. It is a CLI example, not a fresh-machine reproducibility guarantee. Pose conventions are dataset-specific; inspect them before assigning anatomical labels.

### Voluma / DVG-Diffusion

Read [the module README](Volumea/README.md), `Volumea/config/model/ddpm.yaml`, and the dataset configuration. The checkpoint set is:

- `CT_VQGAN.ckpt`
- `VPGE_0_90_VQGAN.ckpt`
- `VPGE_45_VQGAN.ckpt`
- `NewView_Coarse_Model.pt`
- `DVG_Diffusion.pt`

Configure checkpoint paths, prepared LIDC data, and the DRR projector before using `Volumea/train/val_ddpm.py`. The projector has CUDA build requirements. Checkpoint loading alone does not validate real-X-ray conditioning.

## Evaluation approach

1. Reproduce the upstream synthetic-conditioning baseline with the correct split and checkpoints.
2. Validate view ordering, geometry, normalization, and checkpoint/EMA selection.
3. Compare reconstructed CT and reprojected views with reference CT and acquired radiographs.
4. Evaluate acquired frontal/lateral pairs separately from frontal-plus-synthetic-view inputs.
5. Record model versions, poses, preprocessing, quantitative metrics, failures, and limitations.

Keep checkpoint and geometry comparisons to one changed variable. Interpolation and display improvements are not reconstruction-accuracy evidence.

## Local workspace and data

The local workspace also contains research notes, reports, conversion utilities, a Colab notebook, data, and downloaded artifacts. These are not all published. In particular, `scripts/`, `research/`, `reports/`, and `openspec/` are ignored by the root configuration.

The local DICOM converter creates separate display working copies; raw inputs must remain untouched. JPEG/PNG exports are not replacements for DICOM data. Do not commit patient identifiers, medical datasets, private reports, generated patient imagery, or weights.

## Upstream research and attribution

- [SV-DRR source](https://github.com/xiechun-tsukuba/svdrr) and [paper](https://arxiv.org/abs/2507.05148): research basis for Parallax.
- [DVG-Diffusion source](https://github.com/xiexing0916/DVG-Diffusion) and [checkpoints](https://huggingface.co/xing0916/DVG-Diffusion): implementation under `Volumea/`.
- [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/): reference research data used by upstream approaches.

RadioVue integration work is distinct from upstream models and published results. Retain attribution and consult component licences and dataset terms. No project-wide licence is asserted here.
