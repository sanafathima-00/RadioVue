"""Adapter between the view-synthesis output and Volumea (DVG-Diffusion).

Volumea is used unchanged.  The adapter's job:

* Order the views as DVG requires: ``xray[0]`` = frontal (projector pose
  -0.5*pi), ``xray[1]`` = lateral (pose 0).  Swapping them changes the
  volume substantially (Colab Experiment 1B).
* Reproduce one of the two documented preprocessing paths and record which
  one was used.
* Apply only the orientation transforms that ``PipelineConfig`` declares.
  They are uncalibrated until notebook 03 is run.
* Refuse to run when a synthetic view failed QC or came from a disabled route.
  The adapter never substitutes an arbitrary image.
"""

from __future__ import annotations

import ctypes
import json
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .provenance import ORIGIN_SYNTHETIC, ImageRecord, stamp_synthetic, utc_now

DVG_SIZE = 128
CHECKPOINTS = {
    "ct_vqgan": "CT_VQGAN.ckpt",
    "view_vqgan_0_90": "VPGE_0_90_VQGAN.ckpt",
    "view_vqgan_45": "VPGE_45_VQGAN.ckpt",
    "coarse_model": "NewView_Coarse_Model.pt",
    "dvg_diffusion": "DVG_Diffusion.pt",
}


class RefusedInput(RuntimeError):
    """Raised when the adapter will not pass an image pair to Volumea."""


# ---------------------------------------------------------------------------
# Preprocessing (pure; unit-tested)
# ---------------------------------------------------------------------------

def _repo_center_crop(pil_image: Image.Image, crop_size=(160, 160)) -> Image.Image:
    """Verbatim logic of Volumea/dataset/lidc.py ``center_crop``."""
    while pil_image.size[0] >= 2 * crop_size[0] and pil_image.size[1] >= 2 * crop_size[1]:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = max(crop_size[0] / pil_image.size[0], crop_size[1] / pil_image.size[1])
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    left = (pil_image.size[0] - crop_size[0]) // 2
    upper = (pil_image.size[1] - crop_size[1]) // 2
    return pil_image.crop((left, upper, left + crop_size[0], upper + crop_size[1]))


def _repo_center_crop2(image: Image.Image, crop_size=(128, 128)) -> Image.Image:
    """Verbatim logic of Volumea/dataset/lidc.py ``center_crop2``."""
    width, height = image.size
    left = (width - crop_size[0]) // 2
    top = (height - crop_size[1]) // 2
    return image.crop((left, top, left + crop_size[0], top + crop_size[1]))


def preprocess_for_dvg(image: Image.Image, mode: str, transform: str = "identity") -> np.ndarray:
    """Return a (128,128) float32 array in [0,1] following a documented DVG path.

    ``transform`` is a ``calibration.DIHEDRAL`` name mapping display orientation to DVG projector space.
    """
    gray = image.convert("L")
    if mode == "repo_center_crop":
        out = _repo_center_crop2(_repo_center_crop(gray, (160, 160)), (DVG_SIZE, DVG_SIZE))
        out = out.resize((DVG_SIZE, DVG_SIZE), Image.BILINEAR)  # torchvision Resize((128,128)); identity at 128
    elif mode == "contain_pad":
        contained = ImageOps.contain(gray, (DVG_SIZE, DVG_SIZE), method=Image.Resampling.LANCZOS)
        out = Image.new("L", (DVG_SIZE, DVG_SIZE), color=0)
        out.paste(contained, ((DVG_SIZE - contained.width) // 2, (DVG_SIZE - contained.height) // 2))
    else:
        raise ValueError(f"Unknown DVG preprocessing mode '{mode}'.")
    from .calibration import apply_transform

    return apply_transform(np.asarray(out, dtype=np.float32) / 255.0, transform).astype(np.float32)


@dataclass
class DvgInputPair:
    frontal: np.ndarray
    lateral: np.ndarray
    frontal_record: ImageRecord
    lateral_record: ImageRecord
    preprocessing: dict[str, Any]


def build_input_pair(
    frontal_image: Image.Image,
    lateral_image: Image.Image,
    frontal_record: ImageRecord,
    lateral_record: ImageRecord,
    cfg,
    synthesis_qc: dict[str, Any] | None,
    synthesis_route_allowed_for_volumea: bool,
) -> DvgInputPair:
    """Validate provenance/QC, then preprocess in DVG's required order."""
    synthetic = [r for r in (frontal_record, lateral_record) if r.origin == ORIGIN_SYNTHETIC]
    if len(synthetic) > 1:
        raise RefusedInput("Both views are synthetic; Volumea requires at least one acquired view in this pipeline.")
    if synthetic:
        if synthesis_qc is None or not synthesis_qc.get("passed", False):
            failed = (synthesis_qc or {}).get("failed_checks", ["qc_missing"])
            raise RefusedInput(f"Synthetic view failed QC ({', '.join(failed)}); not passing it to Volumea.")
        if not synthesis_route_allowed_for_volumea:
            raise RefusedInput("The synthesis route that produced this view is not permitted to feed Volumea.")
    if frontal_record.view not in {"PA", "AP", "FRONTAL_UNSPECIFIED", "FRONTAL_SYNTHETIC_DRR_LIKE"}:
        raise RefusedInput(f"Slot 0 must be a frontal view, got {frontal_record.view}.")
    if lateral_record.view != "LATERAL":
        raise RefusedInput(f"Slot 1 must be a lateral view, got {lateral_record.view}.")

    mode = cfg.dvg_preprocess_mode
    frontal = preprocess_for_dvg(frontal_image, mode, cfg.dvg_frontal_transform)
    lateral = preprocess_for_dvg(lateral_image, mode, cfg.dvg_lateral_transform)
    return DvgInputPair(
        frontal, lateral, frontal_record, lateral_record,
        {
            "mode": mode,
            "slot0": "frontal -> projector pose -0.5*pi",
            "slot1": "lateral -> projector pose 0",
            "frontal_transform": cfg.dvg_frontal_transform,
            "lateral_transform": cfg.dvg_lateral_transform,
            "geometry_calibrated": cfg.dvg_geometry_calibrated,
            "internal_dvg_normalisation": "per-view min-max inside xray2bp_from_real",
        },
    )


# ---------------------------------------------------------------------------
# DVG model (GPU; lazy imports; mirrors the working Volumea Colab cells)
# ---------------------------------------------------------------------------

@contextmanager
def _trusted_checkpoint_loading():
    """Load the official DVG checkpoints under PyTorch >= 2.6.

    They pickle omegaconf objects, which ``weights_only=True`` (the new default,
    also passed by Lightning) rejects.  The patch is active only while the model is
    built, and only for these published files.
    """
    import torch

    original = torch.load

    def load(*args, **kwargs):
        if kwargs.get("weights_only") in (None, True):
            kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = load
    try:
        yield
    finally:
        torch.load = original


def _stub_unused_video_dependency() -> None:
    """``vq_gan_3d/utils.py`` imports scikit-video for video I/O that inference never uses.

    scikit-video fails to import on NumPy 2, so it is stubbed in ``sys.modules``
    instead of editing Volumea.
    """
    try:
        import skvideo.io  # noqa: F401
    except Exception:
        module = types.ModuleType("skvideo")
        module.io = types.ModuleType("skvideo.io")
        sys.modules["skvideo"], sys.modules["skvideo.io"] = module, module.io


def _preload_torch_libs() -> None:
    import torch

    lib = Path(torch.__file__).parent / "lib"
    for name in ("libc10.so", "libtorch_cpu.so", "libtorch_cuda.so", "libtorch_python.so"):
        path = lib / name
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


class VolumeaReconstructor:
    def __init__(self, volumea_root: str | Path, checkpoint_dir: str | Path):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Volumea requires a CUDA GPU runtime.")
        self.torch = torch
        self.root = Path(volumea_root).resolve()
        self.checkpoint_dir = Path(checkpoint_dir)
        missing = [f for f in CHECKPOINTS.values() if not (self.checkpoint_dir / f).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing DVG checkpoints in {self.checkpoint_dir}: {missing}")
        _preload_torch_libs()
        _stub_unused_video_dependency()
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        from ddpm import FineStage
        from ddpm.new_unet3d import Unet3D

        ck = {k: str(self.checkpoint_dir / v) for k, v in CHECKPOINTS.items()}
        with _trusted_checkpoint_loading():
            model = FineStage(
                fit_model=Unet3D(dim=64, dim_mults=(1, 2, 4, 8), channels=8, out_dim=8).cuda(),
                fine_model=Unet3D(dim=128, dim_mults=(1, 2, 4, 8), channels=24, out_dim=8).cuda(),
                vqgan_ckpt=ck["ct_vqgan"],
                vqgan_ckpt_bp=ck["view_vqgan_0_90"],
                vqgan_ckpt_bp025=ck["view_vqgan_45"],
                image_size=32, input_img_size=128, num_frames=32, channels=8,
                timesteps=100, pose_timesteps=1, loss_type="l2",
            ).cuda()
        state = torch.load(ck["dvg_diffusion"], map_location="cpu", weights_only=False)
        model.load_state_dict(state["ema"], strict=True)
        del state
        coarse = torch.load(ck["coarse_model"], map_location="cpu", weights_only=False)["model"]
        full = model.state_dict()
        updates = {k: v for k, v in coarse.items() if k in full and tuple(v.shape) == tuple(full[k].shape)}
        if not updates:
            raise RuntimeError("No coarse-checkpoint keys matched the DVG architecture.")
        full.update(updates)
        model.load_state_dict(full, strict=True)
        model.eval()
        self.model = model
        self.coarse_keys_loaded = len(updates)

    def reconstruct(self, pair: DvgInputPair, seed: int = 42) -> np.ndarray:
        torch = self.torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        views = [torch.from_numpy(v)[None, None].to("cuda", dtype=torch.float32) for v in (pair.frontal, pair.lateral)]
        placeholder = torch.zeros((1, 1, 128, 128, 128), device="cuda")  # only used for an unused diagnostic reprojection
        with torch.inference_mode():
            ct, _, _ = self.model.sample_from_xray(ct=placeholder, xray=views, batch_size=1)
        volume = ct[0, 0].detach().float().cpu().numpy()
        if volume.shape != (128, 128, 128) or not np.isfinite(volume).all():
            raise RuntimeError("DVG produced an invalid volume.")
        return volume


def save_reconstruction(volume: np.ndarray, pair: DvgInputPair, out_dir: Path, extra: dict[str, Any]) -> dict[str, Any]:
    """Save raw volume + orthogonal previews + provenance.  Values are normalised, not HU."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npy = out_dir / "SYNTHETIC_ct_volume_dvg_normalised.npy"
    np.save(npy, volume.astype(np.float32))
    paths = {"npy": str(npy)}
    try:
        import SimpleITK as sitk

        nii = out_dir / "SYNTHETIC_ct_volume_dvg_normalised.nii.gz"
        sitk.WriteImage(sitk.GetImageFromArray(volume.astype(np.float32)), str(nii))
        paths["nifti"] = str(nii)
    except ImportError:
        pass
    c = 64
    previews = {"axis0": volume[c], "axis1": volume[:, c, :], "axis2": volume[:, :, c]}
    for name, sl in previews.items():
        img = Image.fromarray(np.clip(np.rint(sl * 255), 0, 255).astype(np.uint8)).resize((384, 384), Image.NEAREST)
        path = out_dir / f"SYNTHETIC_ct_{name}_slice64_display.png"
        stamp_synthetic(img, "DVG output | normalised, not HU").save(path)
        paths[f"preview_{name}"] = str(path)
    for slot, arr in (("slot0_frontal", pair.frontal), ("slot1_lateral", pair.lateral)):
        Image.fromarray(np.clip(np.rint(arr * 255), 0, 255).astype(np.uint8)).save(out_dir / f"dvg_input_{slot}.png")
    provenance = {
        "created_utc": utc_now(),
        "output_kind": "SYNTHETIC research CT volume (DVG-Diffusion); not HU, unit spacing, not clinical",
        "shape": list(volume.shape),
        "value_range": [float(volume.min()), float(volume.max())],
        "inputs": {"slot0": pair.frontal_record.to_dict(), "slot1": pair.lateral_record.to_dict()},
        "preprocessing": pair.preprocessing,
        **extra,
    }
    (out_dir / "ct_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    paths["provenance"] = str(out_dir / "ct_provenance.json")
    return paths
