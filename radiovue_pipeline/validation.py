"""Paired-DRR validation of view synthesis and Volumea integration (GPU; notebook 03).

Ground truth comes from a CT.  For each case:

* ``V``: CT in DVG units (128^3, [0,1]), orientation chosen by
  ``search_dvg_volume_orientation`` (maximum coarse-stage self-consistency).
* Display references: NumPy DRRs with orientation known from DICOM
  (``calibration.reference_views``).
* DVG-domain views: Volumea's projector on ``V``.
* SV-DRR-domain views: DiffDRR with Parallax's training geometry.

Arms (only one variable changes between neighbouring arms):

==========  =====================================  ======================================
Arm         slot 0 (frontal)                       slot 1 (lateral)
==========  =====================================  ======================================
A_native    DVG projector frontal of V             DVG projector lateral of V
C_svdrr     SV-DRR PA DRR -> DVG space             SV-DRR true lateral DRR -> DVG space
B_synth     SV-DRR PA DRR -> DVG space             Parallax(SV-DRR PA) -> DVG space
B_ap        SV-DRR AP DRR -> DVG space             Parallax(SV-DRR AP) -> DVG space
C_ap        SV-DRR AP DRR -> DVG space             SV-DRR true lateral DRR -> DVG space
==========  =====================================  ======================================

A vs C_svdrr isolates the DRR style/field-of-view mismatch.  C_svdrr vs
B_synth isolates the synthesis error.  B_synth vs B_ap gives the PA-vs-AP
sensitivity.  Every arm is scored against ``V``.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import apply_transform, best_dihedral_match, reference_views
from .metrics import image_metrics, ncc, volume_metrics

PERMUTATIONS = list(itertools.permutations((0, 1, 2)))
FLIPS = list(itertools.product((False, True), repeat=3))


def orient_volume(volume: np.ndarray, perm: tuple[int, int, int], flips: tuple[bool, bool, bool]) -> np.ndarray:
    out = np.transpose(volume, perm)
    for axis, flip in enumerate(flips):
        if flip:
            out = np.flip(out, axis=axis)
    return np.ascontiguousarray(out)


def _resize2d(a: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    a = np.asarray(a, dtype=np.float32)
    a = (a - a.min()) / max(float(a.max() - a.min()), 1e-8)
    return np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR), dtype=np.float32)


def _scale_about_centre(a: np.ndarray, scale: float, size: int) -> np.ndarray:
    """Zoom ``a`` by ``scale`` about its centre and crop/pad back to ``size``."""
    from PIL import Image

    side = max(2, int(round(size * scale)))
    zoomed = np.asarray(Image.fromarray(a.astype(np.float32)).resize((side, side), Image.BILINEAR))
    out = np.zeros((size, size), dtype=np.float32)
    if side >= size:
        o = (side - size) // 2
        out[:] = zoomed[o : o + size, o : o + size]
    else:
        o = (size - side) // 2
        out[o : o + side, o : o + side] = zoomed
    return out


def fit_similarity(src: np.ndarray, dst: np.ndarray, size: int = 128, scales=None, max_shift: int = 12) -> dict[str, Any]:
    """Grid-search an isotropic scale and an integer shift that map ``src`` onto ``dst`` (max NCC)."""
    scales = scales if scales is not None else np.round(np.arange(0.6, 1.62, 0.04), 2)
    s, d = _resize2d(src, size), _resize2d(dst, size)
    best = {"ncc": -2.0}
    for scale in scales:
        z = _scale_about_centre(s, float(scale), size)
        for dy in range(-max_shift, max_shift + 1, 2):
            for dx in range(-max_shift, max_shift + 1, 2):
                r = ncc(np.roll(z, (dy, dx), axis=(0, 1)), d)
                if r > best["ncc"]:
                    best = {"ncc": r, "scale": float(scale), "dy": dy, "dx": dx}
    return best


def apply_similarity(src: np.ndarray, fit: dict[str, Any], size: int = 128) -> np.ndarray:
    z = _scale_about_centre(_resize2d(src, size), fit["scale"], size)
    return np.clip(np.roll(z, (fit["dy"], fit["dx"]), axis=(0, 1)), 0, 1)


# ---------------------------------------------------------------- DVG helpers (GPU)

def _ddpm(volumea_root: str | Path):
    root = str(Path(volumea_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import ddpm.diffusion as diffusion

    return diffusion


def coarse_reconstruct(model, volume01: np.ndarray, volumea_root: str | Path) -> np.ndarray:
    """DVG coarse stage on views projected from ``volume01`` (mirrors ``FineStage.sample`` lines 142-162)."""
    import torch

    diffusion = _ddpm(volumea_root)
    ct = torch.from_numpy(volume01.astype(np.float32))[None, None].cuda().contiguous()
    t = torch.tensor([model.pose_timesteps - 1], device="cuda").long()
    with torch.no_grad():
        bp = diffusion.add_bp(ct, 2)
        latent = model.vqgan_bp.encode(bp, quantize=False, include_embeddings=True)
        coarse = model.vqgan.decode(model.fit_model(latent, t), quantize=True)
    return coarse[0, 0].float().cpu().numpy()


def search_dvg_volume_orientation(model, volume01: np.ndarray, volumea_root: str | Path, candidates=None) -> list[dict[str, Any]]:
    """Rank axis orientations of ``volume01`` by coarse-stage self-consistency NCC.

    The correct orientation is in the training distribution, so the coarse
    stage should reproduce it best.  Treat a small margin as inconclusive.
    """
    candidates = candidates or [(p, f) for p in PERMUTATIONS for f in FLIPS]
    ranked = []
    for perm, flips in candidates:
        oriented = orient_volume(volume01, perm, flips)
        coarse = coarse_reconstruct(model, oriented, volumea_root)
        ranked.append({"perm": perm, "flips": flips, "coarse_ncc": ncc(coarse, oriented)})
    return sorted(ranked, key=lambda r: r["coarse_ncc"], reverse=True)


def dvg_reconstruct_views(reconstructor, frontal: np.ndarray, lateral: np.ndarray, seed: int = 42) -> np.ndarray:
    """Run DVG on two views that are already in DVG projector space (128x128, [0,1])."""
    from .provenance import ImageRecord
    from .volumea_adapter import DvgInputPair

    stub = ImageRecord("in-memory", "VALIDATION", "N/A", "", 128, 128)
    pair = DvgInputPair(_resize2d(frontal, 128), _resize2d(lateral, 128), stub, stub, {"mode": "validation_direct"})
    return reconstructor.reconstruct(pair, seed=seed)


# ---------------------------------------------------------------- per-case driver

def run_case(
    case_id: str,
    hu_zyx_iso: np.ndarray,
    gt_volume_dicom_oriented: np.ndarray,
    orientation: dict[str, Any],
    svdrr_views: dict[str, np.ndarray],
    synth_from_pa: list[np.ndarray],
    synth_from_ap: list[np.ndarray],
    reconstructor,
    volumea_root: str | Path,
    lpips_fn=None,
    seed: int = 42,
) -> dict[str, Any]:
    """Score every arm for one case.

    ``hu_zyx_iso``: the HU cube (DICOM orientation) that produced the GT volume.
    ``svdrr_views``: display-oriented SV-DRR images ``{"pa", "ap", "lateral"}``.
    ``synth_*``: display-oriented Parallax samples (seed 0 first).
    """
    from .drr import dvg_projector_views

    V = orient_volume(gt_volume_dicom_oriented, tuple(orientation["perm"]), tuple(orientation["flips"]))
    refs = reference_views(hu_zyx_iso)
    dvg_views = dvg_projector_views(V, volumea_root)
    cal_front = best_dihedral_match(dvg_views["frontal"], refs["frontal"])
    cal_lat = best_dihedral_match(dvg_views["lateral"], refs["lateral"])
    to_dvg_front = lambda img: apply_transform(_resize2d(img, 128), cal_front["display_to_model"])
    to_dvg_lat = lambda img: apply_transform(_resize2d(img, 128), cal_lat["display_to_model"])

    # Field-of-view mapping fitted on ground-truth DRRs only, then reused unchanged for synthetic views.
    fov_front = fit_similarity(to_dvg_front(svdrr_views["pa"]), dvg_views["frontal"])
    fov_lat = fit_similarity(to_dvg_lat(svdrr_views["lateral"]), dvg_views["lateral"])
    map_front = lambda img: apply_similarity(to_dvg_front(img), fov_front)
    map_lat = lambda img: apply_similarity(to_dvg_lat(img), fov_lat)

    arms = {
        "A_native": (dvg_views["frontal"], dvg_views["lateral"]),
        "C_svdrr": (map_front(svdrr_views["pa"]), map_lat(svdrr_views["lateral"])),
        "B_synth": (map_front(svdrr_views["pa"]), map_lat(synth_from_pa[0])),
        "B_ap": (map_front(svdrr_views["ap"]), map_lat(synth_from_ap[0])),
        "C_ap": (map_front(svdrr_views["ap"]), map_lat(svdrr_views["lateral"])),
    }
    gt_lat_512 = _resize2d(svdrr_views["lateral"], 512)
    out: dict[str, Any] = {
        "case_id": case_id,
        "calibration": {"dvg_frontal": cal_front, "dvg_lateral": cal_lat, "fov_frontal": fov_front, "fov_lateral": fov_lat},
        "synthesis_2d": {
            "from_pa": image_metrics(_resize2d(synth_from_pa[0], 512), gt_lat_512, lpips_fn),
            "from_ap": image_metrics(_resize2d(synth_from_ap[0], 512), gt_lat_512, lpips_fn),
            "seed_spread_pa": float(np.mean(np.std(np.stack([_resize2d(s, 512) for s in synth_from_pa]), axis=0))),
        },
        "arms": {},
        "volumes": {},
    }
    for name, (front, lat) in arms.items():
        volume = dvg_reconstruct_views(reconstructor, front, lat, seed=seed)
        out["arms"][name] = volume_metrics(volume, V)
        out["volumes"][name] = volume
    out["volumes"]["ground_truth"] = V
    return out
