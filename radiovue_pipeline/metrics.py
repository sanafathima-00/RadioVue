"""2D and 3D comparison metrics.

Image-similarity metrics are reported alongside anatomy-aware ones, because
SSIM/PSNR alone do not establish anatomical correctness.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a


def mae(a, b) -> float:
    return float(np.mean(np.abs(_as01(a) - _as01(b))))


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((_as01(a) - _as01(b)) ** 2)))


def psnr(a, b, data_range: float = 1.0) -> float:
    err = np.mean((_as01(a) - _as01(b)) ** 2)
    return float("inf") if err == 0 else float(10 * np.log10(data_range**2 / err))


def ssim(a, b, data_range: float = 1.0) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(_as01(a), _as01(b), data_range=data_range))


def ncc(a, b) -> float:
    a, b = _as01(a).ravel(), _as01(b).ravel()
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    return float(np.mean(a * b))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else float(2 * np.logical_and(a, b).sum() / denom)


class LPIPSMetric:
    """LPIPS (AlexNet) for grayscale [0,1] images; optional dependency ``lpips``."""

    def __init__(self, net: str = "alex"):
        import lpips
        import torch

        self.torch = torch
        self.model = lpips.LPIPS(net=net, verbose=False).eval()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        t = lambda x: self.torch.from_numpy(np.repeat(np.asarray(x, np.float32)[None, None] * 2 - 1, 3, axis=1))
        with self.torch.no_grad():
            return float(self.model(t(a), t(b)).item())


def image_metrics(pred: np.ndarray, target: np.ndarray, lpips_fn=None) -> dict[str, float]:
    out = {"mae": mae(pred, target), "psnr": psnr(pred, target), "ssim": ssim(pred, target), "ncc": ncc(pred, target)}
    if lpips_fn is not None:
        out["lpips"] = lpips_fn(pred, target)
    return out


def lung_mask_dvg(volume01: np.ndarray, low: float = 0.04, high: float = 0.25) -> np.ndarray:
    """Low-density (lung-like) voxels in DVG [0,1] units (~ -924..-399 HU under the HU+1024 assumption).

    DVG output zeroes values below 0.04, so the lower bound excludes outside air.
    """
    return (volume01 > low) & (volume01 < high)


def bone_mask_dvg(volume01: np.ndarray, low: float = 0.6) -> np.ndarray:
    return volume01 >= low


def volume_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    from skimage.metrics import structural_similarity

    pred, target = _as01(pred), _as01(target)
    return {
        "mae": mae(pred, target),
        "rmse": rmse(pred, target),
        "psnr": psnr(pred, target),
        "ssim_3d": float(structural_similarity(pred, target, data_range=1.0)),
        "ncc": ncc(pred, target),
        "dice_lung_like": dice(lung_mask_dvg(pred), lung_mask_dvg(target)),
        "dice_bone_like": dice(bone_mask_dvg(pred), bone_mask_dvg(target)),
        "lung_like_fraction_pred": float(lung_mask_dvg(pred).mean()),
        "lung_like_fraction_target": float(lung_mask_dvg(target).mean()),
        "note": "Values in DVG-normalised units; not HU. Threshold masks are coarse tissue-class proxies, not segmentations.",
    }


def patient_specificity(pred_views: list[np.ndarray], gt_views: list[np.ndarray], metric=ncc) -> dict[str, Any]:
    """Is each synthetic view closer to its own patient's ground truth than to other patients'?

    Returns the similarity matrix and the top-1 identification rate.  A
    generator that outputs a generic lateral scores near chance (1/N).
    """
    n = len(pred_views)
    matrix = np.array([[metric(pred_views[i], gt_views[j]) for j in range(n)] for i in range(n)])
    top1 = float(np.mean(matrix.argmax(axis=1) == np.arange(n))) if n else float("nan")
    return {"similarity_matrix": matrix.tolist(), "top1_identification": top1, "chance": 1.0 / n if n else None}
