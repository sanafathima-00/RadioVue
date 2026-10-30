"""Quality-control gate for a synthetic complementary view.

The gate decides whether a synthetic view may be passed to Volumea.  It uses
checks that need no ground truth:

1. **Row-integral consistency (physics).**  For parallel-beam line-integral
   projections that rotate about the cranio-caudal axis, the sum across each
   detector row equals the total attenuation of that axial slab.  That sum is
   identical for every rotation angle.  The vertical profile of a frontal view
   and a lateral view of the same patient should therefore agree, up to an
   affine intensity mapping.  Real radiographs are cone-beam, scatter-affected
   and display-processed, so here this is a **heuristic** consistency
   check, not an equality.
2. **Sample agreement (uncertainty).**  Several seeds are generated.  The mean
   per-pixel standard deviation measures how under-determined the synthesis is.
3. **Not-a-copy check.**  A "lateral" that still correlates strongly with the
   frontal input pixel-for-pixel was not rotated.
4. **Dynamic range.**  Rejects blank or saturated outputs.

The thresholds in ``PipelineConfig`` are provisional engineering defaults.
They have not been validated against reconstruction quality (see VALIDATION_REPORT.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import ncc


def row_profile(image01: np.ndarray, body_mask_threshold: float | None = None) -> np.ndarray:
    """Mean attenuation-like value per row (rows = cranio-caudal axis)."""
    img = np.asarray(image01, dtype=np.float64)
    if body_mask_threshold is not None:
        img = np.where(img > body_mask_threshold, img, 0.0)
    return img.mean(axis=1)


def _zscore(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-12)


def row_profile_consistency(
    frontal01: np.ndarray,
    lateral01: np.ndarray,
    max_shift_frac: float = 0.05,
    scales: tuple[float, ...] = (1.0,),
) -> dict[str, float]:
    """Best Pearson correlation of z-scored row profiles over vertical shifts (and optional scales).

    ``scales`` stretches the lateral profile vertically.  A synthetic view
    generated from the input shares its detector frame, so the default (1.0)
    is used in the QC gate.  A wider set is only for comparing independently
    acquired images that have different fields of view.
    """
    a, b0 = row_profile(frontal01), row_profile(lateral01)
    n = len(a)
    if len(b0) != n:
        b0 = np.interp(np.linspace(0, len(b0) - 1, n), np.arange(len(b0)), b0)
    a = _zscore(a)
    max_shift = max(1, int(round(max_shift_frac * n)))
    best, best_shift, best_scale = -1.0, 0, 1.0
    centre = (n - 1) / 2
    for scale in scales:
        # sample lateral row r at the position it would have if the image were scaled about the centre
        b = _zscore(np.interp(centre + (np.arange(n) - centre) / scale, np.arange(n), b0))
        for shift in range(-max_shift, max_shift + 1):
            if shift >= 0:
                x, y = a[shift:], b[: n - shift]
            else:
                x, y = a[: n + shift], b[-shift:]
            if x.std() == 0 or y.std() == 0:
                continue  # a constant profile carries no consistency information
            r = float(np.corrcoef(x, y)[0, 1])
            if r > best:
                best, best_shift, best_scale = r, shift, scale
    return {"row_profile_corr": best, "row_shift_px": float(best_shift), "row_scale": float(best_scale)}


def sample_statistics(samples: list[np.ndarray]) -> dict[str, Any]:
    stack = np.stack([np.asarray(s, dtype=np.float64) for s in samples])
    std_map = stack.std(axis=0)
    return {
        "n_samples": len(samples),
        "mean_pixel_std": float(std_map.mean()),
        "p95_pixel_std": float(np.percentile(std_map, 95)),
        "mean_image": stack.mean(axis=0),
        "std_map": std_map,
    }


DEFAULT_GATING_CHECKS = ("sample_agreement", "dynamic_range", "not_a_copy")


def evaluate_synthetic_view(
    input01: np.ndarray,
    samples01: list[np.ndarray],
    min_row_profile_corr: float,
    max_mean_sample_std: float,
    min_dynamic_range: float,
    max_input_copy_ncc: float = 0.9,
    expect_rotation: bool = True,
    gating_checks: tuple[str, ...] = DEFAULT_GATING_CHECKS,
) -> dict[str, Any]:
    """Run all QC checks.  ``passed`` is True only if every *gating* check passes.

    ``row_profile`` is advisory by default.  On a real, independently
    acquired PA/lateral pair (CC0, 2026-10-30) it did not separate the true
    lateral from a vertically flipped one.  It is only physically justified
    when both views share a detector frame.  Make it gating only after
    calibrating it on paired DRR data (notebook 03).
    """
    stats = sample_statistics(samples01)
    mean_img = stats.pop("mean_image")
    std_map = stats.pop("std_map")
    representative = samples01[0]
    checks: dict[str, Any] = {}

    rp = row_profile_consistency(input01, representative)
    checks["row_profile"] = {**rp, "threshold": min_row_profile_corr, "passed": rp["row_profile_corr"] >= min_row_profile_corr}
    checks["sample_agreement"] = {**stats, "threshold": max_mean_sample_std, "passed": stats["mean_pixel_std"] <= max_mean_sample_std}
    dyn = float(np.percentile(representative, 99) - np.percentile(representative, 1))
    checks["dynamic_range"] = {"p99_minus_p1": dyn, "threshold": min_dynamic_range, "passed": dyn >= min_dynamic_range}
    if expect_rotation and np.shape(input01) == np.shape(representative):
        copy_score = ncc(input01, representative)
        checks["not_a_copy"] = {"input_output_ncc": copy_score, "threshold": max_input_copy_ncc, "passed": copy_score < max_input_copy_ncc}

    for name, check in checks.items():
        check["gating"] = name in gating_checks
    passed = all(c["passed"] for c in checks.values() if c["gating"])
    return {
        "passed": passed,
        "checks": checks,
        "failed_checks": [k for k, c in checks.items() if not c["passed"] and c["gating"]],
        "advisory_failures": [k for k, c in checks.items() if not c["passed"] and not c["gating"]],
        "thresholds_validated": False,
        "_mean_image": mean_img,
        "_std_map": std_map,
    }
