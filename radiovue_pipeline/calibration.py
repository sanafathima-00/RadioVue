"""Geometry calibration by matching projections to orientation-known references.

The reference is a NumPy parallel-beam DRR of a DICOM CT whose array axes
have known patient directions.  For axial slices with
ImageOrientationPatient = [1,0,0, 0,1,0], the axes are: column -> patient
left, row -> posterior, and (after ``load_dicom_ct_series``) slice ->
inferior.  Therefore:

* ``reference_frontal``  = sum over rows (y): image (z, x).  Head is at the
  top and the patient's left is on the viewer's right.  This is standard
  PA display orientation.
* ``reference_lateral``  = sum over columns (x): image (z, y).  Head is at
  the top and anterior is on the viewer's left.

A model's projection or generated view is compared against all 8 dihedral
transforms (identity, flips, 90-degree rotations).  The best match by NCC
gives the transform that maps model space to display space.  The margin over
the runner-up shows how decisive the result is.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

DIHEDRAL: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda a: a,
    "flip_lr": lambda a: a[:, ::-1],
    "flip_ud": lambda a: a[::-1, :],
    "rot180": lambda a: a[::-1, ::-1],
    "transpose": lambda a: a.T,
    "rot90_ccw": lambda a: np.rot90(a, 1),
    "rot90_cw": lambda a: np.rot90(a, -1),
    "anti_transpose": lambda a: np.rot90(a, 2).T,
}

INVERSE = {
    "identity": "identity", "flip_lr": "flip_lr", "flip_ud": "flip_ud", "rot180": "rot180",
    "transpose": "transpose", "rot90_ccw": "rot90_cw", "rot90_cw": "rot90_ccw", "anti_transpose": "anti_transpose",
}


def apply_transform(image: np.ndarray, name: str) -> np.ndarray:
    return np.ascontiguousarray(DIHEDRAL[name](np.asarray(image)))


def _resize(a: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    a = np.asarray(a, dtype=np.float32)
    a = (a - a.min()) / max(float(a.max() - a.min()), 1e-8)
    return np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR), dtype=np.float64)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    return float(np.mean(a * b))


def best_dihedral_match(model_image: np.ndarray, reference: np.ndarray, size: int = 128) -> dict[str, Any]:
    """Transform T such that T(model_image) best matches ``reference`` (display orientation)."""
    ref = _resize(reference, size)
    scores = {name: _ncc(_resize(apply_transform(model_image, name), size), ref) for name in DIHEDRAL}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "model_to_display": ranked[0][0],
        "display_to_model": INVERSE[ranked[0][0]],
        "ncc": ranked[0][1],
        "margin_over_runner_up": ranked[0][1] - ranked[1][1],
        "scores": scores,
    }


def reference_views(hu_zyx: np.ndarray) -> dict[str, np.ndarray]:
    from .drr import hu_to_mu

    mu = hu_to_mu(hu_zyx)
    return {"frontal": mu.sum(axis=1), "lateral": mu.sum(axis=2)}
