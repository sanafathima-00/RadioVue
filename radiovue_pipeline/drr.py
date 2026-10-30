"""CT loading and digitally reconstructed radiographs (DRRs) for paired validation data.

Three renderers, used for different purposes:

* ``parallel_drr``: NumPy parallel-beam line integrals.  CPU-only; used for
  unit tests and for the projection-invariance QC metric.
* ``SvdrrDrrRenderer``: DiffDRR with the exact camera geometry that Parallax
  (SV-DRR) was trained on (``Parallax/scripts/create_drr_img.py``).  GPU.
* ``dvg_projector_views``: Volumea's own XraySyn CUDA projector at DVG's
  frontal (-0.5*pi) and lateral (0) poses.  GPU; reproduces the conditioning
  domain of the released DVG checkpoints.

``ct_to_dvg_volume`` reproduces the documented DVG/X2CT normalisation.
Resampling choices that are not documented upstream are marked ASSUMPTION.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# CT loading (DICOM series -> HU volume, z/y/x order, no identifiers kept)
# ---------------------------------------------------------------------------

@dataclass
class CTVolume:
    hu: np.ndarray  # (z, y, x), z increasing superior -> inferior after sort
    spacing_zyx: tuple[float, float, float]
    metadata: dict[str, Any]


def load_dicom_ct_series(series_dir: str | Path, min_slices: int = 64) -> CTVolume:
    """Load the thinnest axial CT series in a directory tree.

    Localizers and non-axial images are excluded.  Only technical metadata
    (slice thickness, pixel spacing, kVp, kernel) is retained.
    """
    import pydicom

    groups: dict[tuple, list] = {}
    for path in Path(series_dir).rglob("*"):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, force=True)
        except Exception:
            continue
        if str(ds.get("Modality")) != "CT" or "PixelData" not in ds:
            continue
        if "LOCALIZER" in [str(v).upper() for v in ds.get("ImageType", [])]:
            continue
        iop = [round(float(v), 3) for v in ds.get("ImageOrientationPatient", [])]
        if iop[:6] != [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]:
            continue
        key = (str(ds.get("SeriesInstanceUID")), float(ds.get("SliceThickness", 0) or 0))
        groups.setdefault(key, []).append(ds)
    candidates = [(k, v) for k, v in groups.items() if len(v) >= min_slices]
    if not candidates:
        raise ValueError("No axial CT series with enough slices was found.")
    (_, thickness), slices = min(candidates, key=lambda kv: (kv[0][1] or 99, -len(kv[1])))
    slices.sort(key=lambda d: float(d.ImagePositionPatient[2]), reverse=True)  # superior first
    hu = np.stack([d.pixel_array.astype(np.float32) * float(d.RescaleSlope) + float(d.RescaleIntercept) for d in slices])
    z_positions = np.array([float(d.ImagePositionPatient[2]) for d in slices])
    dz = float(np.median(np.abs(np.diff(z_positions)))) if len(z_positions) > 1 else float(thickness)
    dy, dx = (float(v) for v in slices[0].PixelSpacing)
    meta = {
        "n_slices": len(slices),
        "slice_thickness_mm": thickness,
        "slice_spacing_mm": dz,
        "pixel_spacing_mm": [dy, dx],
        "kvp": str(slices[0].get("KVP", "")),
        "kernel": str(slices[0].get("ConvolutionKernel", "")),
    }
    return CTVolume(hu, (dz, dy, dx), meta)


def resample_isotropic(volume: np.ndarray, spacing_zyx: tuple[float, float, float], target_mm: float) -> np.ndarray:
    from scipy.ndimage import zoom

    factors = [s / target_mm for s in spacing_zyx]
    return zoom(volume, factors, order=1)


def center_crop_or_pad(volume: np.ndarray, size: int, fill: float) -> np.ndarray:
    out = np.full((size, size, size), fill, dtype=volume.dtype)
    src, dst = [], []
    for dim in volume.shape:
        if dim >= size:
            start = (dim - size) // 2
            src.append(slice(start, start + size))
            dst.append(slice(0, size))
        else:
            start = (size - dim) // 2
            src.append(slice(0, dim))
            dst.append(slice(start, start + dim))
    out[tuple(dst)] = volume[tuple(src)]
    return out


def ct_to_dvg_volume(ct: CTVolume, size: int = 128, voxel_mm: float = 2.5) -> np.ndarray:
    """HU volume -> DVG-normalised cube in [0,1].

    Normalisation follows Volumea/dataset/lidc.py (``Limit_Min_Max_Threshold(0,
    2500)`` then ``Normalization(0, 2500)``), assuming the X2CT-GAN HDF5
    convention of HU + 1024 (ASSUMPTION: not stated in the DVG code).
    ASSUMPTION: 2.5 mm isotropic resampling with a centred 128^3 (320 mm)
    crop approximates the X2CT-GAN preprocessing; verify the field of view
    against DVG's projector before treating CT metrics as comparable.
    """
    iso = resample_isotropic(ct.hu, ct.spacing_zyx, voxel_mm)
    cube = center_crop_or_pad(iso, size, fill=-1024.0)
    return np.clip(cube + 1024.0, 0.0, 2500.0) / 2500.0


# ---------------------------------------------------------------------------
# CPU parallel-beam DRR (tests and QC reasoning)
# ---------------------------------------------------------------------------

def hu_to_mu(hu: np.ndarray, mu_water: float = 0.02) -> np.ndarray:
    """Linear attenuation per mm from HU (mono-energetic approximation)."""
    return np.clip(mu_water * (1.0 + hu / 1000.0), 0.0, None)


def parallel_drr(mu_zyx: np.ndarray, azimuth_deg: float, spacing_mm: float = 1.0) -> np.ndarray:
    """Parallel-beam line integrals for rotation about the cranio-caudal (z) axis.

    Returns a (z, u) line-integral image.  0 deg integrates along y
    (antero-posterior: a frontal view); 90 deg integrates along x (lateral).
    """
    from scipy.ndimage import rotate

    rotated = rotate(mu_zyx, angle=azimuth_deg, axes=(1, 2), reshape=True, order=1, mode="constant", cval=0.0)
    return rotated.sum(axis=1) * spacing_mm


def line_integral_to_display(line_integral: np.ndarray) -> np.ndarray:
    """Min-max display mapping used by SV-DRR (plt.imsave of the raw DRR)."""
    lo, hi = float(line_integral.min()), float(line_integral.max())
    return (line_integral - lo) / max(hi - lo, 1e-8)


# ---------------------------------------------------------------------------
# GPU renderers (lazy imports)
# ---------------------------------------------------------------------------

class SvdrrDrrRenderer:
    """DiffDRR renderer replicating Parallax/scripts/create_drr_img.py geometry.

    SV-DRR camera: ``read(nii, orientation="PA")``, sdd=1800 mm, 512 px at
    0.7 mm, translation (0, -1650, 0), ZXY Euler (azimuth, polar, 0).
    Azimuth 0 = the standard PA view (``0000.png``).
    """

    def __init__(self, nifti_path: str | Path, image_size: int = 512, orientation: str = "PA", sdd: float = 1800.0,
                 source_to_isocenter: float = 1650.0):
        """Defaults reproduce SV-DRR training.  For an AP-like bedside geometry use
        ``orientation="AP", sdd=1000, source_to_isocenter=850`` (ASSUMPTION: typical portable
        SID ~100 cm; supine physiology such as fluid redistribution is not modelled)."""
        import torch
        from diffdrr.data import read
        from diffdrr.drr import DRR

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        subject = read(str(nifti_path), orientation=orientation)
        delx = 0.7 * 512 / image_size
        self.source_to_isocenter = source_to_isocenter
        self.drr = DRR(subject, sdd=sdd, height=image_size, delx=delx, renderer="trilinear").to(self.device)

    def render(self, azimuth_deg: float, polar_deg: float = 0.0) -> np.ndarray:
        torch = self.torch
        rotations = torch.tensor([[np.deg2rad(azimuth_deg), np.deg2rad(polar_deg), 0.0]], device=self.device, dtype=torch.float32)
        translations = torch.tensor([[0.0, -self.source_to_isocenter, 0.0]], device=self.device)
        with torch.no_grad():
            img = self.drr(rotations, translations, parameterization="euler_angles", convention="ZXY")
        return line_integral_to_display(img[0, 0].detach().cpu().numpy())


def write_nifti(volume_zyx_hu: np.ndarray, spacing_zyx: tuple[float, float, float], path: str | Path) -> None:
    """Write a HU volume for DiffDRR.  Uses an LPS-style identity direction (ASSUMPTION; see calibration)."""
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(volume_zyx_hu[::-1].astype(np.int16))  # SimpleITK z increases inferior->superior
    image.SetSpacing((spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]))
    sitk.WriteImage(image, str(path))


def dvg_projector_views(dvg_volume: np.ndarray, volumea_root: str | Path) -> dict[str, np.ndarray]:
    """Render DVG-domain frontal (pose -0.5) and lateral (pose 0) views with Volumea's projector.

    Mirrors ``FineStage.q_sample_pose`` / ``xray2bp``: forward projection of a
    (1,1,128,128,128) tensor followed by per-view min-max normalisation.
    """
    import sys

    import torch

    root = str(Path(volumea_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from ddpm.diffusion import fp, get_T  # noqa: E402  (module-level CUDA projector)

    tensor = torch.from_numpy(dvg_volume.astype(np.float32))[None, None].cuda().contiguous()
    views = {}
    with torch.no_grad():
        for name, pose in (("frontal", -0.5), ("lateral", 0.0)):
            proj = fp(tensor, get_T([pose, 0, 0, 0, 0, 0]))
            proj = (proj - proj.min()) / (proj.max() - proj.min())
            views[name] = proj[0, 0].detach().cpu().numpy()
    return views
