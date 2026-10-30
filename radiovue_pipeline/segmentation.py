"""Frontal chest X-ray anatomical segmentation (TorchXRayVision PSPNet, ChestX-Det).

Provides the precise boundaries that a VLM cannot.  Validated by its authors
on frontal images only; this module refuses lateral input rather than produce
meaningless masks.  Measurements are in pixels of the input image.  No
millimetre values are reported, because pixel spacing of rendered PNG/JPEG is
unknown.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def mask_measurements(masks: dict[str, np.ndarray], threshold: float = 0.5) -> dict[str, Any]:
    """Pixel-space descriptors from soft masks (pure; unit-tested)."""
    out: dict[str, Any] = {}
    for name, mask in masks.items():
        binary = mask >= threshold
        area = int(binary.sum())
        if area == 0:
            out[name] = {"present": False}
            continue
        rows = np.where(binary.any(axis=1))[0]
        cols = np.where(binary.any(axis=0))[0]
        out[name] = {
            "present": True,
            "area_px": area,
            "bbox_xyxy_px": [int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1],
            "vertical_extent_px": int(rows[-1] - rows[0] + 1),
            "horizontal_extent_px": int(cols[-1] - cols[0] + 1),
        }
    lungs = [out.get("Left Lung", {}), out.get("Right Lung", {})]
    heart = out.get("Heart", {})
    if all(l.get("present") for l in lungs) and heart.get("present"):
        thorax_left = min(l["bbox_xyxy_px"][0] for l in lungs)
        thorax_right = max(l["bbox_xyxy_px"][2] for l in lungs)
        out["derived"] = {
            "cardiothoracic_width_ratio_px": heart["horizontal_extent_px"] / max(thorax_right - thorax_left, 1),
            "note": "Mask-width ratio on the given projection; AP projection magnifies the heart. Not a clinical CTR.",
            "lung_apex_row_px": min(l["bbox_xyxy_px"][1] for l in lungs),
            "lung_base_row_px": max(l["bbox_xyxy_px"][3] for l in lungs),
        }
    return out


class AnatomySegmenter:
    def __init__(self):
        import torch
        import torchxrayvision as xrv

        self.torch = torch
        self.xrv = xrv
        self.model = xrv.baseline_models.chestx_det.PSPNet().eval()

    def segment(self, image: Image.Image) -> dict[str, np.ndarray]:
        from .io_utils import pad_to_square

        w, h = image.size
        square, (left, top, _, _) = pad_to_square(image.convert("L"))  # model needs square input; no cropping
        side = square.size[0]
        array = self.xrv.datasets.normalize(np.asarray(square.resize((512, 512), Image.BILINEAR), dtype=np.float32), 255)
        tensor = self.torch.from_numpy(array)[None, None, ...]
        with self.torch.no_grad():
            logits = self.model(tensor)
        probs = self.torch.sigmoid(logits)  # PSPNet.forward returns raw logits
        probs = self.torch.nn.functional.interpolate(probs, size=(side, side), mode="bilinear", align_corners=False)
        probs = probs[:, :, top : top + h, left : left + w]
        return {name: probs[0, i].numpy() for i, name in enumerate(self.model.targets)}

    def run(self, image: Image.Image, is_frontal: bool, out_dir: Path | None = None) -> dict[str, Any]:
        if not is_frontal:
            return {"skipped": True, "reason": "ChestX-Det PSPNet is a frontal-view model; lateral input not segmented."}
        masks = self.segment(image)
        result: dict[str, Any] = {
            "model": "torchxrayvision chestx_det.PSPNet (ChestX-Det, 14 structures)",
            "measurements_px": mask_measurements(masks),
            "note": "Pixel-space masks from a dedicated segmentation model; not validated on this site's data.",
        }
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            stacked = np.stack([masks[k] for k in self.model.targets]).astype(np.float16)
            np.savez_compressed(out_dir / "segmentation_masks.npz", masks=stacked, targets=np.array(self.model.targets))
            result["masks_path"] = str(out_dir / "segmentation_masks.npz")
            overlay = overlay_masks(image, masks)
            overlay.save(out_dir / "segmentation_overlay.png")
            result["overlay_path"] = str(out_dir / "segmentation_overlay.png")
        return result


def overlay_masks(image: Image.Image, masks: dict[str, np.ndarray], threshold: float = 0.5) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    rng = np.random.default_rng(7)
    for mask in masks.values():
        colour = rng.uniform(60, 255, size=3)
        sel = mask >= threshold
        base[sel] = 0.6 * base[sel] + 0.4 * colour
    return Image.fromarray(base.astype(np.uint8))
