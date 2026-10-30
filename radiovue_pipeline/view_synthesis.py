"""Complementary-view synthesis with Parallax (SV-DRR), wrapped for provenance and uncertainty.

Selected architecture (see VIEW_SYNTHESIS_RESEARCH.md): the pretrained SV-DRR
view-conditioned DiT is the generator.  Patient specificity comes from its
conditioning on the input image (VAE latent + CLIP embedding).  Anatomical
priors come from its training on CT-derived DRRs.  This wrapper adds:

* the exact pose convention of ``Parallax/infer.py`` (``pose = 0 - [0, az, 0]``);
* fixed seeds and N samples, so per-pixel disagreement is an explicit
  uncertainty map;
* a route policy (PA in-distribution, AP experimental, lateral out of
  distribution) recorded with every output.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROUTE_IN_DISTRIBUTION = "PA_TO_LATERAL (pose geometry in-distribution; real-radiograph input is out-of-domain vs DRR training)"
ROUTE_AP = "AP_TO_LATERAL (EXPERIMENTAL: model conditioned on PA DRRs only; AP magnification/positioning differ)"
ROUTE_FRONTAL_UNSPECIFIED = "FRONTAL_TO_LATERAL (PA/AP unverified; treated as frontal)"
ROUTE_LATERAL = "LATERAL_TO_FRONTAL (OUT-OF-DISTRIBUTION: pretrained model never conditioned on lateral views)"


@dataclass
class SynthesisPlan:
    input_view: str
    target_view: str
    azimuth_deg: float
    route: str
    allowed: bool
    in_distribution: bool
    reasons: list[str] = field(default_factory=list)


def plan_synthesis(input_view: str, cfg) -> SynthesisPlan:
    """Decide target view and pose from the identified input view (pure; unit-tested)."""
    reasons: list[str] = []
    az = float(cfg.lateral_azimuth_deg)
    if not cfg.lateral_azimuth_calibrated:
        reasons.append(
            f"Lateral azimuth {az:+.0f} deg was selected on single-subject evidence without CT ground truth; "
            "laterality is not yet validated against a ground-truth CT (run notebook 03 with paired data)."
        )
    if input_view == "PA":
        return SynthesisPlan("PA", "LATERAL", az, ROUTE_IN_DISTRIBUTION, True, True, reasons)
    if input_view == "FRONTAL_UNSPECIFIED":
        return SynthesisPlan("FRONTAL_UNSPECIFIED", "LATERAL", az, ROUTE_FRONTAL_UNSPECIFIED, True, False,
                             reasons + ["PA vs AP not verified; if the image is AP, results follow the experimental AP route."])
    if input_view == "AP":
        if not cfg.allow_ap_experimental:
            return SynthesisPlan("AP", "LATERAL", az, ROUTE_AP, False, False, reasons + ["AP route disabled by configuration."])
        return SynthesisPlan("AP", "LATERAL", az, ROUTE_AP, True, False, reasons + ["AP input kept as AP; never relabelled PA."])
    if input_view == "LATERAL":
        # Generated frontal would be a DRR-like frontal projection: neither a PA nor an AP acquisition.
        target = "FRONTAL_SYNTHETIC_DRR_LIKE"
        if not cfg.allow_lateral_experimental:
            return SynthesisPlan("LATERAL", target, -az, ROUTE_LATERAL, False, False,
                                 reasons + ["Lateral -> frontal synthesis is disabled: no validated generator exists."])
        return SynthesisPlan("LATERAL", target, -az, ROUTE_LATERAL, True, False,
                             reasons + ["Experimental out-of-distribution generation enabled by configuration."])
    return SynthesisPlan(input_view, "NONE", 0.0, "NONE", False, False, reasons + [f"Unsupported or unknown input view '{input_view}'."])


def parallax_pose(azimuth_deg: float) -> np.ndarray:
    """Pose array exactly as Parallax/infer.py builds it without --flip_pose."""
    return np.array([0.0, 0.0, 0.0]) - np.array([0.0, float(azimuth_deg), 0.0])


class ParallaxSynthesizer:
    """Loads ``SvdrrDiTPipeline`` from a local model directory (GPU, fp16)."""

    def __init__(self, parallax_root: str | Path, model_dir: str | Path, image_size: int = 512):
        import torch

        root = str(Path(parallax_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from pipeline import SvdrrDiTPipeline  # Parallax/pipeline.py

        if not torch.cuda.is_available():
            raise RuntimeError("Parallax synthesis requires a CUDA GPU runtime.")
        self.torch = torch
        self.image_size = image_size
        self.model_dir = str(model_dir)
        pipe = SvdrrDiTPipeline.from_pretrained(self.model_dir, torch_dtype=torch.float16)
        pipe.transformer.caption_projection = None
        try:
            pipe.enable_xformers_memory_efficient_attention()
            self.xformers = True
        except Exception:
            self.xformers = False
        pipe.enable_vae_tiling()
        pipe.enable_attention_slicing()
        self.pipe = pipe.to("cuda")

    def prepare_input(self, image: Image.Image) -> Image.Image:
        """Stock Parallax preprocessing: square resize then grayscale (infer.py:120-121)."""
        return image.convert("RGB").resize((self.image_size, self.image_size)).convert("L")

    def generate(self, image: Image.Image, azimuth_deg: float, seeds: list[int], steps: int = 30, guidance: float = 3.0) -> list[Image.Image]:
        torch = self.torch
        conditioned = self.prepare_input(image)
        outputs = []
        for seed in seeds:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))
            with torch.no_grad(), torch.autocast("cuda"):
                result = self.pipe(
                    input_imgs=conditioned,
                    prompt_imgs=conditioned,
                    poses=parallax_pose(azimuth_deg),
                    height=self.image_size,
                    width=self.image_size,
                    guidance_scale=guidance,
                    num_images_per_prompt=1,
                    num_inference_steps=steps,
                    generator=generator,
                )
            outputs.append(result.images[0].convert("L"))
        return outputs

    def describe(self) -> dict[str, Any]:
        return {"generator": "Parallax/SV-DRR SvdrrDiTPipeline", "model_dir": self.model_dir,
                "image_size": self.image_size, "xformers": self.xformers, "dtype": "float16"}
