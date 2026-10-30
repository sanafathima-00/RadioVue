"""Explicit pipeline configuration.

Every value that changes scientific behaviour lives here so a run can be
reproduced from its saved ``config.json``.  Values marked ``UNCALIBRATED`` are
placeholders that must be confirmed by the geometry calibration in
``notebooks/03_Paired_DRR_Validation.ipynb`` before results are interpreted.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    # --- locations -------------------------------------------------------
    output_root: str = "outputs/radiovue_runs"
    volumea_root: str = "Volumea"
    parallax_root: str = "Parallax"
    dvg_checkpoint_dir: str = "Volumea/models/DVG-Diffusion"
    parallax_model_dir: str = "Parallax/models/DiT-fb-512"

    # --- models ----------------------------------------------------------
    medgemma_model_id: str = "google/medgemma-1.5-4b-it"
    medsiglip_model_id: str = "google/medsiglip-448"
    run_medgemma: bool = True
    run_segmentation: bool = True
    max_new_tokens: int = 1500

    # --- routing policy --------------------------------------------------
    # AP inputs are never relabelled PA.  They are processed only when this
    # flag is set, and every output is tagged as an experimental AP route.
    allow_ap_experimental: bool = True
    # Pretrained Parallax was trained with the 0-degree PA DRR as condition
    # only (Parallax/dataset.py "Sampling method 2"), so lateral -> frontal is
    # out of distribution.  Disabled by default; even when enabled, the result
    # is only passed to Volumea if ``allow_lateral_to_volumea`` is also set.
    allow_lateral_experimental: bool = False
    allow_lateral_to_volumea: bool = False
    # A user/DICOM view declaration outranks pixel classifiers for PA vs AP.
    declared_view: str | None = None

    # --- view synthesis (Parallax) --------------------------------------
    parallax_image_size: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 3.0
    synthesis_seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    # Azimuth sign of the Parallax lateral fed to Volumea slot 1.  -90 was selected
    # on a Colab T4 (2026-10-30): with a real PA in slot 0, the CT from the -90
    # lateral agreed with the CT from the subject's real lateral (NCC 0.55) far better
    # than +90 (0.25), and a mirrored real lateral gave 0.20.  This is single-subject
    # evidence without CT ground truth; paired CT validation is still required.
    lateral_azimuth_deg: float = -90.0
    lateral_azimuth_calibrated: bool = False
    # UNCALIBRATED: dihedral transform taking Parallax's raw output into display
    # orientation (head up, anterior on the viewer's left).  Fitted by notebook 03.
    parallax_output_transform: str = "identity"

    # --- Volumea (DVG) adapter -----------------------------------------
    # "repo_center_crop" reproduces Volumea/dataset/lidc.py SingleDataGenerator_real;
    # "contain_pad" reproduces the existing Volumea Colab cells.
    dvg_preprocess_mode: str = "repo_center_crop"
    # UNCALIBRATED: dihedral transform (calibration.DIHEDRAL name) mapping a
    # display-orientation radiograph into DVG projector space for each slot.
    # Fitted by notebook 03 as ``display_to_model``.
    dvg_frontal_transform: str = "identity"
    dvg_lateral_transform: str = "identity"
    dvg_geometry_calibrated: bool = False
    dvg_seed: int = 42
    run_volumea: bool = True
    # Drop each GPU model after its stage (needed on a 16 GB T4).
    free_gpu_between_stages: bool = True

    # --- QC gate (provisional thresholds, see VALIDATION_REPORT.md) ------
    qc_min_row_profile_corr: float = 0.80
    qc_max_mean_sample_std: float = 0.12
    qc_min_dynamic_range: float = 0.15
    # row_profile is advisory until calibrated on paired DRR data (see
    # research/failed-experiments/2026-10-30-row-profile-qc-real-radiographs.md).
    qc_gating_checks: list[str] = field(default_factory=lambda: ["sample_agreement", "dynamic_range", "not_a_copy"])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
