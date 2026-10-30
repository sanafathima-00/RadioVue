"""Acquisition-view identification by explicit evidence fusion.

Evidence hierarchy (highest first):

1. DICOM ``ViewPosition`` / ``ViewCodeSequence`` (authoritative when present).
2. An operator declaration (fills missing metadata only; never overrides it).
3. Pixel classifier for FRONTAL vs LATERAL (TorchXRayVision ``xinario.ViewModel``).
4. MedGemma's PA/AP/LATERAL answer (uncalibrated; used as weak evidence only).

PA vs AP cannot be reliably separated from pixels by any validated model in
this project.  Without metadata or a declaration, a frontal image is reported
as ``FRONTAL_UNSPECIFIED`` with low confidence, and AP is never relabelled as PA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from PIL import Image

VIEWS = ("PA", "AP", "LATERAL", "FRONTAL_UNSPECIFIED", "UNKNOWN")


@dataclass
class ViewDecision:
    view: str
    family: str  # FRONTAL | LATERAL | UNKNOWN
    confidence: str  # high | medium | low
    sources: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _family(view: str | None) -> str | None:
    if view in {"PA", "AP", "FRONTAL_UNSPECIFIED"}:
        return "FRONTAL"
    if view == "LATERAL":
        return "LATERAL"
    return None


def fuse_view_evidence(
    metadata_view: str | None = None,
    declared_view: str | None = None,
    classifier: dict[str, float] | None = None,
    vlm_view: str | None = None,
    classifier_threshold: float = 0.8,
) -> ViewDecision:
    """Combine view evidence.  Pure function; see module docstring for precedence.

    ``classifier`` is a probability dict such as ``{"Frontal": 0.97, "Lateral": 0.03}``.
    """
    sources = {
        "dicom_metadata": metadata_view,
        "declared": declared_view,
        "pixel_classifier": classifier,
        "medgemma": vlm_view,
    }
    warnings: list[str] = []
    if metadata_view == "OBLIQUE" or declared_view == "OBLIQUE":
        return ViewDecision("UNKNOWN", "UNKNOWN", "high", sources, ["Oblique projection is unsupported."])

    pixel_family = None
    if classifier:
        label, prob = max(classifier.items(), key=lambda kv: kv[1])
        if prob >= classifier_threshold:
            pixel_family = "LATERAL" if label.lower().startswith("lat") else "FRONTAL"
        else:
            warnings.append(f"Pixel view classifier is indecisive (max p={prob:.2f}).")

    # 1-2: authoritative metadata, then declaration (declaration may not conflict).
    if metadata_view and declared_view and metadata_view != declared_view:
        return ViewDecision(
            "UNKNOWN", "UNKNOWN", "high", sources,
            [f"Declared view {declared_view} conflicts with DICOM metadata {metadata_view}; refusing to guess."],
        )
    stated = metadata_view or declared_view
    if stated:
        if pixel_family and _family(stated) != pixel_family:
            return ViewDecision(
                "UNKNOWN", "UNKNOWN", "low", sources,
                [f"{'DICOM' if metadata_view else 'Declared'} view {stated} conflicts with pixel classifier ({pixel_family})."],
            )
        if vlm_view in {"PA", "AP", "LATERAL"} and vlm_view != stated:
            warnings.append(f"MedGemma suggested {vlm_view}; {stated} retained from {'DICOM' if metadata_view else 'declaration'}.")
        confidence = "high" if metadata_view else "medium"
        return ViewDecision(stated, _family(stated) or "UNKNOWN", confidence, sources, warnings)

    # 3-4: pixels only.
    if pixel_family == "LATERAL":
        if vlm_view in {"PA", "AP"}:
            warnings.append("MedGemma disagrees with the lateral pixel classification.")
            return ViewDecision("LATERAL", "LATERAL", "low", sources, warnings)
        return ViewDecision("LATERAL", "LATERAL", "medium", sources, warnings)
    if pixel_family == "FRONTAL":
        warnings.append(
            "PA vs AP is not verifiable from pixels in this pipeline; supply DICOM or a declared view. "
            "Treating as frontal with unspecified projection."
        )
        if vlm_view in {"PA", "AP"}:
            warnings.append(f"MedGemma (uncalibrated) suggests {vlm_view}; recorded as weak evidence only.")
        if vlm_view == "LATERAL":
            warnings.append("MedGemma disagrees with the frontal pixel classification.")
            return ViewDecision("UNKNOWN", "UNKNOWN", "low", sources, warnings)
        return ViewDecision("FRONTAL_UNSPECIFIED", "FRONTAL", "low", sources, warnings)
    if vlm_view in {"PA", "AP", "LATERAL"}:
        warnings.append("Only the uncalibrated VLM answer is available.")
        return ViewDecision("UNKNOWN", _family(vlm_view) or "UNKNOWN", "low", sources, warnings)
    return ViewDecision("UNKNOWN", "UNKNOWN", "low", sources, warnings + ["No usable view evidence."])


class PixelViewClassifier:
    """TorchXRayVision frontal/lateral classifier (xinario/chestViewSplit ResNet-50, Apache-2.0)."""

    def __init__(self):
        import torch
        import torchxrayvision as xrv

        self.torch = torch
        self.xrv = xrv
        self.model = xrv.baseline_models.xinario.ViewModel().eval()

    def predict(self, image: Image.Image) -> dict[str, float]:
        from .io_utils import pad_to_square

        square, _ = pad_to_square(image.convert("L"))  # the model requires square input; padding keeps all anatomy
        array = np.asarray(square.resize((224, 224), Image.BILINEAR), dtype=np.float32)
        array = self.xrv.datasets.normalize(array, 255)  # -> [-1024, 1024]
        tensor = self.torch.from_numpy(array)[None, None, ...]
        with self.torch.no_grad():
            logits = self.model(tensor)
            probs = self.torch.softmax(logits, dim=1)[0].tolist()
        return dict(zip(self.model.targets, probs))
