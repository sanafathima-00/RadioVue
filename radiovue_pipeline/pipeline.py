"""``run_pipeline(input_xray)``: one chest X-ray -> complementary synthetic view -> Volumea CT.

Model components are injected through ``Components``, so tests can run the
whole control flow with fakes on a CPU.  ``Components.load(cfg)`` builds the
real GPU components and records any that fail instead of hiding the failure.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .anatomy_extraction import GenerationRecord, run_extraction
from .config import PipelineConfig
from .consistency import evaluate_synthetic_view, self_reconstruction_fidelity
from .io_utils import from_float01, load_xray, to_float01
from .provenance import ORIGIN_ORIGINAL, ImageRecord, preserve_original, save_synthetic, sha256_file, stamp_synthetic
from .view_classification import fuse_view_evidence
from .view_synthesis import plan_synthesis
from .volumea_adapter import RefusedInput, build_input_pair, save_reconstruction


class Lazy:
    """Deferred component construction, so large GPU models load only when their stage runs."""

    def __init__(self, name: str, factory: Callable[[], Any]):
        self.name, self.factory = name, factory


@dataclass
class Components:
    view_classifier: Any = None  # .predict(PIL) -> {"Frontal": p, "Lateral": p}
    segmenter: Any = None  # .run(PIL, is_frontal, out_dir) -> dict
    medgemma_generate: Callable[[Image.Image, str], GenerationRecord] | None = None  # in-process MedGemma
    medgemma_extract: Any = None  # .extract(PIL, is_frontal) -> dict  (isolated subprocess MedGemma)
    medgemma_revision: str | None = None
    synthesizer: Any = None  # .generate(PIL, azimuth, seeds, steps, guidance) -> [PIL]; .describe()
    reconstructor: Any = None  # .reconstruct(pair, seed) -> ndarray
    load_errors: dict[str, str] = field(default_factory=dict)
    _lazy: dict[str, Lazy] = field(default_factory=dict, repr=False)

    def resolve(self, attr: str) -> Any:
        """Return the component, constructing a ``Lazy`` one on first use and recording failures."""
        value = getattr(self, attr)
        if isinstance(value, Lazy):
            lazy = self._lazy[attr] = value
            try:
                value = lazy.factory()
                self.load_errors.pop(lazy.name, None)
            except Exception as error:  # recorded and surfaced as a warning, never silently ignored
                self.load_errors[lazy.name] = f"{type(error).__name__}: {error}"
                value = None
            setattr(self, attr, value)
        return value

    def release(self, attr: str) -> None:
        """Free a GPU model; a lazily built one is rebuilt on its next use."""
        setattr(self, attr, self._lazy.get(attr))
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @classmethod
    def load(
        cls,
        cfg: PipelineConfig,
        medgemma: bool = True,
        synthesis: bool = True,
        volumea: bool = True,
        medgemma_python: str | None = None,
        repo_root: str | Path = ".",
    ) -> "Components":
        """Real components.  Small CPU models load now; GPU models are ``Lazy``.

        ``medgemma_python``: interpreter of an isolated venv with ``transformers>=4.50``.
        When given, MedGemma runs in a subprocess; otherwise it runs in-process.
        """
        from .segmentation import AnatomySegmenter
        from .view_classification import PixelViewClassifier

        comps = cls()
        comps.view_classifier = Lazy("view_classifier", PixelViewClassifier)
        comps.resolve("view_classifier")
        if cfg.run_segmentation:
            comps.segmenter = Lazy("segmenter", AnatomySegmenter)
            comps.resolve("segmenter")
        if medgemma and cfg.run_medgemma:
            if medgemma_python:
                from .medgemma_worker import SubprocessMedGemma

                comps.medgemma_extract = SubprocessMedGemma(medgemma_python, repo_root, cfg.medgemma_model_id)
            else:
                from .anatomy_extraction import MedGemmaClient

                client = Lazy("medgemma", lambda: MedGemmaClient(cfg.medgemma_model_id, max_new_tokens=cfg.max_new_tokens))
                comps.medgemma_generate = client  # resolved in run_pipeline
        if synthesis:
            from .view_synthesis import ParallaxSynthesizer

            comps.synthesizer = Lazy(
                "parallax", lambda: ParallaxSynthesizer(cfg.parallax_root, cfg.parallax_model_dir, cfg.parallax_image_size)
            )
        if volumea and cfg.run_volumea:
            from .volumea_adapter import VolumeaReconstructor

            comps.reconstructor = Lazy("volumea", lambda: VolumeaReconstructor(cfg.volumea_root, cfg.dvg_checkpoint_dir))
        return comps


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return f"<array shape={value.shape}>"
    return value


def run_pipeline(
    input_xray: str | Path,
    config: PipelineConfig | None = None,
    components: Components | None = None,
    declared_view: str | None = None,
) -> dict[str, Any]:
    """Run the single-X-ray research pipeline.  Returns a JSON-serialisable result dict.

    The result always contains every documented key.  When a stage cannot run
    or a check fails, the corresponding entry is ``None`` and the reason is in
    ``warnings``.  Volumea is never given an image that failed QC.
    """
    cfg = config or PipelineConfig()
    comps = components or Components()
    declared = declared_view or cfg.declared_view
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = Path(cfg.output_root) / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg.save(run_dir / "config.json")

    warnings: list[str] = [f"Component unavailable: {k} ({v})" for k, v in comps.load_errors.items()]
    result: dict[str, Any] = {
        "status": "started",
        "run_dir": str(run_dir),
        "input_view": None,
        "view_decision": None,
        "anatomical_extraction": None,
        "original_image": None,
        "generated_view": None,
        "generated_view_type": None,
        "view_synthesis_metadata": None,
        "ct_reconstruction": None,
        "validation_results": {},
        "warnings": warnings,
        "disclaimer": (
            "Research output only. Generated views and CT volumes are synthetic, patient-conditioned estimates, "
            "not acquisitions of this patient and not clinical ground truth."
        ),
    }

    def finish(status: str) -> dict[str, Any]:
        result["status"] = status
        clean = _jsonable(result)
        (run_dir / "result.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
        return clean

    # 1. Load and preserve the original -------------------------------------
    try:
        loaded = load_xray(input_xray)
    except Exception as error:
        warnings.append(f"Input rejected: {error}")
        return finish("rejected_input")
    original_dir = run_dir / "original"
    preserved = preserve_original(input_xray, original_dir, view="PENDING")
    rendered_path = original_dir / "ORIGINAL_rendered_grayscale.png"
    loaded.image.save(rendered_path)
    result["original_image"] = {
        "preserved_file": preserved.path,
        "sha256": preserved.sha256,
        "rendered_grayscale": str(rendered_path),
        "rendered_sha256": sha256_file(rendered_path),
        "kind": loaded.kind,
        "size": list(loaded.image.size),
        "transformations": loaded.transformations,
        "technical_metadata": loaded.metadata,
        "origin": ORIGIN_ORIGINAL,
    }
    if loaded.metadata.get("BurnedInAnnotation") == "YES":
        warnings.append("DICOM reports burned-in annotation: pixels may contain identifiers.")

    # 2. View identification --------------------------------------------------
    classifier_probs = None
    view_classifier = comps.resolve("view_classifier")
    if view_classifier is not None:
        try:
            classifier_probs = view_classifier.predict(loaded.image)
        except Exception as error:
            warnings.append(f"Pixel view classifier failed: {error}")
    pre_family = None
    if classifier_probs:
        pre_family = "LATERAL" if classifier_probs.get("Lateral", 0) > classifier_probs.get("Frontal", 0) else "FRONTAL"
    if loaded.metadata_view in {"PA", "AP"} or (declared or "").upper() in {"PA", "AP"}:
        pre_family = "FRONTAL"
    elif loaded.metadata_view == "LATERAL" or (declared or "").upper() == "LATERAL":
        pre_family = "LATERAL"

    # 3. Anatomical extraction (MedGemma) + segmentation ---------------------
    extraction = None
    is_frontal_guess = pre_family != "LATERAL"
    if comps.medgemma_extract is not None:
        try:
            extraction = comps.medgemma_extract.extract(loaded.image, is_frontal_guess)
        except Exception as error:
            warnings.append(f"MedGemma extraction failed: {error}")
    elif comps.resolve("medgemma_generate") is not None:
        generate = comps.medgemma_generate
        generate = getattr(generate, "generate", generate)  # MedGemmaClient instance or plain callable
        try:
            extraction = run_extraction(
                loaded.image, generate, is_frontal=is_frontal_guess,
                model_revision=getattr(comps.medgemma_generate, "revision", comps.medgemma_revision),
            )
        except Exception as error:
            warnings.append(f"MedGemma extraction failed: {error}")
        if cfg.free_gpu_between_stages:
            comps.release("medgemma_generate")
    else:
        warnings.append("MedGemma not loaded: anatomical extraction skipped. " + comps.load_errors.get("medgemma", ""))
    vlm_view = extraction["image_metadata"]["view"] if extraction else None

    decision = fuse_view_evidence(
        metadata_view=loaded.metadata_view,
        declared_view=(declared or "").upper() or None,
        classifier=classifier_probs,
        vlm_view=vlm_view,
    )
    warnings.extend(decision.warnings)
    result["input_view"] = decision.view
    result["view_decision"] = decision.to_dict()

    segmenter = comps.resolve("segmenter")
    if segmenter is not None and decision.family == "FRONTAL":
        try:
            seg = segmenter.run(loaded.image, True, run_dir / "segmentation")
            if extraction is not None:
                extraction["segmentation"] = seg
            else:
                extraction = {"segmentation": seg}
        except Exception as error:
            warnings.append(f"Segmentation failed: {error}")
    if extraction is not None:
        (run_dir / "anatomical_extraction.json").write_text(json.dumps(_jsonable(extraction), indent=2), encoding="utf-8")
    result["anatomical_extraction"] = extraction
    original_record = ImageRecord(
        str(rendered_path), ORIGIN_ORIGINAL, decision.view, result["original_image"]["rendered_sha256"], *loaded.image.size
    )

    # 4. View synthesis -------------------------------------------------------
    plan = plan_synthesis(decision.view, cfg)
    warnings.extend(plan.reasons)
    result["generated_view_type"] = plan.target_view if plan.allowed else None
    result["view_synthesis_metadata"] = {
        "route": plan.route, "allowed": plan.allowed, "in_distribution": plan.in_distribution,
        "azimuth_deg": plan.azimuth_deg, "azimuth_calibrated": cfg.lateral_azimuth_calibrated,
    }
    if not plan.allowed:
        warnings.append("No complementary view generated; Volumea not run.")
        return finish("stopped_no_supported_synthesis")
    synthesizer = comps.resolve("synthesizer")
    if synthesizer is None:
        warnings.append(
            "Parallax synthesizer unavailable; no complementary view generated; Volumea not run. "
            + comps.load_errors.get("parallax", "")
        )
        return finish("stopped_synthesizer_unavailable")

    try:
        # Pose-0 self-reconstruction: an out-of-domain-input check that needs no ground truth.
        reconstruction = synthesizer.generate(loaded.image, 0.0, cfg.synthesis_seeds[:1], cfg.num_inference_steps, cfg.guidance_scale)[0]
        self_recon = self_reconstruction_fidelity(
            to_float01(loaded.image.convert("L").resize(reconstruction.size)), to_float01(reconstruction)
        )
        conditioning = reconstruction if cfg.bridge_to_drr_domain else loaded.image
        samples = synthesizer.generate(
            conditioning, plan.azimuth_deg, cfg.synthesis_seeds, cfg.num_inference_steps, cfg.guidance_scale
        )
    except Exception as error:
        warnings.append(f"View synthesis failed: {error}")
        result["validation_results"]["synthesis_traceback"] = traceback.format_exc(limit=3)
        return finish("failed_synthesis")
    if cfg.parallax_output_transform != "identity":
        from .calibration import apply_transform

        samples = [from_float01(apply_transform(to_float01(s), cfg.parallax_output_transform)) for s in samples]
    synth_description = {**synthesizer.describe(), "output_transform_to_display": cfg.parallax_output_transform}
    if cfg.free_gpu_between_stages:
        del synthesizer
        comps.release("synthesizer")

    size = samples[0].size
    input01 = to_float01(loaded.image.convert("L").resize(size))
    samples01 = [to_float01(s) for s in samples]
    qc = evaluate_synthetic_view(
        input01, samples01, cfg.qc_min_row_profile_corr, cfg.qc_max_mean_sample_std, cfg.qc_min_dynamic_range,
        gating_checks=tuple(cfg.qc_gating_checks), self_reconstruction=self_recon,
        min_self_reconstruction_edge_ncc=cfg.qc_min_self_reconstruction_edge_ncc,
    )
    if not qc["checks"]["self_reconstruction"]["passed"]:
        warnings.append(
            "Input appears to be outside the view-synthesis model's domain (it could not re-render the input at pose 0); "
            "the generated view is likely unreliable."
        )
    if qc["advisory_failures"]:
        warnings.append(f"Advisory (non-gating, uncalibrated) QC checks failed: {', '.join(qc['advisory_failures'])}.")
    synth_dir = run_dir / "synthetic"
    synth_meta = {
        **synth_description, "route": plan.route, "bridged_to_drr_domain": cfg.bridge_to_drr_domain, "azimuth_deg": plan.azimuth_deg,
        "seeds": cfg.synthesis_seeds, "steps": cfg.num_inference_steps, "guidance": cfg.guidance_scale,
        "conditioning_input_sha256": result["original_image"]["rendered_sha256"],
    }
    records = [
        save_synthetic(s, synth_dir, f"{plan.target_view.lower()}_seed{seed}", plan.target_view, "Parallax/SV-DRR", {**synth_meta, "seed": seed})
        for s, seed in zip(samples, cfg.synthesis_seeds)
    ]
    std_map = qc["_std_map"]
    std_img = from_float01(np.clip(std_map / max(float(std_map.max()), 1e-6), 0, 1))
    stamp_synthetic(std_img, "per-pixel std across seeds (uncertainty)").save(synth_dir / "SYNTHETIC_uncertainty_std_display.png")
    np.save(synth_dir / "uncertainty_std_map.npy", std_map.astype(np.float32))
    generated = records[0]  # an actual sample (seed 0); the mean image is not a radiograph-like sample
    result["generated_view"] = generated.to_dict()
    result["view_synthesis_metadata"].update({
        **synth_meta,
        "all_samples": [r.path for r in records],
        "uncertainty_map": str(synth_dir / "SYNTHETIC_uncertainty_std_display.png"),
        "selected_sample": "seed index 0",
    })
    result["validation_results"]["synthesis_qc"] = qc
    if not qc["passed"]:
        warnings.append(
            f"Synthetic view FAILED QC ({', '.join(qc['failed_checks'])}). It is saved for inspection but NOT passed to Volumea."
        )

    # 5. Volumea --------------------------------------------------------------
    synthetic_image = samples[0]
    if decision.family == "FRONTAL":
        frontal_img, lateral_img = loaded.image, synthetic_image
        frontal_rec, lateral_rec = original_record, generated
        route_ok = True
    else:
        frontal_img, lateral_img = synthetic_image, loaded.image
        frontal_rec, lateral_rec = generated, original_record
        route_ok = cfg.allow_lateral_to_volumea
    try:
        pair = build_input_pair(frontal_img, lateral_img, frontal_rec, lateral_rec, cfg, qc, route_ok)
    except RefusedInput as refusal:
        warnings.append(f"Volumea refused: {refusal}")
        return finish("stopped_qc_refusal")
    if not cfg.dvg_geometry_calibrated:
        warnings.append("DVG laterality/mirroring is UNCALIBRATED; CT orientation conventions are unverified.")
    reconstructor = comps.resolve("reconstructor")
    if reconstructor is None:
        warnings.append("Volumea reconstructor unavailable; CT not reconstructed. " + comps.load_errors.get("volumea", ""))
        return finish("stopped_volumea_unavailable")
    try:
        volume = reconstructor.reconstruct(pair, seed=cfg.dvg_seed)
    except Exception as error:
        warnings.append(f"Volumea reconstruction failed: {error}")
        return finish("failed_volumea")
    result["ct_reconstruction"] = save_reconstruction(
        volume, pair, run_dir / "ct", {"dvg_seed": cfg.dvg_seed, "synthesis_route": plan.route}
    )
    result["validation_results"]["ct_volume_stats"] = {
        "min": float(volume.min()), "max": float(volume.max()), "mean": float(volume.mean()),
        "fraction_zero": float((volume == 0).mean()), "ground_truth_available": False,
        "note": "No ground-truth CT exists for a single uploaded X-ray; volumetric accuracy cannot be measured here.",
    }
    return finish("completed")
