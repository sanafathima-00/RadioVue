"""CPU unit tests for the single-X-ray pipeline control flow, provenance and QC.

GPU models are replaced by fakes; these tests verify orchestration, safety
gates and metric correctness, not model quality.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from radiovue_pipeline import PipelineConfig, run_pipeline
from radiovue_pipeline.anatomy_extraction import (
    LOCALIZATION_PROMPT,
    GenerationRecord,
    extract_json,
    normalized_box_to_pixels,
    parse_localization,
    run_extraction,
    sanitize_semantic,
)
from radiovue_pipeline.consistency import evaluate_synthetic_view, row_profile_consistency
from radiovue_pipeline.drr import ct_to_dvg_volume, CTVolume, center_crop_or_pad, hu_to_mu, parallel_drr
from radiovue_pipeline.io_utils import pad_to_square
from radiovue_pipeline.metrics import dice, ncc, patient_specificity, psnr, ssim, volume_metrics
from radiovue_pipeline.pipeline import Components
from radiovue_pipeline.provenance import ORIGIN_ORIGINAL, ORIGIN_SYNTHETIC, ImageRecord, read_origin, save_synthetic
from radiovue_pipeline.view_classification import fuse_view_evidence
from radiovue_pipeline.view_synthesis import parallax_pose, plan_synthesis
from radiovue_pipeline.volumea_adapter import RefusedInput, build_input_pair, preprocess_for_dvg


# ---------------------------------------------------------------- fixtures

def phantom(seed: int = 0, n: int = 64) -> np.ndarray:
    """Anatomy-like HU phantom: body ellipse, two lungs, heart, spine (z, y, x)."""
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[0:n, 0:n, 0:n].astype(float)
    c = n / 2
    vol = np.full((n, n, n), -1000.0)
    body = ((y - c) / (0.30 * n)) ** 2 + ((x - c) / (0.42 * n)) ** 2 <= 1
    vol[body] = 40.0
    for side in (-1, 1):
        lung = ((y - c) / (0.20 * n)) ** 2 + ((x - c - side * 0.18 * n) / (0.13 * n)) ** 2 <= 1
        vol[lung & (z > 0.15 * n) & (z < (0.75 + 0.05 * rng.random()) * n)] = -800.0
    heart = ((y - c + 0.05 * n) / (0.12 * n)) ** 2 + ((x - c + 0.05 * n) / (0.12 * n)) ** 2 + ((z - 0.6 * n) / (0.15 * n)) ** 2 <= 1
    vol[heart] = 50.0
    spine = ((y - c - 0.22 * n) / (0.05 * n)) ** 2 + ((x - c) / (0.05 * n)) ** 2 <= 1
    vol[spine] = 700.0
    return vol + rng.normal(0, 5, vol.shape)


def to01(a):
    a = np.asarray(a, float)
    return (a - a.min()) / (a.max() - a.min())


class FakeClassifier:
    def __init__(self, frontal=0.98):
        self.frontal = frontal

    def predict(self, image):
        return {"Frontal": self.frontal, "Lateral": 1 - self.frontal}


class FakeSynthesizer:
    """Produces a parallel-beam lateral of a phantom: a physically consistent 'perfect' synthesis."""

    def __init__(self, lateral01, noise=0.0):
        self.lateral01, self.noise = lateral01, noise

    def generate(self, image, azimuth, seeds, steps, guidance):
        if azimuth == 0 and not getattr(self, "out_of_domain", False):  # faithful pose-0 re-rendering of the input
            return [image.convert("L").resize(self.lateral01.shape[::-1]) for _ in seeds]
        out = []
        for s in seeds:
            arr = self.lateral01 + np.random.default_rng(s).normal(0, self.noise, self.lateral01.shape)
            out.append(Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8)))
        return out

    def describe(self):
        return {"generator": "fake"}


class FakeReconstructor:
    def __init__(self):
        self.calls = []

    def reconstruct(self, pair, seed=42):
        self.calls.append(pair)
        return np.full((128, 128, 128), 0.3, dtype=np.float32)


def fake_medgemma(view="PA"):
    def generate(image, prompt):
        if "acquisition projection" in prompt:
            return GenerationRecord("v", f'```json{{"view": "{view}", "confidence": "medium", "evidence": ["scapulae"], "orientation_notes": []}}```')
        if "Where is the" in prompt:
            return GenerationRecord("l", 'reasoning... Final Answer: ```json[{"box_2d": [100, 150, 700, 480], "label": "right lung"}]```')
        return GenerationRecord("s", '<unused94>thinking<unused95>```json{"anatomical_structures": {"heart": {"visible": true, "description": "normal size", "confidence": "medium", "bbox": [1,2,3,4]}}, "spatial_relationships": ["heart left of midline"], "uncertainties": []}```')
    return generate


@pytest.fixture
def frontal_lateral_pair():
    mu = hu_to_mu(phantom())
    return to01(parallel_drr(mu, 0)), to01(parallel_drr(mu, 90))


def save_png(arr01, path):
    Image.fromarray(np.clip(arr01 * 255, 0, 255).astype(np.uint8)).save(path)
    return path


# ---------------------------------------------------------------- physics / QC

def test_parallel_beam_row_sums_are_rotation_invariant():
    mu = hu_to_mu(phantom())
    frontal, lateral = parallel_drr(mu, 0), parallel_drr(mu, 90)
    rel = np.abs(frontal.sum(axis=1) - lateral.sum(axis=1)) / frontal.sum(axis=1)
    assert rel.max() < 0.02  # equal up to interpolation error


def test_row_profile_consistency_separates_same_and_different_patient(frontal_lateral_pair):
    frontal, lateral = frontal_lateral_pair
    other = to01(parallel_drr(hu_to_mu(np.flip(phantom(5), axis=0)), 90))
    same = row_profile_consistency(frontal, lateral)["row_profile_corr"]
    diff = row_profile_consistency(frontal, other)["row_profile_corr"]
    assert same > 0.95 and same > diff


def test_qc_rejects_copy_of_input_and_accepts_consistent_lateral(frontal_lateral_pair):
    frontal, lateral = frontal_lateral_pair
    ok = evaluate_synthetic_view(frontal, [lateral, lateral], 0.8, 0.12, 0.15)
    assert ok["passed"], ok["failed_checks"]
    copy = evaluate_synthetic_view(frontal, [frontal, frontal], 0.8, 0.12, 0.15)
    assert not copy["passed"] and "not_a_copy" in copy["failed_checks"]
    noisy = [np.clip(lateral + np.random.default_rng(i).normal(0, 0.4, lateral.shape), 0, 1) for i in range(3)]
    assert "sample_agreement" in evaluate_synthetic_view(frontal, noisy, 0.8, 0.12, 0.15)["failed_checks"]


def test_row_profile_is_advisory_unless_configured_gating(frontal_lateral_pair):
    frontal, lateral = frontal_lateral_pair
    flipped = [np.roll(lateral, lateral.shape[0] // 2, axis=0)] * 2  # rows misaligned far beyond the shift search
    advisory = evaluate_synthetic_view(frontal, flipped, 0.8, 0.12, 0.15)
    assert advisory["passed"] and advisory["advisory_failures"] == ["row_profile"]
    gating = evaluate_synthetic_view(frontal, flipped, 0.8, 0.12, 0.15, gating_checks=("row_profile",))
    assert not gating["passed"] and gating["failed_checks"] == ["row_profile"]


# ---------------------------------------------------------------- calibration

@pytest.mark.parametrize("hidden", ["identity", "flip_lr", "rot90_cw", "transpose", "rot180"])
def test_dihedral_calibration_recovers_hidden_transform(hidden):
    from radiovue_pipeline.calibration import apply_transform, best_dihedral_match, reference_views

    ref = reference_views(phantom(3))["frontal"]
    ref = ref + np.linspace(0, ref.max() * 0.3, ref.shape[1])[None, :]  # break left/right symmetry
    model_space = apply_transform(ref, hidden)
    fit = best_dihedral_match(model_space, ref)
    assert np.allclose(apply_transform(model_space, fit["model_to_display"]), ref)
    assert fit["margin_over_runner_up"] > 0


def test_reference_views_orientation():
    from radiovue_pipeline.calibration import reference_views

    vol = phantom(0)
    views = reference_views(vol)
    assert views["frontal"].shape == (64, 64) and views["lateral"].shape == (64, 64)
    # spine (posterior, high attenuation) lies on the right half of the lateral reference (anterior on the left)
    lat = views["lateral"]
    assert lat[:, 40:].max() > lat[:, :24].max()


def test_fov_similarity_fit_recovers_scale_and_shift():
    from radiovue_pipeline.validation import _resize2d, _scale_about_centre, apply_similarity, fit_similarity, orient_volume

    ref = _resize2d(parallel_drr(hu_to_mu(phantom(2)), 0), 128)
    moved = np.roll(_scale_about_centre(ref, 0.8, 128), (4, -6), axis=(0, 1))
    fit = fit_similarity(moved, ref, max_shift=12)
    assert fit["ncc"] > 0.9 and 1.15 <= fit["scale"] <= 1.4  # true inverse 1.25; 2-px shift grid trades off with scale
    assert ncc(apply_similarity(moved, fit), ref) > 0.9
    vol = np.arange(24).reshape(2, 3, 4)
    assert orient_volume(vol, (2, 1, 0), (True, False, False)).shape == (4, 3, 2)


# ---------------------------------------------------------------- metrics

def test_metrics_basics():
    a = np.random.default_rng(0).random((32, 32))
    assert psnr(a, a) == float("inf") and ssim(a, a) == pytest.approx(1.0)
    assert dice(a > 0.5, a > 0.5) == 1.0
    v = np.random.default_rng(1).random((24, 24, 24)) * 0.5
    m = volume_metrics(v, v)
    assert m["ssim_3d"] == pytest.approx(1.0) and m["mae"] == 0.0


def test_patient_specificity_detects_generic_generator():
    gts = [to01(parallel_drr(hu_to_mu(phantom(s, 48)), 90)) for s in range(3)]
    gts = [np.roll(g, 3 * i, axis=0) for i, g in enumerate(gts)]
    assert patient_specificity(gts, gts)["top1_identification"] == 1.0
    generic = [gts[0]] * 3
    assert patient_specificity(generic, gts)["top1_identification"] <= 1 / 3 + 1e-9


def test_ct_to_dvg_volume_normalisation():
    ct = CTVolume(np.full((40, 50, 50), -1024.0), (2.5, 2.5, 2.5), {})
    ct.hu[20, 25, 25] = 1476.0
    vol = ct_to_dvg_volume(ct)
    assert vol.shape == (128, 128, 128) and vol.min() == 0.0 and vol.max() <= 1.0
    assert center_crop_or_pad(np.ones((200, 10, 128)), 128, 0).shape == (128, 128, 128)


# ---------------------------------------------------------------- MedGemma parsing

def test_extract_json_handles_thinking_and_fences():
    assert extract_json('<unused94>x<unused95>```json{"a": 1}```') == {"a": 1}
    assert extract_json('Final Answer: [{"box_2d": [1, 2, 3, 4]}]') == [{"box_2d": [1, 2, 3, 4]}]
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_semantic_coordinates_are_rejected_not_used():
    clean, rejected = sanitize_semantic({"anatomical_structures": {"heart": {"visible": True, "bbox": [1, 2, 3, 4]}, "spine": {"visible": True}}})
    assert "heart" not in clean["anatomical_structures"] and "spine" in clean["anatomical_structures"]
    assert rejected


def test_localization_box_maps_back_through_square_padding():
    image = Image.new("RGB", (200, 100))
    padded, padding = pad_to_square(image)
    assert padded.size == (200, 200) and padding == (0, 50, 0, 50)
    # box covering the full original image area in padded normalised coords
    box = normalized_box_to_pixels([250, 0, 750, 1000], 200, padding, (200, 100))
    assert box == pytest.approx([0, 0, 200, 100])
    entry = parse_localization('```json[{"box_2d": [900, 10, 100, 20], "label": "x"}]```', "x", 200, padding, (200, 100))
    assert not entry["parse_ok"]  # y0 > y1 is geometrically invalid


def test_box_vs_mask_comparison():
    from radiovue_pipeline.anatomy_extraction import box_iou, compare_boxes_with_segmentation

    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0 and box_iou([0, 0, 1, 1], [5, 5, 6, 6]) == 0.0
    extraction = {"localization": {"items": [
        {"query": "right lung", "boxes": [{"geometrically_valid": True, "box_xyxy_px_original": [0, 0, 10, 10]}]},
        {"query": "right hemidiaphragm", "boxes": [{"geometrically_valid": True, "box_xyxy_px_original": [0, 0, 1, 1]}]},
    ]}}
    rows = compare_boxes_with_segmentation(extraction, {"Right Lung": {"present": True, "bbox_xyxy_px": [0, 0, 10, 5]}})
    assert rows[0]["iou"] == pytest.approx(0.5) and rows[1]["iou"] is None


def test_localization_prompt_matches_official_braces():
    rendered = LOCALIZATION_PROMPT.format(object_name="right clavicle")
    assert '[{"box_2d": [800, 3, 840, 471], "label": "car"}' in rendered and "Where is the right clavicle?" in rendered


def test_run_extraction_schema_and_lateral_skip():
    image = Image.new("L", (300, 200), 128)
    frontal = run_extraction(image, fake_medgemma("PA"), is_frontal=True)
    assert frontal["image_metadata"]["view"] == "PA" and frontal["image_metadata"]["confidence_calibrated"] is False
    assert frontal["rejected_spatial_claims"] and frontal["anatomical_structures"]["heart"] == {}
    assert all(item["parse_ok"] for item in frontal["localization"]["items"])
    lateral = run_extraction(image, fake_medgemma("LATERAL"), is_frontal=False)
    assert lateral["localization"]["items"] == [] and "skipped_reason" in lateral["localization"]


# ---------------------------------------------------------------- view logic

@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(metadata_view="AP", classifier={"Frontal": 0.99, "Lateral": 0.01}), "AP"),
        (dict(metadata_view="PA", declared_view="AP"), "UNKNOWN"),
        (dict(declared_view="PA", classifier={"Frontal": 0.02, "Lateral": 0.98}), "UNKNOWN"),
        (dict(classifier={"Frontal": 0.99, "Lateral": 0.01}, vlm_view="PA"), "FRONTAL_UNSPECIFIED"),
        (dict(classifier={"Frontal": 0.05, "Lateral": 0.95}), "LATERAL"),
        (dict(classifier={"Frontal": 0.6, "Lateral": 0.4}), "UNKNOWN"),
        (dict(metadata_view="OBLIQUE"), "UNKNOWN"),
    ],
)
def test_view_fusion(kwargs, expected):
    assert fuse_view_evidence(**kwargs).view == expected


def test_ap_is_never_relabelled_pa():
    d = fuse_view_evidence(metadata_view="AP", vlm_view="PA", classifier={"Frontal": 0.99, "Lateral": 0.01})
    assert d.view == "AP"


def test_synthesis_plan_policy():
    cfg = PipelineConfig()
    assert plan_synthesis("PA", cfg).allowed and plan_synthesis("PA", cfg).in_distribution
    assert plan_synthesis("AP", cfg).allowed and not plan_synthesis("AP", cfg).in_distribution
    lat = plan_synthesis("LATERAL", cfg)
    assert not lat.allowed and lat.target_view == "FRONTAL_SYNTHETIC_DRR_LIKE"
    assert not plan_synthesis("UNKNOWN", cfg).allowed
    assert list(parallax_pose(90)) == [0.0, -90.0, 0.0]
    assert cfg.lateral_azimuth_deg == -90.0  # selected on GPU evidence, see config.py


# ---------------------------------------------------------------- provenance / adapter

def test_synthetic_outputs_are_labelled(tmp_path):
    rec = save_synthetic(Image.new("L", (64, 64), 100), tmp_path, "lateral_seed0", "LATERAL", "fake", {"seed": 0})
    assert "SYNTHETIC_" in rec.path and read_origin(rec.path) == ORIGIN_SYNTHETIC
    assert rec.display_path and json.loads((tmp_path / "SYNTHETIC_lateral_seed0.provenance.json").read_text())["origin"] == "SYNTHETIC"
    with pytest.raises(FileExistsError):
        save_synthetic(Image.new("L", (64, 64)), tmp_path, "lateral_seed0", "LATERAL", "fake", {})


def test_dvg_preprocessing_modes():
    img = Image.fromarray((np.random.default_rng(0).random((900, 700)) * 255).astype(np.uint8))
    for mode in ("repo_center_crop", "contain_pad"):
        arr = preprocess_for_dvg(img, mode)
        assert arr.shape == (128, 128) and arr.dtype == np.float32 and 0 <= arr.min() and arr.max() <= 1
    with pytest.raises(ValueError):
        preprocess_for_dvg(img, "stretch")


def test_adapter_refuses_failed_qc_and_wrong_slots():
    cfg = PipelineConfig()
    img = Image.new("L", (256, 256), 90)
    front = ImageRecord("f.png", ORIGIN_ORIGINAL, "PA", "x", 256, 256)
    lat = ImageRecord("l.png", ORIGIN_SYNTHETIC, "LATERAL", "y", 256, 256)
    with pytest.raises(RefusedInput):
        build_input_pair(img, img, front, lat, cfg, {"passed": False, "failed_checks": ["row_profile"]}, True)
    with pytest.raises(RefusedInput):
        build_input_pair(img, img, lat, front, cfg, {"passed": True}, True)
    pair = build_input_pair(img, img, front, lat, cfg, {"passed": True}, True)
    assert pair.frontal.shape == (128, 128) and pair.preprocessing["slot0"].startswith("frontal")


# ---------------------------------------------------------------- end to end (fakes)

def _cfg(tmp_path, **kw):
    return PipelineConfig(output_root=str(tmp_path / "runs"), synthesis_seeds=[0, 1], **kw)


def test_run_pipeline_pa_completes_with_provenance(tmp_path, frontal_lateral_pair):
    frontal, lateral = frontal_lateral_pair
    src = save_png(frontal, tmp_path / "input.png")
    recon = FakeReconstructor()
    comps = Components(view_classifier=FakeClassifier(), medgemma_generate=fake_medgemma("PA"),
                       synthesizer=FakeSynthesizer(lateral, noise=0.01), reconstructor=recon)
    out = run_pipeline(src, _cfg(tmp_path), comps, declared_view="PA")
    assert out["status"] == "completed", out["warnings"]
    for key in ("input_view", "anatomical_extraction", "original_image", "generated_view", "generated_view_type",
                "view_synthesis_metadata", "ct_reconstruction", "validation_results", "warnings"):
        assert key in out
    assert out["input_view"] == "PA" and out["generated_view"]["origin"] == "SYNTHETIC"
    assert read_origin(out["generated_view"]["path"]) == "SYNTHETIC"
    assert recon.calls[0].frontal_record.origin == "ORIGINAL" and recon.calls[0].lateral_record.origin == "SYNTHETIC"
    assert any("not yet validated" in w for w in out["warnings"])
    assert (tmp_path / "input.png").read_bytes() == open(out["original_image"]["preserved_file"], "rb").read()


def test_run_pipeline_refuses_volumea_when_synthesis_inconsistent(tmp_path, frontal_lateral_pair):
    frontal, _ = frontal_lateral_pair
    src = save_png(frontal, tmp_path / "input.png")
    recon = FakeReconstructor()
    comps = Components(view_classifier=FakeClassifier(), synthesizer=FakeSynthesizer(frontal), reconstructor=recon)
    out = run_pipeline(src, _cfg(tmp_path), comps, declared_view="PA")
    assert out["status"] == "stopped_qc_refusal" and not recon.calls and out["ct_reconstruction"] is None


def test_run_pipeline_lateral_input_stops_by_default(tmp_path, frontal_lateral_pair):
    _, lateral = frontal_lateral_pair
    src = save_png(lateral, tmp_path / "lat.png")
    recon = FakeReconstructor()
    comps = Components(view_classifier=FakeClassifier(frontal=0.03), synthesizer=FakeSynthesizer(lateral), reconstructor=recon)
    out = run_pipeline(src, _cfg(tmp_path), comps)
    assert out["input_view"] == "LATERAL" and out["status"] == "stopped_no_supported_synthesis" and not recon.calls


def test_run_pipeline_ap_route_is_labelled_experimental(tmp_path, frontal_lateral_pair):
    frontal, lateral = frontal_lateral_pair
    src = save_png(frontal, tmp_path / "ap.png")
    comps = Components(view_classifier=FakeClassifier(), synthesizer=FakeSynthesizer(lateral, 0.01), reconstructor=FakeReconstructor())
    out = run_pipeline(src, _cfg(tmp_path), comps, declared_view="AP")
    assert out["input_view"] == "AP" and "EXPERIMENTAL" in out["view_synthesis_metadata"]["route"]
    assert out["status"] == "completed"


def test_lazy_components_and_isolated_medgemma(tmp_path, frontal_lateral_pair):
    from radiovue_pipeline.pipeline import Lazy

    frontal, lateral = frontal_lateral_pair
    src = save_png(frontal, tmp_path / "input.png")

    class FakeIsolated:
        def extract(self, image, is_frontal):
            return run_extraction(image, fake_medgemma("PA"), is_frontal)

    def broken():
        raise RuntimeError("Volumea requires a CUDA GPU runtime.")

    comps = Components(view_classifier=FakeClassifier(), medgemma_extract=FakeIsolated(),
                       synthesizer=Lazy("parallax", lambda: FakeSynthesizer(lateral, 0.01)),
                       reconstructor=Lazy("volumea", broken))
    out = run_pipeline(src, _cfg(tmp_path), comps, declared_view="PA")
    assert out["anatomical_extraction"]["image_metadata"]["view"] == "PA"
    assert out["generated_view"] is not None and out["status"] == "stopped_volumea_unavailable"
    assert any("CUDA" in w for w in out["warnings"])


def test_run_pipeline_without_gpu_components_reports_blockers(tmp_path, frontal_lateral_pair):
    frontal, _ = frontal_lateral_pair
    src = save_png(frontal, tmp_path / "input.png")
    out = run_pipeline(src, _cfg(tmp_path), Components(load_errors={"parallax": "RuntimeError: no CUDA"}), declared_view="PA")
    assert out["status"] == "stopped_synthesizer_unavailable" and out["generated_view"] is None
    assert any("parallax" in w for w in out["warnings"])


# ---------------------------------------------------------------- app helpers (gradio not required)

def test_app_helpers(tmp_path, frontal_lateral_pair):
    from radiovue_pipeline.app import ct_slice, status_markdown, zip_run

    frontal, lateral = frontal_lateral_pair
    src = save_png(frontal, tmp_path / "input.png")
    comps = Components(view_classifier=FakeClassifier(), synthesizer=FakeSynthesizer(lateral, 0.01), reconstructor=FakeReconstructor())
    out = run_pipeline(src, _cfg(tmp_path), comps, declared_view="PA")
    text = status_markdown(out)
    assert "`completed`" in text and "not HU" in text and "Warnings" in text
    image = ct_slice(out["ct_reconstruction"]["npy"], 1, 500)  # out-of-range index is clamped
    assert image.size == (384, 384) and ct_slice(None, 0, 0) is None
    archive = zip_run(out["run_dir"])
    assert archive.endswith(".zip") and (tmp_path / "runs").exists()


def test_released_lazy_components_reload_on_next_run(tmp_path, frontal_lateral_pair):
    from radiovue_pipeline.pipeline import Lazy

    frontal, lateral = frontal_lateral_pair
    src = save_png(frontal, tmp_path / "input.png")
    builds = []
    factory = lambda: builds.append(1) or FakeSynthesizer(lateral, 0.01)
    comps = Components(view_classifier=FakeClassifier(), synthesizer=Lazy("parallax", factory), reconstructor=FakeReconstructor())
    cfg = _cfg(tmp_path, free_gpu_between_stages=True)
    first = run_pipeline(src, cfg, comps, declared_view="PA")
    second = run_pipeline(src, cfg, comps, declared_view="PA")
    assert first["status"] == second["status"] == "completed" and len(builds) == 2


def test_out_of_domain_input_is_refused_before_volumea(tmp_path, frontal_lateral_pair):
    frontal, lateral = frontal_lateral_pair
    src = save_png(frontal, tmp_path / "input.png")
    synth = FakeSynthesizer(lateral, 0.01)
    synth.out_of_domain = True  # cannot re-render its input at pose 0, like the hallucinating CC0 subject
    recon = FakeReconstructor()
    out = run_pipeline(src, _cfg(tmp_path), Components(view_classifier=FakeClassifier(), synthesizer=synth, reconstructor=recon),
                       declared_view="PA")
    assert out["status"] == "stopped_qc_refusal" and not recon.calls
    assert "self_reconstruction" in out["validation_results"]["synthesis_qc"]["failed_checks"]
    assert any("outside the view-synthesis model's domain" in w for w in out["warnings"])


def test_self_reconstruction_fidelity_separates_faithful_and_corrupted(frontal_lateral_pair):
    from radiovue_pipeline.consistency import self_reconstruction_fidelity

    frontal, _ = frontal_lateral_pair
    noisy = np.clip(np.random.default_rng(0).normal(0.5, 0.3, frontal.shape), 0, 1)
    assert self_reconstruction_fidelity(frontal, frontal)["edge_ncc"] > 0.99
    assert self_reconstruction_fidelity(frontal, noisy)["edge_ncc"] < 0.4
