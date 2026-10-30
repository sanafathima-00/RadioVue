"""Generate the RadioVue single-X-ray Colab notebooks.

The notebooks are generated rather than hand-edited so that their shared setup
cells stay identical.  Run ``python notebooks/build_notebooks.py`` after
editing this file.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
        "cells": cells,
    }


SAFETY = """
> **Research use only.** Generated radiographs and CT volumes are synthetic and patient-conditioned. They are
> estimates, not acquisitions of the patient and not clinical ground truth. Never use them for diagnosis or treatment.
> Upload only images you are authorised to use, and remove burned-in identifiers first.
"""

CONFIG = '''
#@title 1. Configuration { display-mode: "form" }
REPO_URL = "https://github.com/sanafathima-00/RadioVue.git"  #@param {type:"string"}
REPO_BRANCH = "claude/single-xray-vlm-pipeline"  #@param {type:"string"}
#@markdown If the pipeline code is not pushed yet, upload `radiovue_colab_bundle.zip` (see SETUP_GUIDE.md) to this Drive path:
CODE_BUNDLE_ON_DRIVE = "RadioVue/code/radiovue_colab_bundle.zip"  #@param {type:"string"}
DRIVE_ROOT = "/content/drive/MyDrive/RadioVue"  #@param {type:"string"}
DVG_CHECKPOINTS_ON_DRIVE = DRIVE_ROOT + "/Volumea/models/DVG-Diffusion"
PARALLAX_MODEL_ON_DRIVE = DRIVE_ROOT + "/Parallax/models/DiT-fb-512"  # used if present, else downloaded from Hugging Face
OUTPUT_ROOT = DRIVE_ROOT + "/outputs/single_xray_pipeline"
REPO_DIR = "/content/RadioVue"
'''

BOOTSTRAP = '''
#@title 2. GPU, Drive and code
import os, sys, subprocess, zipfile, importlib.util
from pathlib import Path

smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(smi.stdout[:700] or smi.stderr)
assert smi.returncode == 0, "No GPU. Runtime > Change runtime type > GPU (L4/A100 recommended, T4 minimum)."

from google.colab import drive
drive.mount("/content/drive")

bundle = Path("/content/drive/MyDrive") / CODE_BUNDLE_ON_DRIVE
if not Path(REPO_DIR, "radiovue_pipeline").exists():
    if bundle.is_file():
        Path(REPO_DIR).mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(bundle).extractall(REPO_DIR)
        print("Code from Drive bundle:", bundle)
    else:
        subprocess.run(["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, REPO_DIR], check=True)
assert Path(REPO_DIR, "radiovue_pipeline").exists(), (
    "radiovue_pipeline/ not found. Push the branch that contains it, or upload the code bundle (SETUP_GUIDE.md).")
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print("Code revision:", rev or "(bundle, see BUNDLE_MANIFEST.json)")

# colab_setup has no third-party imports, so it can run before packages are installed.
spec = importlib.util.spec_from_file_location("colab_setup", f"{REPO_DIR}/radiovue_pipeline/colab_setup.py")
cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)
GPU = cs.gpu_report()
'''


def install_cell(parallax: bool, dvg: bool, medgemma: str) -> str:
    lines = ["#@title 3. Install dependencies and stage models (first run ~10-15 min)",
             f"cs.install_runtime(parallax={parallax}, dvg={dvg}, tools=True)"]
    if medgemma == "in_process":
        lines.append('cs.pip("transformers>=4.50", "accelerate>=1.1", extra=["-U"])  # this notebook does not load Parallax')
    if dvg:
        lines += ['cs.build_drr_projector(f"{REPO_DIR}/Volumea")',
                  "DVG_CKPT_DIR = cs.stage_dvg_checkpoints(DVG_CHECKPOINTS_ON_DRIVE, verify_hash=False)"]
    if parallax:
        lines += ['cs.ensure_parallax_shims(f"{REPO_DIR}/Parallax")',
                  "PARALLAX_MODEL_DIR = cs.stage_parallax_model(PARALLAX_MODEL_ON_DRIVE)"]
    if medgemma != "none":
        lines += ["HF_TOKEN = cs.hf_token_from_colab_secrets()  # value is never printed",
                  'assert HF_TOKEN, "Add HF_TOKEN in Colab Secrets (key icon) and enable notebook access."']
    if medgemma in ("in_process", "subprocess"):
        lines.append('cs.check_gated_access("google/medgemma-1.5-4b-it")')
    if medgemma == "subprocess":
        lines.append('MEDGEMMA_PYTHON = cs.make_medgemma_venv("/content/medgemma_venv")')
    return "\n".join(lines)


PUBLIC_IMAGES = '''
#@title Public CC0 test radiographs (same subject, same session; Wikimedia Commons, user Stillwaterising, 2010)
import hashlib, urllib.request
PUBLIC = {
    "pa_cc0_2010.png": ("https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
                        "a43431b089a6dfab031c10eceef6c821e4f326913bf9015f73f4a1f0446d2be1"),
    "lateral_cc0_2010.png": ("https://upload.wikimedia.org/wikipedia/commons/9/9d/Chest_Xray_Lateral_3-8-2010.png",
                             "d07a306fb285a8456739e0349fb208b64ade12eb4bb84bc09889c7b3ec7f99a7"),
}
PUBLIC_DIR = Path("/content/public_cc0"); PUBLIC_DIR.mkdir(exist_ok=True)
for name, (url, digest) in PUBLIC.items():
    path = PUBLIC_DIR / name
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "RadioVueResearch/0.1"})
        path.write_bytes(urllib.request.urlopen(req).read())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Checksum mismatch: {name}"
print("Public test images verified:", sorted(p.name for p in PUBLIC_DIR.iterdir()))
'''


# ---------------------------------------------------------------- 01 MedGemma

NB01 = [
    md("# 01 · MedGemma 1.5 4B: structured chest X-ray anatomical extraction\n" + SAFETY + """
**What this notebook does:** runs `google/medgemma-1.5-4b-it` on public CC0 radiographs. It produces the
`radiovue.anatomy.v1` JSON schema (view, structures, landmarks, relationships, findings, uncertainties), the official
MedGemma 1.5 anatomy bounding boxes (frontal only), and a cross-check of those boxes against a dedicated
segmentation model.

**Requirements:** a GPU (L4 24 GB recommended; a T4 should work in bf16 but slower; *unverified*), plus a Hugging Face
account that has accepted the HAI-DEF terms on the model page, with `HF_TOKEN` in Colab Secrets.

**Design rules:** semantic text and spatial output are separate. Coordinates in the semantic answer are rejected.
Boxes are approximate and are not masks. Decoding is greedy, and prompts are hashed so runs are reproducible.
"""),
    code(CONFIG), code(BOOTSTRAP), code(install_cell(parallax=False, dvg=False, medgemma="in_process")),
    code(PUBLIC_IMAGES),
    code('''
#@title 4. Load MedGemma (bf16) and the segmentation model
import json, time, torch
from radiovue_pipeline.anatomy_extraction import MedGemmaClient, run_extraction, compare_boxes_with_segmentation
from radiovue_pipeline.segmentation import AnatomySegmenter
from radiovue_pipeline.io_utils import load_xray
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
client = MedGemmaClient("google/medgemma-1.5-4b-it", max_new_tokens=1500)
print(f"Loaded in {time.time()-t0:.0f}s; revision={client.revision}; VRAM after load: {torch.cuda.memory_allocated()/1024**3:.2f} GiB")
segmenter = AnatomySegmenter()
OUT = Path(OUTPUT_ROOT) / "medgemma_examples"; OUT.mkdir(parents=True, exist_ok=True)
'''),
    code('''
#@title 5. Run structured extraction on the public PA and lateral images
results, timings = {}, {}
for name, frontal in [("pa_cc0_2010.png", True), ("lateral_cc0_2010.png", False)]:
    image = load_xray(PUBLIC_DIR / name).image
    t0 = time.time()
    extraction = run_extraction(image, client.generate, is_frontal=frontal, model_revision=client.revision)
    timings[name] = round(time.time() - t0, 1)
    if frontal:
        seg = segmenter.run(image, True, OUT / f"{name}_segmentation")
        extraction["segmentation"] = seg
        extraction["box_vs_mask_iou"] = compare_boxes_with_segmentation(extraction, seg["measurements_px"])
    results[name] = extraction
    (OUT / f"{Path(name).stem}_extraction.json").write_text(json.dumps(extraction, indent=2, default=str))
print("Seconds per image:", timings)
print("Peak VRAM GiB:", round(torch.cuda.max_memory_allocated()/1024**3, 2))
for name, r in results.items():
    print("\\n==", name, "| view:", r["image_metadata"], "| rejected spatial claims:", len(r["rejected_spatial_claims"]))
    print(json.dumps({k: r[k] for k in ("anatomical_structures", "spatial_relationships", "uncertainties")}, indent=1)[:2500])
'''),
    code('''
#@title 6. Box quality vs dedicated segmentation (cross-model agreement, not ground truth)
import pandas as pd
display(pd.DataFrame(results["pa_cc0_2010.png"]["box_vs_mask_iou"]))
'''),
    code('''
#@title 7. Draw MedGemma boxes (approximate) on the PA image
from PIL import ImageDraw
img = load_xray(PUBLIC_DIR / "pa_cc0_2010.png").image.convert("RGB")
draw = ImageDraw.Draw(img)
for item in results["pa_cc0_2010.png"]["localization"]["items"]:
    for b in item["boxes"]:
        if b["box_xyxy_px_original"]:
            draw.rectangle(b["box_xyxy_px_original"], outline=(255, 0, 0), width=6)
            draw.text((b["box_xyxy_px_original"][0] + 8, b["box_xyxy_px_original"][1] + 8), item["query"], fill=(255, 255, 0))
img.save(OUT / "pa_medgemma_boxes.png"); img.resize((img.width // 3, img.height // 3))
'''),
    code('''
#@title 8. Reproducibility check: identical prompt twice -> identical output under greedy decoding
from radiovue_pipeline.anatomy_extraction import VIEW_PROMPT
pa = load_xray(PUBLIC_DIR / "pa_cc0_2010.png").image.convert("RGB")
a, b = client.generate(pa, VIEW_PROMPT).response, client.generate(pa, VIEW_PROMPT).response
print("Identical:", a == b)
(OUT / "environment.json").write_text(json.dumps({"gpu": GPU, "torch": torch.__version__,
    "transformers": __import__("transformers").__version__, "model_revision": client.revision,
    "seconds_per_image": timings, "peak_vram_gib": round(torch.cuda.max_memory_allocated()/1024**3, 2),
    "greedy_reproducible": a == b}, indent=2))
print("Saved to", OUT)
'''),
]

# ---------------------------------------------------------------- 02 MedSigLIP

NB02 = [
    md("# 02 · MedSigLIP vs CLIP vs CNN embeddings for chest X-ray view synthesis\n" + SAFETY + """
**Question:** which visual representation carries information that a view-synthesis model could use, and could
it replace what Parallax (SV-DRR) conditions on today?

**Compared:** `google/medsiglip-448` (gated), Parallax's own CLIP image encoder (its real conditioning signal), and the
TorchXRayVision DenseNet-121 (a conventional CXR CNN).

**Probes:** zero-shot PA/AP/lateral. Cosine similarity of the PA vs its mirror image, its upside-down copy, the same
subject's lateral, and noise. An embedding that cannot tell a PA from its mirror image carries little laterality
or geometry information. With one public subject these probes are *descriptive only*.
"""),
    code(CONFIG), code(BOOTSTRAP), code(install_cell(parallax=True, dvg=False, medgemma="token_only")),
    code(PUBLIC_IMAGES),
    code('''
#@title 4. Embeddings
import json, numpy as np, torch
from PIL import Image
from radiovue_pipeline.io_utils import load_xray
from radiovue_pipeline.embeddings import MedSigLIPEmbedder, ParallaxClipEmbedder, XrvDenseNetEmbedder, summarize
cs.check_gated_access("google/medsiglip-448")
pa = load_xray(PUBLIC_DIR / "pa_cc0_2010.png").image
lat = load_xray(PUBLIC_DIR / "lateral_cc0_2010.png").image
probes = {"pa": pa, "lateral": lat, "pa_mirrored": pa.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
          "pa_upside_down": pa.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
          "noise": Image.fromarray((np.random.default_rng(0).random((512, 512)) * 255).astype(np.uint8))}
report = {}
for label, factory in [("medsiglip-448", lambda: MedSigLIPEmbedder()),
                       ("parallax-clip", lambda: ParallaxClipEmbedder(PARALLAX_MODEL_DIR)),
                       ("xrv-densenet121", lambda: XrvDenseNetEmbedder(device="cuda"))]:
    model = factory()
    report[label] = summarize(label, {k: model.embed(v) for k, v in probes.items()})
    if label == "medsiglip-448":
        report[label]["zero_shot_view"] = {k: model.zero_shot(probes[k]) for k in ("pa", "lateral")}
    del model; torch.cuda.empty_cache()
for label, r in report.items():
    c = r["cosine"]
    print(f"{label:16s} dim={r['dim']:5d}  PA~mirror={c['pa']['pa_mirrored']:.3f}  PA~upside={c['pa']['pa_upside_down']:.3f}  "
          f"PA~lateral={c['pa']['lateral']:.3f}  PA~noise={c['pa']['noise']:.3f}")
print("MedSigLIP zero-shot:", json.dumps(report["medsiglip-448"]["zero_shot_view"], indent=1))
OUT = Path(OUTPUT_ROOT) / "embeddings"; OUT.mkdir(parents=True, exist_ok=True)
(OUT / "embedding_probe_report.json").write_text(json.dumps(report, indent=2))
'''),
    md("""
**Interpreting the results.** Record them in `EXPERIMENT_RESULTS.md`. Parallax's `cc_projection` and its DiT cross-attention were
trained on CLIP features. Replacing CLIP with MedSigLIP therefore means retraining, even if MedSigLIP probes better.
The retraining data requirement is in `VIEW_SYNTHESIS_RESEARCH.md` §6.
"""),
]

# ---------------------------------------------------------------- 03 paired validation

NB03 = [
    md("# 03 · Paired-DRR validation and geometry calibration (Tests A/B/C)\n" + SAFETY + """
Ground truth comes from **CT volumes**. Each CT gives a ground-truth volume, display-oriented reference DRRs (NumPy),
SV-DRR-geometry DRRs (PA, AP-like and lateral), and DVG-projector views. Each test arm changes one variable at a time:

| Arm | slot 0 | slot 1 | isolates |
|---|---|---|---|
| A_native | DVG projector frontal | DVG projector lateral | DVG ceiling on its own domain |
| C_svdrr | SV-DRR PA | true SV-DRR lateral | DRR style / field-of-view mismatch (vs A) |
| B_synth | SV-DRR PA | **Parallax lateral** | synthesis error (vs C) |
| B_ap | SV-DRR AP-like | **Parallax lateral from AP** | PA-vs-AP sensitivity (vs B) |
| C_ap | SV-DRR AP-like | true SV-DRR lateral | AP frontal effect with a true lateral |

The notebook also **calibrates** the uncalibrated conventions in `PipelineConfig`: the Parallax lateral azimuth sign,
the Parallax output orientation, and the DVG slot transforms. It writes them to
`OUTPUT_ROOT/calibration/geometry_calibration.json`, which notebook 04 loads.

**Data authorisation.** Use only CTs you are authorised to process. LIDC-IDRI is **contaminated** for this purpose:
all 16 SV-DRR validation patients are in the X2CT/DVG training split (`validation_data/lidc_split/`). The cleanest
options are your own authorised local CTs, or a public non-LIDC chest CT collection (e.g. TCIA NSCLC-Radiomics).
"""),
    code(CONFIG), code(BOOTSTRAP), code(install_cell(parallax=True, dvg=True, medgemma="none")),
    code('''
#@title 4. Cases (DICOM CT series folders on Drive)
DATA_AUTHORISED = False  #@param {type:"boolean"}
CT_ROOT = DRIVE_ROOT + "/validation_ct"  #@param {type:"string"}
MAX_CASES = 8  #@param {type:"integer"}
SEEDS = [0, 1, 2, 3]
assert DATA_AUTHORISED, "Confirm you are authorised to process these CT series, then tick DATA_AUTHORISED."
case_dirs = sorted(p for p in Path(CT_ROOT).iterdir() if p.is_dir())[:MAX_CASES]
print(len(case_dirs), "case folders (names are not printed)")
VAL = Path(OUTPUT_ROOT) / "validation"; VAL.mkdir(parents=True, exist_ok=True)
'''),
    code('''
#@title 5. CT -> ground-truth volume, reference DRRs, SV-DRR DRRs (PA, AP-like, lateral +/-90)
import json, numpy as np, torch
from radiovue_pipeline import drr, calibration as cal
cases = {}
for i, d in enumerate(case_dirs):
    cid = f"case{i:02d}"
    ct = drr.load_dicom_ct_series(d)
    iso = drr.resample_isotropic(ct.hu, ct.spacing_zyx, 2.5)
    cube = drr.center_crop_or_pad(iso, 128, -1024.0)
    gt01 = (np.clip(cube + 1024.0, 0, 2500) / 2500).astype(np.float32)
    nii = VAL / f"{cid}_hu.nii.gz"; drr.write_nifti(cube, (2.5, 2.5, 2.5), nii)
    refs = cal.reference_views(cube)
    pa_r, ap_r = drr.SvdrrDrrRenderer(nii, 512, "PA"), drr.SvdrrDrrRenderer(nii, 512, "AP", sdd=1000.0, source_to_isocenter=850.0)
    raw = {"pa": pa_r.render(0), "ap": ap_r.render(0), "lat_p90": pa_r.render(90), "lat_m90": pa_r.render(-90)}
    fits = {"pa": cal.best_dihedral_match(raw["pa"], refs["frontal"]), "ap": cal.best_dihedral_match(raw["ap"], refs["frontal"]),
            "lat_p90": cal.best_dihedral_match(raw["lat_p90"], refs["lateral"]), "lat_m90": cal.best_dihedral_match(raw["lat_m90"], refs["lateral"])}
    disp = {k: cal.apply_transform(v, fits[k]["model_to_display"]) for k, v in raw.items()}
    cases[cid] = {"cube": cube, "gt01": gt01, "refs": refs, "display": disp, "svdrr_fits": fits, "meta": ct.metadata}
    print(cid, {k: (f["model_to_display"], round(f["ncc"], 3), round(f["margin_over_runner_up"], 3)) for k, f in fits.items()})
'''),
    code('''
#@title 6. Parallax synthesis from display-oriented PA and AP DRRs at both azimuth signs (then free GPU)
from PIL import Image
from radiovue_pipeline.view_synthesis import ParallaxSynthesizer
from radiovue_pipeline.io_utils import from_float01, to_float01
synth = ParallaxSynthesizer(f"{REPO_DIR}/Parallax", PARALLAX_MODEL_DIR, 512)
for cid, c in cases.items():
    c["synth"] = {}
    for src in ("pa", "ap"):
        for az in (90, -90):
            outs = synth.generate(from_float01(c["display"][src]), az, SEEDS)
            c["synth"][(src, az)] = [to_float01(o) for o in outs]
    print(cid, "done")
del synth; torch.cuda.empty_cache()
'''),
    code('''
#@title 7. Calibrate Parallax: which azimuth gives a lateral, and which transform maps its output to display orientation
votes = {}
for cid, c in cases.items():
    for az in (90, -90):
        f = cal.best_dihedral_match(c["synth"][("pa", az)][0], c["refs"]["lateral"])
        votes.setdefault(az, []).append(f)
summary = {az: {"mean_ncc": float(np.mean([f["ncc"] for f in fs])),
                "transforms": [f["model_to_display"] for f in fs]} for az, fs in votes.items()}
best_az = max(summary, key=lambda a: summary[a]["mean_ncc"])
transforms = summary[best_az]["transforms"]
best_tf = max(set(transforms), key=transforms.count)
consistent = transforms.count(best_tf) == len(transforms)
print(json.dumps(summary, indent=1)); print("Chosen azimuth:", best_az, "| output->display:", best_tf, "| consistent across cases:", consistent)
for c in cases.values():
    c["synth_pa"] = [cal.apply_transform(s, best_tf) for s in c["synth"][("pa", best_az)]]
    c["synth_ap"] = [cal.apply_transform(s, best_tf) for s in c["synth"][("ap", best_az)]]
    c["gt_lateral"] = c["display"]["lat_p90"] if c["svdrr_fits"]["lat_p90"]["ncc"] >= c["svdrr_fits"]["lat_m90"]["ncc"] else c["display"]["lat_m90"]
'''),
    code('''
#@title 8. Load Volumea, find DVG volume orientation (coarse self-consistency), run arms A/B/C
from radiovue_pipeline.volumea_adapter import VolumeaReconstructor
from radiovue_pipeline import validation as V
from radiovue_pipeline.metrics import LPIPSMetric
recon = VolumeaReconstructor(f"{REPO_DIR}/Volumea", DVG_CKPT_DIR)
first = next(iter(cases.values()))
ranking = V.search_dvg_volume_orientation(recon.model, first["gt01"], f"{REPO_DIR}/Volumea")
orientation = ranking[0]
print("Top orientations:", [(r["perm"], r["flips"], round(r["coarse_ncc"], 3)) for r in ranking[:4]])
lp = LPIPSMetric()
results = {}
for cid, c in cases.items():
    views = {"pa": c["display"]["pa"], "ap": c["display"]["ap"], "lateral": c["gt_lateral"]}
    r = V.run_case(cid, c["cube"], c["gt01"], orientation, views, c["synth_pa"], c["synth_ap"], recon, f"{REPO_DIR}/Volumea", lp)
    np.savez_compressed(VAL / f"{cid}_volumes.npz", **r.pop("volumes"))
    results[cid] = r
    print(cid, {arm: round(m["ssim_3d"], 3) for arm, m in r["arms"].items()})
'''),
    code('''
#@title 9. Aggregate metrics, patient specificity, QC calibration; write calibration JSON
import pandas as pd
from radiovue_pipeline.metrics import patient_specificity
from radiovue_pipeline.consistency import row_profile_consistency
rows = [{"case": cid, "arm": arm, **{k: v for k, v in m.items() if k != "note"}} for cid, r in results.items() for arm, m in r["arms"].items()]
arm_table = pd.DataFrame(rows); arm_table.to_csv(VAL / "ct_arm_metrics.csv", index=False)
display(arm_table.groupby("arm").agg(["mean", "std"]).round(4))
syn = pd.DataFrame([{"case": cid, "input": k, **r["synthesis_2d"][k]} for cid, r in results.items() for k in ("from_pa", "from_ap")])
syn.to_csv(VAL / "synthesis_2d_metrics.csv", index=False); display(syn.groupby("input").mean(numeric_only=True).round(4))
ids = list(cases)
spec = patient_specificity([cases[i]["synth_pa"][0] for i in ids], [cases[i]["gt_lateral"] for i in ids])
print("Patient specificity top-1:", spec["top1_identification"], "chance:", spec["chance"])
qc_true = [row_profile_consistency(cases[i]["display"]["pa"], cases[i]["synth_pa"][0])["row_profile_corr"] for i in ids]
qc_gt = [row_profile_consistency(cases[i]["display"]["pa"], cases[i]["gt_lateral"])["row_profile_corr"] for i in ids]
qc_mismatch = [row_profile_consistency(cases[i]["display"]["pa"], cases[j]["gt_lateral"])["row_profile_corr"] for i in ids for j in ids if i != j]
print("Row-profile corr  PA~synthetic:", np.round(qc_true, 3), " PA~true lateral:", np.round(qc_gt, 3),
      " PA~other patient's lateral (mean):", round(float(np.mean(qc_mismatch)), 3) if qc_mismatch else None)
dvg_front = [r["calibration"]["dvg_frontal"]["display_to_model"] for r in results.values()]
dvg_lat = [r["calibration"]["dvg_lateral"]["display_to_model"] for r in results.values()]
calib = {
    "lateral_azimuth_deg": float(best_az), "lateral_azimuth_calibrated": bool(consistent),
    "parallax_output_transform": best_tf,
    "dvg_frontal_transform": max(set(dvg_front), key=dvg_front.count),
    "dvg_lateral_transform": max(set(dvg_lat), key=dvg_lat.count),
    "dvg_geometry_calibrated": len(set(dvg_front)) == 1 and len(set(dvg_lat)) == 1,
    "dvg_volume_orientation": {"perm": list(orientation["perm"]), "flips": list(orientation["flips"]), "coarse_ncc": orientation["coarse_ncc"]},
    "evidence": {"n_cases": len(ids), "parallax_votes": summary, "patient_specificity": spec,
                 "row_profile": {"pa_vs_synthetic": qc_true, "pa_vs_true_lateral": qc_gt, "pa_vs_other_lateral_mean": float(np.mean(qc_mismatch)) if qc_mismatch else None}},
}
(Path(OUTPUT_ROOT) / "calibration").mkdir(parents=True, exist_ok=True)
(Path(OUTPUT_ROOT) / "calibration" / "geometry_calibration.json").write_text(json.dumps(calib, indent=2, default=str))
(VAL / "full_results.json").write_text(json.dumps(results, indent=2, default=str))
print(json.dumps({k: v for k, v in calib.items() if k != "evidence"}, indent=1))
'''),
    code('''
#@title 10. Visual comparison for one case (all synthetic panels labelled)
import matplotlib.pyplot as plt
cid = ids[0]; c = cases[cid]; vols = np.load(VAL / f"{cid}_volumes.npz")
fig, ax = plt.subplots(2, 6, figsize=(24, 8))
panels = [("PA DRR (input)", c["display"]["pa"]), ("True lateral DRR", c["gt_lateral"]), ("SYNTHETIC lateral (PA)", c["synth_pa"][0]),
          ("SYNTHETIC lateral (AP)", c["synth_ap"][0]), ("Seed std (uncertainty)", np.std(np.stack(c["synth_pa"]), 0)), ("AP-like DRR", c["display"]["ap"])]
for a, (t, im) in zip(ax[0], panels): a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
for a, k in zip(ax[1], ["ground_truth", "A_native", "C_svdrr", "B_synth", "B_ap", "C_ap"]):
    a.imshow(vols[k][:, 64, :], cmap="gray", vmin=0, vmax=1); a.set_title(k if k == "ground_truth" else f"SYNTHETIC CT: {k}"); a.axis("off")
plt.tight_layout(); plt.savefig(VAL / f"{cid}_comparison.png", dpi=110); plt.show()
'''),
]

# ---------------------------------------------------------------- 04 end to end

NB04 = [
    md("# 04 · RadioVue single-X-ray research pipeline (end to end)\n" + SAFETY + """
**How to use:** `Runtime > Change runtime type > GPU (L4 recommended)`, then `Runtime > Run all`.

1. Setup installs everything and stages the models (first run ~15 min).
2. **Step 5** runs the public CC0 PA demo, so you can confirm the pipeline works before using your own image.
3. **Step 6 is where you upload your X-ray** (one DICOM, PNG or JPEG). Optionally declare its view first.
4. Step 7 shows the original vs the SYNTHETIC view, the uncertainty map, QC, anatomy and the CT.

Everything goes through one function: `run_pipeline(input_xray)`. Outputs and provenance are saved to Drive under
`OUTPUT_ROOT`. A synthetic view that fails QC is **never** passed to Volumea.
"""),
    code(CONFIG), code(BOOTSTRAP), code(install_cell(parallax=True, dvg=True, medgemma="subprocess")),
    code('''
#@title 4. Build the pipeline (loads calibration from notebook 03 if it exists)
import json
from radiovue_pipeline import PipelineConfig, run_pipeline
from radiovue_pipeline.pipeline import Components
cfg = PipelineConfig(output_root=OUTPUT_ROOT, volumea_root=f"{REPO_DIR}/Volumea", parallax_root=f"{REPO_DIR}/Parallax",
                     dvg_checkpoint_dir=DVG_CKPT_DIR, parallax_model_dir=PARALLAX_MODEL_DIR)
calib_path = Path(OUTPUT_ROOT) / "calibration" / "geometry_calibration.json"
if calib_path.exists():
    calib = json.loads(calib_path.read_text())
    for key in ("lateral_azimuth_deg", "lateral_azimuth_calibrated", "parallax_output_transform",
                "dvg_frontal_transform", "dvg_lateral_transform", "dvg_geometry_calibrated"):
        setattr(cfg, key, calib[key])
    print("Geometry calibration loaded:", {k: calib[k] for k in calib if k != "evidence"})
else:
    print("WARNING: no geometry calibration found. Lateral sign and DVG orientation are UNCALIBRATED (run notebook 03).")
comps = Components.load(cfg, medgemma_python=MEDGEMMA_PYTHON, repo_root=REPO_DIR)
print("Load errors so far:", comps.load_errors or "none")
'''),
    code('''
#@title Display helper
import matplotlib.pyplot as plt
from PIL import Image
def show(result):
    print("STATUS:", result["status"], "| input view:", result["input_view"], "| generated:", result["generated_view_type"])
    for w in result["warnings"]: print("  !", w)
    panels = [("ORIGINAL (rendered)", result["original_image"]["rendered_grayscale"])]
    if result["generated_view"]:
        panels += [("SYNTHETIC view (seed 0)", result["generated_view"]["display_path"]),
                   ("Uncertainty: per-pixel std over seeds", result["view_synthesis_metadata"]["uncertainty_map"])]
    seg = (result.get("anatomical_extraction") or {}).get("segmentation") or {}
    if seg.get("overlay_path"): panels.append(("Segmentation (PSPNet)", seg["overlay_path"]))
    if result["ct_reconstruction"]:
        panels += [(f"SYNTHETIC CT {k[-5:]}", result["ct_reconstruction"][k]) for k in ("preview_axis0", "preview_axis1", "preview_axis2")]
    fig, ax = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    for a, (t, p) in zip(np.atleast_1d(ax), panels): a.imshow(Image.open(p), cmap="gray"); a.set_title(t); a.axis("off")
    plt.tight_layout(); plt.show()
    qc = result["validation_results"].get("synthesis_qc")
    if qc: print("QC passed:", qc["passed"], json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("passed", "gating", "row_profile_corr", "mean_pixel_std", "input_output_ncc", "p99_minus_p1")} for k, v in qc["checks"].items()}, indent=1))
    ext = result.get("anatomical_extraction") or {}
    if ext.get("image_metadata"): print("MedGemma view:", ext["image_metadata"])
    print("Run folder:", result["run_dir"])
import numpy as np
'''),
    code('''
#@title 5. Verification run on the public CC0 PA (declared PA because PNG has no view metadata)
''' + PUBLIC_IMAGES.split("\n", 2)[2] + '''
demo = run_pipeline(PUBLIC_DIR / "pa_cc0_2010.png", cfg, comps, declared_view="PA")
show(demo)
'''),
    md("""
## 6 · Upload your X-ray here
Choose the view only if you know it from the acquisition. Leave it as `auto` otherwise. AP is never relabelled PA.
A lateral input stops before synthesis by default, because no validated lateral-to-frontal generator exists.
"""),
    code('''
#@title 6. Upload ONE chest X-ray (DICOM / PNG / JPEG)
DECLARED_VIEW = "auto"  #@param ["auto", "PA", "AP", "LATERAL"]
from google.colab import files
uploaded = files.upload()
assert len(uploaded) == 1, "Upload exactly one file."
name = next(iter(uploaded))
UPLOAD_DIR = Path("/content/user_upload"); UPLOAD_DIR.mkdir(exist_ok=True)
user_path = UPLOAD_DIR / ("user_input" + Path(name).suffix.lower())
user_path.write_bytes(uploaded[name]); del uploaded
print("Stored as", user_path.name, f"({user_path.stat().st_size} bytes). Original filename is not recorded in outputs.")
'''),
    code('''
#@title 7. Run the pipeline on your X-ray
result = run_pipeline(user_path, cfg, comps, declared_view=None if DECLARED_VIEW == "auto" else DECLARED_VIEW)
show(result)
'''),
    code('''
#@title 8. Inspect the CT volume interactively (normalised DVG units, not HU)
import ipywidgets as W
if result["ct_reconstruction"]:
    vol = np.load(result["ct_reconstruction"]["npy"])
    def view(axis=0, index=64):
        sl = np.take(vol, index, axis=axis)
        plt.figure(figsize=(5, 5)); plt.imshow(sl, cmap="gray", vmin=0, vmax=1); plt.title(f"SYNTHETIC CT axis {axis} slice {index}"); plt.axis("off"); plt.show()
    W.interact(view, axis=W.IntSlider(0, 0, 2), index=W.IntSlider(64, 0, 127))
else:
    print("No CT was produced:", result["status"])
'''),
]


def main() -> None:
    outputs = {
        "01_MedGemma_Anatomical_Extraction.ipynb": NB01,
        "02_MedSigLIP_Embedding_Experiment.ipynb": NB02,
        "03_Paired_DRR_Validation.ipynb": NB03,
        "04_RadioVue_End_to_End.ipynb": NB04,
    }
    for name, cells in outputs.items():
        (HERE / name).write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
        print("wrote", name, len(cells), "cells")


if __name__ == "__main__":
    main()
