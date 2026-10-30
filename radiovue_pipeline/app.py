"""Single entry point: upload one chest X-ray, click Run, get the synthetic view and the CT.

Launched from ``notebooks/05_RadioVue_App.ipynb`` on a Colab GPU, which prints a
public Gradio link.  Every stage runs inside ``run_pipeline``.  This module only
renders inputs and outputs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import PipelineConfig
from .pipeline import Components, run_pipeline
from .provenance import stamp_synthetic

DISCLAIMER = (
    "**Research use only.** The generated view and CT are synthetic, patient-conditioned estimates. They are not "
    "acquisitions of this patient and not clinical ground truth. Do not use them for diagnosis or treatment. "
    "Upload only images you are authorised to use, with burned-in identifiers removed."
)

AXES = {"Axial (axis 0)": 0, "Coronal (axis 1)": 1, "Sagittal (axis 2)": 2}


def status_markdown(result: dict[str, Any]) -> str:
    qc = (result.get("validation_results") or {}).get("synthesis_qc") or {}
    meta = result.get("view_synthesis_metadata") or {}
    lines = [
        f"**Status:** `{result['status']}`",
        f"**Detected input view:** `{result.get('input_view')}`",
        f"**Generated view:** `{result.get('generated_view_type')}`  ·  route: {meta.get('route', 'n/a')}",
    ]
    if qc:
        lines.append(f"**Synthetic-view QC:** {'passed' if qc.get('passed') else 'FAILED'}")
    if result.get("ct_reconstruction"):
        lines.append("**CT:** reconstructed (128³, normalised DVG units, not HU)")
    if result.get("warnings"):
        lines.append("**Warnings:**\n" + "\n".join(f"- {w}" for w in result["warnings"]))
    return "\n\n".join(lines)


def ct_slice(npy_path: str | None, axis: int, index: int, size: int = 384) -> Image.Image | None:
    """One CT slice as a labelled display image."""
    if not npy_path:
        return None
    volume = np.load(npy_path, mmap_mode="r")
    index = int(np.clip(index, 0, volume.shape[axis] - 1))
    plane = np.clip(np.take(volume, index, axis=axis), 0.0, 1.0)
    image = Image.fromarray(np.rint(plane * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    return stamp_synthetic(image, f"SYNTHETIC CT axis {axis} slice {index} | not HU")


def zip_run(run_dir: str) -> str:
    """Archive the whole run folder (original, synthetic views, CT, provenance) for download."""
    return shutil.make_archive(run_dir, "zip", run_dir)


def build_app(cfg: PipelineConfig, comps: Components):
    import gradio as gr

    def run(file_path: str | None, declared: str):
        if not file_path:
            raise gr.Error("Upload one chest X-ray first.")
        result = run_pipeline(file_path, cfg, comps, declared_view=None if declared == "auto" else declared)
        generated = (result.get("generated_view") or {}).get("display_path")
        uncertainty = (result.get("view_synthesis_metadata") or {}).get("uncertainty_map")
        npy = (result.get("ct_reconstruction") or {}).get("npy")
        files = [zip_run(result["run_dir"])]
        if npy:
            files += [p for k, p in result["ct_reconstruction"].items() if k in ("npy", "nifti")]
        return (
            status_markdown(result),
            result["original_image"]["rendered_grayscale"] if result.get("original_image") else None,
            generated,
            uncertainty,
            ct_slice(npy, 0, 64),
            ct_slice(npy, 1, 64),
            ct_slice(npy, 2, 64),
            npy,
            json.dumps(result, indent=2, default=str),
            files,
        )

    with gr.Blocks(title="RadioVue single-X-ray research pipeline") as demo:
        gr.Markdown("# RadioVue: one chest X-ray → synthetic complementary view → CT (research prototype)")
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            upload = gr.File(label="Chest X-ray (DICOM, PNG or JPEG)", type="filepath", file_types=[".dcm", ".dicom", ".png", ".jpg", ".jpeg"])
            declared = gr.Dropdown(["auto", "PA", "AP", "LATERAL"], value="auto", label="Acquisition view (only if known)")
            button = gr.Button("Run", variant="primary")
        status = gr.Markdown()
        with gr.Row():
            original = gr.Image(label="ORIGINAL (rendered)", type="filepath")
            synthetic = gr.Image(label="SYNTHETIC complementary view", type="filepath")
            uncertainty = gr.Image(label="Uncertainty: per-pixel std over seeds", type="filepath")
        with gr.Row():
            axial, coronal, sagittal = (gr.Image(label=f"SYNTHETIC CT {name}", type="pil") for name in ("axial", "coronal", "sagittal"))
        npy_state = gr.State()
        with gr.Row():
            axis = gr.Radio(list(AXES), value="Axial (axis 0)", label="Browse CT")
            index = gr.Slider(0, 127, value=64, step=1, label="Slice")
            browser = gr.Image(label="CT slice browser", type="pil")
        downloads = gr.File(label="Downloads (full run folder, CT .npy / .nii.gz)", file_count="multiple")
        with gr.Accordion("Full result JSON (provenance, QC, anatomy)", open=False):
            raw = gr.Code(language="json")

        button.click(run, [upload, declared],
                     [status, original, synthetic, uncertainty, axial, coronal, sagittal, npy_state, raw, downloads])
        browse = lambda path, ax, i: ct_slice(path, AXES[ax], i)
        index.change(browse, [npy_state, axis, index], browser)
        axis.change(browse, [npy_state, axis, index], browser)
    return demo
