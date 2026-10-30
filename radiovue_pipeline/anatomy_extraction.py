"""MedGemma 1.5 structured anatomical extraction.

Design rules (see MEDGEMMA_SETUP.md):

* Semantic description and spatial localization are separate calls.
* The semantic call is told not to emit coordinates.  Any coordinate-like keys
  it emits anyway are moved to ``rejected_spatial_claims`` and never used.
* Spatial output comes only from the localization prompt that MedGemma 1.5 was
  trained on (Chest ImaGenome boxes, official Google notebook).  The boxes are
  coarse and approximate.  They are never segmentation masks.  Precise
  boundaries come from ``segmentation.py``.
* Localization is only requested for frontal images, because Chest ImaGenome
  is a frontal dataset.
* VLM self-reported confidence is uncalibrated and is labelled that way.
* Decoding is greedy (``do_sample=False``) and every prompt is hashed, so a run
  is reproducible for a fixed model revision and software stack.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image

from .io_utils import pad_to_square
from .provenance import utc_now

SCHEMA_VERSION = "radiovue.anatomy.v1"

STRUCTURES = (
    "left_lung",
    "right_lung",
    "heart",
    "mediastinum",
    "spine",
    "ribs",
    "clavicles",
    "diaphragm",
    "chest_wall",
    "trachea",
)

# Chest ImaGenome-style anatomy names (the localization training vocabulary).
FRONTAL_LOCALIZATION_TARGETS = (
    "right lung",
    "left lung",
    "cardiac silhouette",
    "mediastinum",
    "trachea",
    "right clavicle",
    "left clavicle",
    "right hemidiaphragm",
    "left hemidiaphragm",
    "spine",
)

SPATIAL_KEYS = {"x", "y", "x0", "y0", "x1", "y1", "bbox", "box", "box_2d", "coordinates", "coords", "pixel", "pixels", "mm", "cm"}

VIEW_PROMPT = """You are assisting a research pipeline. Look at this chest radiograph and identify the acquisition projection.
Answer ONLY with a JSON object enclosed in ```json ... ``` with exactly these keys:
{"view": "PA" | "AP" | "LATERAL" | "UNKNOWN",
 "confidence": "low" | "medium" | "high",
 "evidence": [short strings naming the visual cues you used, e.g. scapular position, clavicle orientation, markers such as "AP" or "portable", cardiac magnification],
 "orientation_notes": [short strings, e.g. laterality marker position, rotation, whether the image appears mirrored]}
If you cannot distinguish PA from AP reliably, say "UNKNOWN" for view rather than guessing."""

SEMANTIC_PROMPT = """You are assisting a medical-imaging research pipeline. Describe the anatomy visible in this chest radiograph.
Rules:
- Do NOT output any coordinates, pixel positions, bounding boxes or measurements in millimetres. Describe positions in words only (e.g. "upper zone", "left of midline").
- Distinguish what is clearly visible from what is uncertain. Use "uncertain" rather than guessing.
- Do not give a diagnosis or clinical advice. Findings are descriptive research annotations only.
Answer ONLY with one JSON object enclosed in ```json ... ``` with exactly these keys:
{"anatomical_structures": {STRUCTURE_KEYS},
 "anatomical_landmarks": [{"name": str, "description": str, "visible": true | false | "uncertain"}],
 "spatial_relationships": [str],
 "visible_findings": [{"finding": str, "location_words": str, "confidence": "low" | "medium" | "high"}],
 "image_quality": {"rotation": str, "inspiration": str, "penetration": str, "artifacts": [str]},
 "uncertainties": [str],
 "limitations": [str]}
Each value in anatomical_structures must be {"visible": true | false | "uncertain", "description": str, "confidence": "low" | "medium" | "high"}."""

# Verbatim structure of the official MedGemma 1.5 localization prompt
# (google-health/medgemma notebooks/cxr_anatomy_localization_with_hugging_face.ipynb).
LOCALIZATION_PROMPT = """Instructions:
The following user query will require outputting bounding boxes. The format of bounding boxes coordinates is [y0, x0, y1, x1] where (y0, x0) must be top-left corner and (y1, x1) the bottom-right corner. This implies that x0 < x1 and y0 < y1. Always normalize the x and y coordinates the range [0, 1000], meaning that a bounding box starting at 15% of the image width would be associated with an x coordinate of 150. You MUST output a single parseable json list of objects enclosed into ```json...``` brackets, for instance ```json[{{"box_2d": [800, 3, 840, 471], "label": "car"}}, {{"box_2d": [400, 22, 600, 73], "label": "dog"}}]``` is a valid output. Now answer to the user query.

Remember "left" refers to the patient's left side where the heart is and sometimes underneath an L in the upper right corner of the image.

Query:
Where is the {object_name}? Don't give a final answer without reasoning. Output the final answer in the format "Final Answer: X" where X is a JSON list of objects. The object needs a "box_2d" and "label" key. Answer:"""


def semantic_prompt() -> str:
    keys = ", ".join(f'"{name}": {{...}}' for name in STRUCTURES)
    return SEMANTIC_PROMPT.replace("{STRUCTURE_KEYS}", "{" + keys + "}")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Response parsing (pure, unit-tested)
# ---------------------------------------------------------------------------

def strip_thinking(response: str) -> str:
    """Remove MedGemma's optional thinking trace (between <unused94> and <unused95>)."""
    if "<unused95>" in response:
        return response.split("<unused95>", 1)[1].lstrip()
    return response


def extract_json(response: str) -> Any:
    """Parse the last ```json fenced block, else the outermost {...} or [...] span."""
    text = strip_thinking(response)
    fences = re.findall(r"```json\s*(.*?)```", text, flags=re.DOTALL)
    candidates = list(reversed(fences))
    spans = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            spans.append((start, text[start : end + 1]))
    candidates += [span for _, span in sorted(spans)]  # outermost (earliest-opening) first
    for candidate in candidates:
        try:
            return json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
    raise ValueError("No parseable JSON in model response.")


def _contains_spatial(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(k).lower() in SPATIAL_KEYS or _contains_spatial(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_contains_spatial(v) for v in value)
    return False


def sanitize_semantic(payload: dict[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    """Drop coordinate-like content from the semantic answer and return it separately."""
    rejected: list[Any] = []
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        if str(key).lower() in SPATIAL_KEYS:
            rejected.append({key: value})
            continue
        if isinstance(value, dict):
            kept = {}
            for sub_key, sub_value in value.items():
                if str(sub_key).lower() in SPATIAL_KEYS or _contains_spatial(sub_value):
                    rejected.append({f"{key}.{sub_key}": sub_value})
                else:
                    kept[sub_key] = sub_value
            clean[key] = kept
        elif isinstance(value, list):
            kept_list = []
            for item in value:
                if _contains_spatial(item):
                    rejected.append({key: item})
                else:
                    kept_list.append(item)
            clean[key] = kept_list
        else:
            clean[key] = value
    return clean, rejected


def normalized_box_to_pixels(
    box_2d: list[float], padded_side: int, padding: tuple[int, int, int, int], original_size: tuple[int, int]
) -> list[float]:
    """Map MedGemma [y0,x0,y1,x1] in [0,1000] on the padded square back to original (x0,y0,x1,y1) pixels."""
    y0, x0, y1, x1 = (float(v) for v in box_2d)
    left, top, _, _ = padding
    w, h = original_size
    scale = padded_side / 1000.0
    xs = [min(max(x0 * scale - left, 0.0), w), min(max(x1 * scale - left, 0.0), w)]
    ys = [min(max(y0 * scale - top, 0.0), h), min(max(y1 * scale - top, 0.0), h)]
    return [xs[0], ys[0], xs[1], ys[1]]


def parse_localization(
    response: str, label: str, padded_side: int, padding: tuple[int, int, int, int], original_size: tuple[int, int]
) -> dict[str, Any]:
    entry: dict[str, Any] = {"query": label, "parse_ok": False, "boxes": []}
    try:
        payload = extract_json(response)
    except ValueError as error:
        entry["error"] = str(error)
        return entry
    items = payload if isinstance(payload, list) else [payload]
    for item in items:
        box = item.get("box_2d") if isinstance(item, dict) else None
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
            continue
        y0, x0, y1, x1 = box
        valid = 0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000
        entry["boxes"].append(
            {
                "label": item.get("label", label),
                "box_2d_norm1000_yxyx": box,
                "box_xyxy_px_original": normalized_box_to_pixels(box, padded_side, padding, original_size)
                if valid
                else None,
                "geometrically_valid": valid,
            }
        )
    entry["parse_ok"] = any(b["geometrically_valid"] for b in entry["boxes"])
    return entry


BOX_TO_MASK = {
    "right lung": "Right Lung",
    "left lung": "Left Lung",
    "cardiac silhouette": "Heart",
    "mediastinum": "Mediastinum",
    "trachea": "Weasand",
    "right clavicle": "Right Clavicle",
    "left clavicle": "Left Clavicle",
    "spine": "Spine",
    "right hemidiaphragm": None,  # ChestX-Det has a single diaphragm class
    "left hemidiaphragm": None,
}


def box_iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare_boxes_with_segmentation(extraction: dict[str, Any], measurements_px: dict[str, Any]) -> list[dict[str, Any]]:
    """IoU between MedGemma boxes and the bounding boxes of PSPNet masks.

    Both are model outputs, so agreement is a cross-model consistency signal,
    not accuracy against ground truth.
    """
    rows = []
    for item in extraction.get("localization", {}).get("items", []):
        mask_name = BOX_TO_MASK.get(item["query"])
        mask = measurements_px.get(mask_name, {}) if mask_name else {}
        boxes = [b for b in item["boxes"] if b["geometrically_valid"]]
        if not boxes or not mask.get("present"):
            rows.append({"structure": item["query"], "iou": None, "reason": "no valid box" if not boxes else "no matching mask"})
            continue
        rows.append({"structure": item["query"], "iou": max(box_iou(b["box_xyxy_px_original"], mask["bbox_xyxy_px"]) for b in boxes)})
    return rows


def empty_extraction() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "image_metadata": {"view": "UNKNOWN", "confidence": None, "evidence": [], "orientation_notes": []},
        "anatomical_structures": {name: {} for name in STRUCTURES},
        "anatomical_landmarks": [],
        "spatial_relationships": [],
        "visible_findings": [],
        "image_quality": {},
        "localization": {
            "source": "medgemma-1.5 bounding boxes (Chest ImaGenome-trained)",
            "note": "Approximate boxes only. Not segmentation masks. Not measurements.",
            "items": [],
        },
        "segmentation": None,
        "uncertainties": [],
        "limitations": [
            "VLM output is a research annotation, not a radiology report or diagnosis.",
            "VLM self-reported confidence is uncalibrated.",
            "Semantic descriptions are not spatial measurements; boxes are approximate.",
        ],
        "rejected_spatial_claims": [],
        "provenance": {},
    }


# ---------------------------------------------------------------------------
# Model client (GPU; lazy imports)
# ---------------------------------------------------------------------------

@dataclass
class GenerationRecord:
    prompt_sha: str
    response: str


class MedGemmaClient:
    """Thin deterministic wrapper around ``google/medgemma-1.5-4b-it``.

    Requires a GPU runtime and a Hugging Face token whose account has accepted
    the Health AI Developer Foundations terms.
    """

    def __init__(self, model_id: str = "google/medgemma-1.5-4b-it", dtype: str = "bfloat16", max_new_tokens: int = 1500):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("MedGemma requires a CUDA GPU runtime; CPU inference is intentionally not supported.")
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.torch = torch
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model.eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def generate(self, image: Image.Image, prompt: str) -> GenerationRecord:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=self.model.dtype)
        input_len = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.processor.decode(generation[0][input_len:], skip_special_tokens=True)
        return GenerationRecord(prompt_hash(prompt), text)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_extraction(
    image: Image.Image,
    generate: Callable[[Image.Image, str], GenerationRecord],
    is_frontal: bool,
    model_id: str = "google/medgemma-1.5-4b-it",
    model_revision: str | None = None,
    localization_targets: tuple[str, ...] = FRONTAL_LOCALIZATION_TARGETS,
) -> dict[str, Any]:
    """Run view, semantic and (frontal only) localization prompts and assemble the schema.

    ``generate`` is injected so the assembly logic is testable without a GPU.
    """
    result = empty_extraction()
    raw: dict[str, str] = {}
    rgb = image.convert("RGB")

    view_record = generate(rgb, VIEW_PROMPT)
    raw["view"] = view_record.response
    try:
        view_payload = extract_json(view_record.response)
        view = str(view_payload.get("view", "UNKNOWN")).upper()
        result["image_metadata"] = {
            "view": view if view in {"PA", "AP", "LATERAL"} else "UNKNOWN",
            "confidence": view_payload.get("confidence"),
            "confidence_calibrated": False,
            "evidence": view_payload.get("evidence", []),
            "orientation_notes": view_payload.get("orientation_notes", []),
        }
    except (ValueError, AttributeError) as error:
        result["uncertainties"].append(f"View answer unparseable: {error}")

    sem_prompt = semantic_prompt()
    semantic_record = generate(rgb, sem_prompt)
    raw["semantic"] = semantic_record.response
    try:
        payload = extract_json(semantic_record.response)
        if not isinstance(payload, dict):
            raise ValueError("Semantic answer is not a JSON object.")
        clean, rejected = sanitize_semantic(payload)
        result["rejected_spatial_claims"] = rejected
        for key in ("anatomical_landmarks", "spatial_relationships", "visible_findings", "image_quality"):
            if key in clean:
                result[key] = clean[key]
        structures = clean.get("anatomical_structures", {})
        for name in STRUCTURES:
            if isinstance(structures.get(name), dict):
                result["anatomical_structures"][name] = structures[name]
        result["uncertainties"] += list(clean.get("uncertainties", []))
        result["limitations"] += list(clean.get("limitations", []))
    except ValueError as error:
        result["uncertainties"].append(f"Semantic answer unparseable: {error}")

    if is_frontal:
        padded, padding = pad_to_square(rgb)
        for target in localization_targets:
            loc_prompt = LOCALIZATION_PROMPT.format(object_name=target)
            record = generate(padded, loc_prompt)
            raw[f"loc:{target}"] = record.response
            result["localization"]["items"].append(
                parse_localization(record.response, target, padded.size[0], padding, rgb.size)
            )
    else:
        result["localization"]["skipped_reason"] = (
            "MedGemma 1.5 anatomy boxes were trained on frontal Chest ImaGenome images; not requested for lateral input."
        )

    result["provenance"] = {
        "model_id": model_id,
        "model_revision": model_revision,
        "decoding": "greedy (do_sample=False)",
        "prompt_sha": {
            "view": prompt_hash(VIEW_PROMPT),
            "semantic": prompt_hash(sem_prompt),
            "localization_template": prompt_hash(LOCALIZATION_PROMPT),
        },
        "created_utc": utc_now(),
    }
    result["raw_responses"] = raw
    return result
