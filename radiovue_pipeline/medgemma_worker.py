"""Run MedGemma extraction in an isolated interpreter.

MedGemma 1.5 (Gemma 3) needs ``transformers>=4.50``.  The Parallax/SV-DRR
stack that already works in Colab pins ``transformers==4.46.2``.  Running
MedGemma in a separate process (normally a ``--system-site-packages`` venv
that only overrides ``transformers``/``accelerate``) avoids version conflicts.
GPU memory is also released when the process exits.

CLI:  python -m radiovue_pipeline.medgemma_worker --image in.png --out out.json [--frontal]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frontal", action="store_true")
    parser.add_argument("--model-id", default="google/medgemma-1.5-4b-it")
    parser.add_argument("--max-new-tokens", type=int, default=1500)
    args = parser.parse_args()

    from .anatomy_extraction import MedGemmaClient, run_extraction

    client = MedGemmaClient(args.model_id, max_new_tokens=args.max_new_tokens)
    image = Image.open(args.image).convert("L")
    result = run_extraction(image, client.generate, is_frontal=args.frontal, model_id=args.model_id, model_revision=client.revision)
    import torch
    import transformers

    result["provenance"]["software"] = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


class SubprocessMedGemma:
    """``extract(image, is_frontal) -> dict`` that runs the worker in another interpreter."""

    def __init__(self, python_executable: str, repo_root: str | Path, model_id: str = "google/medgemma-1.5-4b-it", timeout_s: int = 3600):
        self.python = python_executable
        self.repo_root = str(Path(repo_root).resolve())
        self.model_id = model_id
        self.timeout_s = timeout_s

    def extract(self, image: Image.Image, is_frontal: bool) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmp:
            img_path, out_path = Path(tmp) / "input.png", Path(tmp) / "extraction.json"
            image.convert("L").save(img_path)
            cmd = [self.python, "-m", "radiovue_pipeline.medgemma_worker", "--image", str(img_path), "--out", str(out_path),
                   "--model-id", self.model_id] + (["--frontal"] if is_frontal else [])
            env = {**os.environ, "PYTHONPATH": self.repo_root + os.pathsep + os.environ.get("PYTHONPATH", "")}
            proc = subprocess.run(cmd, cwd=self.repo_root, env=env, capture_output=True, text=True, timeout=self.timeout_s)
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout)[-2000:]
                raise RuntimeError(f"MedGemma worker failed (exit {proc.returncode}): {tail}")
            return json.loads(out_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    sys.exit(main())
