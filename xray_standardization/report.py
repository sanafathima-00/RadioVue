"""De-identified report and comparison-image writers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


def source_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_comparison(before: Image.Image, after: Image.Image, path: str | Path) -> None:
    """Write a label-free side-by-side visual comparison."""
    before = before.convert("L")
    after = after.convert("L")
    height = max(before.height, after.height)
    canvas = Image.new("L", (before.width + after.width, height), color=0)
    canvas.paste(before, (0, 0))
    canvas.paste(after, (before.width, 0))
    canvas.save(path, format="PNG")
