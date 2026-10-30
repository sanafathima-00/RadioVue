"""Explicit export geometry.  Aspect padding is intentionally experimental."""

from __future__ import annotations

from PIL import Image


def export_geometry(image: Image.Image, target_size: int, strategy: str) -> tuple[Image.Image, dict[str, object]]:
    """Export grayscale image with a fully recorded resize strategy."""
    source_size = image.size
    if strategy == "fixed_square":
        # PIL's stock SV-DRR path calls resize() without a resampling argument.
        output = image.resize((target_size, target_size), Image.Resampling.NEAREST)
        return output, {
            "strategy": strategy,
            "source_size": list(source_size),
            "output_size": [target_size, target_size],
            "resample": "nearest_matches_stock_pillow_default",
            "cropped": False,
            "padded": False,
        }
    if strategy == "aspect_pad":
        scale = min(target_size / source_size[0], target_size / source_size[1])
        resized_size = (max(1, round(source_size[0] * scale)), max(1, round(source_size[1] * scale)))
        resized = image.resize(resized_size, Image.Resampling.NEAREST)
        output = Image.new("L", (target_size, target_size), color=0)
        offset = ((target_size - resized_size[0]) // 2, (target_size - resized_size[1]) // 2)
        output.paste(resized, offset)
        return output, {
            "strategy": strategy,
            "source_size": list(source_size),
            "output_size": [target_size, target_size],
            "resample": "nearest",
            "cropped": False,
            "padded": True,
            "padding_offset": list(offset),
        }
    raise ValueError(f"Unsupported geometry strategy '{strategy}'.")
