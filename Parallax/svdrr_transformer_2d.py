"""Compatibility import for serialized Parallax checkpoints.

The published DiT-fb checkpoints store ``svdrr_transformer_2d`` in their
``model_index.json``.  Keep the implementation in ``transformer.py`` while
preserving that import path so existing weights can be loaded unchanged.
"""

from transformer import SvdrrTransformer2DModel

__all__ = ["SvdrrTransformer2DModel"]
