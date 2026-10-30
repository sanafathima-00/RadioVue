"""Compatibility import for serialized Parallax checkpoints.

The published DiT-fb checkpoints store ``svdrr_transformer_2d`` in their
``model_index.json``.  Keep the implementation in ``transformer.py`` while
preserving that import path so existing weights can be loaded unchanged.
diffusers resolves the loading mixins (``ModelMixin``) from this module's
namespace, so they must be importable from here as well.
"""

from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from transformer import SvdrrTransformer2DModel

__all__ = ["SvdrrTransformer2DModel", "ModelMixin", "ConfigMixin"]
