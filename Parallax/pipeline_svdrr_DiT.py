"""Compatibility import for serialized Parallax checkpoints.

Published checkpoints reference ``pipeline_svdrr_DiT.CCProjection``.  The
implementation now lives in ``pipeline.py``; this module preserves the stored
import path without duplicating code.  diffusers resolves the loading mixins
from this module's namespace, so they are re-exported here.
"""

from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from pipeline import CCProjection

__all__ = ["CCProjection", "ModelMixin", "ConfigMixin"]
