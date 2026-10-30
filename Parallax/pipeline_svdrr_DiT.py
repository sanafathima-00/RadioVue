"""Compatibility import for serialized Parallax checkpoints.

Published checkpoints reference ``pipeline_svdrr_DiT.CCProjection``.  The
implementation now lives in ``pipeline.py``; this module preserves the stored
import path without duplicating code.
"""

from pipeline import CCProjection

__all__ = ["CCProjection"]
