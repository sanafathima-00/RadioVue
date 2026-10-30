"""Conservative, auditable chest X-ray standardization for SV-DRR.

This package deliberately does not modify SV-DRR itself and does not attempt
to infer projection geometry or alter patient anatomy.
"""

from .pipeline import StandardizationError, standardize
from .profiles import get_profile

__all__ = ["StandardizationError", "get_profile", "standardize"]
