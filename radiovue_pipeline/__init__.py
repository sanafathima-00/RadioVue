"""RadioVue single-X-ray research pipeline.

One chest radiograph -> view identification -> anatomical extraction ->
complementary synthetic view (Parallax / SV-DRR) -> Volumea (DVG-Diffusion).

Research software only.  Every generated image is synthetic and is labelled as
such; nothing produced here is clinical ground truth.  GPU-dependent modules
import their heavy dependencies lazily so the orchestration, provenance, and
validation layers can be tested on a CPU-only machine.
"""

from .config import PipelineConfig
from .pipeline import run_pipeline

__all__ = ["PipelineConfig", "run_pipeline"]
__version__ = "0.1.0"
