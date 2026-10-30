"""Colab environment setup used by the notebooks in ``notebooks/``.

Every step prints what it did and fails loudly.  Nothing here is needed
for local CPU tests.  Package pins reproduce the combination that already
ran in the Volumea Colab notebook (Torch 2.11 + T4).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

PARALLAX_PINS = [
    "diffusers==0.30.3", "transformers==4.46.2", "accelerate==1.1.1",
    "huggingface_hub==0.26.2", "safetensors==0.4.5", "kornia==0.7.1",
]
DVG_PACKAGES = [
    "hydra-core>=1.3,<1.4", "omegaconf>=2.3,<2.4", "monai>=1.4,<1.5", "pytorch-lightning>=2.4,<2.6",
    "einops>=0.8,<0.9", "einops-exts>=0.0.4", "rotary-embedding-torch>=0.8", "ema-pytorch>=0.7",
    "SimpleITK>=2.3", "scikit-video>=1.1.11", "tensorboardX>=2.6", "nibabel>=5.2", "ninja>=1.11", "h5py",
    "wandb",  # imported at module level by Volumea/ddpm/diffusion.py
]
TOOL_PACKAGES = ["torchxrayvision>=1.3", "scikit-image", "pydicom>=3.0,<4", "lpips", "diffdrr==0.4.8", "ipywidgets"]
MEDGEMMA_VENV_PACKAGES = ["transformers>=4.50", "accelerate>=1.1"]

# Sizes and SHA-256 recorded in reports/voluma_baseline/BASELINE_REPORT.md.
DVG_CHECKPOINTS = {
    "CT_VQGAN.ckpt": (242612651, "e6f026d5dfb2f2f4134a3c7935cd671f0d50df3481a42b4ecbd2baf061bbe786"),
    "VPGE_0_90_VQGAN.ckpt": (242612523, "11f9428a60dc37d49af3e3c4879391d13118e8c1900c6bf538c4a202258e7520"),
    "VPGE_45_VQGAN.ckpt": (242612587, "37ec6a80db07cb239dda35fd755a22edd409ee23db9a39298e40a467d007a959"),
    "NewView_Coarse_Model.pt": (887178973, "0a23e72582f54cfc693bd57adcc6a1cc25246dede0d9eb55af1ad8f291b5d5fd"),
    "DVG_Diffusion.pt": (2205253707, "1332871160ecb2928f36e71546cf4e827ac54e9651f579f405d7cdf6ba58c283"),
}


def pip(*packages: str, python: str | None = None, extra: list[str] | None = None) -> None:
    cmd = [python or sys.executable, "-m", "pip", "install", "-q", *(extra or []), *packages]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def gpu_report() -> dict:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("No NVIDIA GPU visible. Runtime > Change runtime type > GPU (L4 or A100 recommended; T4 minimum).")
    name, mem, driver = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
    report = {"gpu": name, "memory": mem, "driver": driver}
    print(report)
    return report


def torchvision_imports_cleanly() -> bool:
    return subprocess.run([sys.executable, "-c", "import torchvision"], capture_output=True).returncode == 0


def install_runtime(parallax: bool = True, dvg: bool = True, tools: bool = True) -> None:
    """Install into the main kernel environment.  Run before importing torch/numpy in the kernel."""
    if parallax:
        pip(*PARALLAX_PINS)
    if dvg:
        pip(*DVG_PACKAGES)
    if tools:
        pip(*TOOL_PACKAGES)
    if not torchvision_imports_cleanly():
        # Observed in the Volumea Colab: a NumPy upgrade broke torchvision; this pin restored it.
        pip("numpy==2.2.6", extra=["--force-reinstall", "--no-cache-dir"])


def make_medgemma_venv(path: str = "/content/medgemma_venv") -> str:
    """Isolated interpreter that reuses Colab's torch but has transformers>=4.50 for Gemma 3."""
    python = str(Path(path) / "bin" / "python")
    if not Path(python).exists():
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", path], check=True)
    pip(*MEDGEMMA_VENV_PACKAGES, python=python, extra=["-U"])
    check = subprocess.run([python, "-c", "import transformers, torch; print(transformers.__version__, torch.__version__)"],
                           capture_output=True, text=True, check=True)
    print("MedGemma venv:", check.stdout.strip())
    return python


def hf_token_from_colab_secrets(name: str = "HF_TOKEN") -> str | None:
    try:
        from google.colab import userdata

        token = userdata.get(name)
    except Exception:
        token = os.environ.get(name)
    if token:
        os.environ["HF_TOKEN"] = token  # never printed
    return token


def check_gated_access(repo_id: str) -> None:
    """Fail with an actionable message if the HF account has not accepted the HAI-DEF terms."""
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(repo_id, "config.json", token=os.environ.get("HF_TOKEN"))
        print(f"Hugging Face access OK: {repo_id}")
    except Exception as error:
        raise PermissionError(
            f"Cannot access gated model {repo_id}. 1) Log in at huggingface.co and accept the Health AI Developer "
            f"Foundations terms on https://huggingface.co/{repo_id}  2) Create a read token  3) Add it to Colab "
            f"Secrets as HF_TOKEN with notebook access enabled. Underlying error: {type(error).__name__}"
        ) from None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_dvg_checkpoints(drive_dir: str, local_dir: str = "/content/dvg_checkpoints", verify_hash: bool = False) -> str:
    """Copy the five DVG checkpoints from Drive to local disk and verify them against the recorded fingerprints."""
    src, dst = Path(drive_dir), Path(local_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in DVG_CHECKPOINTS.items():
        source = src / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing DVG checkpoint on Drive: {source}")
        target = dst / name
        if not (target.is_file() and target.stat().st_size == size):
            shutil.copyfile(source, target)
        if target.stat().st_size != size:
            raise IOError(f"{name}: size {target.stat().st_size} != recorded {size}")
        if verify_hash and sha256(target) != digest:
            raise IOError(f"{name}: SHA-256 mismatch with reports/voluma_baseline/BASELINE_REPORT.md")
        print(f"DVG checkpoint OK: {name}")
    return str(dst)


def stage_parallax_model(drive_dir: str | None, local_dir: str = "/content/parallax_model/DiT-fb-512",
                         repo_id: str = "xiechun-tsukuba/svdrr-dit-fb-512", shared_repo: str = "xiechun-tsukuba/svdrr-dit-fb-256") -> str:
    """Use the Drive copy if present, else assemble it from Hugging Face like Parallax/scripts/download_models.py."""
    dst = Path(local_dir)
    if (dst / "model_index.json").is_file():
        return str(dst)
    if drive_dir and (Path(drive_dir) / "model_index.json").is_file():
        shutil.copytree(drive_dir, dst, dirs_exist_ok=True, symlinks=False)
        print("Parallax model copied from Drive:", drive_dir)
        return str(dst)
    from huggingface_hub import snapshot_download

    dst.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id, local_dir=str(dst), allow_patterns=["transformer/**", "cc_projection/**", "model_index.json"])
    snapshot_download(shared_repo, local_dir=str(dst), allow_patterns=["vae/**", "scheduler/**", "image_encoder/**"])
    print("Parallax model assembled from Hugging Face:", repo_id, "+ shared components from", shared_repo)
    return str(dst)


def build_drr_projector(volumea_root: str) -> None:
    """Compile XraySyn's CUDA projector (as in the Volumea Colab) and make it importable."""
    projector = Path(volumea_root) / "xraysyn" / "networks" / "drr_projector"
    try:
        preload_torch_libs()
        import drr_projector_function  # noqa: F401

        print("DRR projector already importable.")
        return
    except ImportError:
        pass
    env = {**os.environ, "MAX_JOBS": "2"}
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", str(projector)], check=True, env=env)
    preload_torch_libs()
    import drr_projector_function  # noqa: F401

    print("DRR projector built and imported.")


def preload_torch_libs() -> None:
    import torch

    lib = Path(torch.__file__).parent / "lib"
    for name in ("libc10.so", "libtorch_cpu.so", "libtorch_cuda.so", "libtorch_python.so"):
        if (lib / name).exists():
            ctypes.CDLL(str(lib / name), mode=ctypes.RTLD_GLOBAL)


def ensure_parallax_shims(parallax_root: str) -> None:
    """The published model_index.json imports ``pipeline_svdrr_DiT`` and ``svdrr_transformer_2d``."""
    root = Path(parallax_root)
    shims = {
        # Same content the Volumea Colab needed to load the published checkpoint.
        "pipeline_svdrr_DiT.py": (
            "from diffusers.models.modeling_utils import ModelMixin\n"
            "from diffusers.configuration_utils import ConfigMixin\n"
            "from pipeline import CCProjection\n"
        ),
        "svdrr_transformer_2d.py": "from transformer import SvdrrTransformer2DModel\n__all__ = ['SvdrrTransformer2DModel']\n",
    }
    for name, body in shims.items():
        if not (root / name).exists():
            (root / name).write_text(body)
            print("Created Parallax import shim:", name)
