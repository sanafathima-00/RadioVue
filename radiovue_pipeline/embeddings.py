"""Image-embedding extractors compared in notebook 02.

* MedSigLIP-448 (gated, HAI-DEF terms): medical image-text embedding + zero-shot.
* Parallax's own CLIP image encoder: the embedding SV-DRR actually conditions on.
* TorchXRayVision DenseNet-121 ("densenet121-res224-all"): a conventional CXR CNN feature baseline.

Swapping the conditioning encoder of a pretrained SV-DRR model needs retraining
(``cc_projection`` and the transformer's cross-attention were trained on CLIP
features).  These embeddings are therefore compared as *analysis signals*, not
as drop-in conditioning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

ZERO_SHOT_VIEW_PROMPTS = {
    "PA": "a posteroanterior (PA) frontal chest x-ray",
    "AP": "an anteroposterior (AP) portable frontal chest x-ray",
    "LATERAL": "a lateral chest x-ray",
}


def cosine_matrix(embeddings: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    names = list(embeddings)
    unit = {k: v / (np.linalg.norm(v) + 1e-12) for k, v in embeddings.items()}
    return {a: {b: float(unit[a] @ unit[b]) for b in names} for a in names}


class MedSigLIPEmbedder:
    def __init__(self, model_id: str = "google/medsiglip-448", device: str = "cuda"):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch, self.device = torch, device
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def embed(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=[image.convert("RGB").resize((448, 448))], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            return self.model.get_image_features(**inputs)[0].float().cpu().numpy()

    def zero_shot(self, image: Image.Image, prompts: dict[str, str] = ZERO_SHOT_VIEW_PROMPTS) -> dict[str, float]:
        texts = list(prompts.values())
        inputs = self.processor(text=texts, images=[image.convert("RGB").resize((448, 448))],
                                padding="max_length", return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(**inputs).logits_per_image, dim=1)[0].float().cpu().numpy()
        return dict(zip(prompts.keys(), probs.tolist()))


class ParallaxClipEmbedder:
    """CLIP image encoder from the Parallax model directory (``image_encoder/``)."""

    def __init__(self, model_dir: str, device: str = "cuda"):
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        self.torch, self.device = torch, device
        self.model = CLIPVisionModelWithProjection.from_pretrained(f"{model_dir}/image_encoder").to(device).eval()
        self.processor = CLIPImageProcessor()

    def embed(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            return self.model(**inputs).image_embeds[0].float().cpu().numpy()


class XrvDenseNetEmbedder:
    def __init__(self, weights: str = "densenet121-res224-all", device: str = "cpu"):
        import torch
        import torchxrayvision as xrv

        self.torch, self.xrv, self.device = torch, xrv, device
        self.model = xrv.models.DenseNet(weights=weights).to(device).eval()

    def embed(self, image: Image.Image) -> np.ndarray:
        from .io_utils import pad_to_square

        square, _ = pad_to_square(image.convert("L"))
        arr = self.xrv.datasets.normalize(np.asarray(square.resize((224, 224), Image.BILINEAR), dtype=np.float32), 255)
        tensor = self.torch.from_numpy(arr)[None, None].to(self.device)
        with self.torch.no_grad():
            feats = self.model.features(tensor)[0].float().cpu().numpy()
        return feats.reshape(feats.shape[0], -1).mean(axis=1) if feats.ndim > 1 else feats  # global average pool


def summarize(name: str, vectors: dict[str, np.ndarray]) -> dict[str, Any]:
    first = next(iter(vectors.values()))
    return {"model": name, "dim": int(first.shape[-1]), "cosine": cosine_matrix(vectors)}
