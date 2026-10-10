"""DINOv2 image embeddings (lazy torch/transformers import).

Used for the hybrid stock-level signal: embedding distance of a crop to the
per-area empty/full reference crops, plus features for the small classifier.
torch/transformers live in requirements-gpu.txt so the core pipeline works
without them.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def image_key(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image)).hexdigest()


class Dinov2Encoder:
    """CLS-token embeddings, L2-normalized, with an on-disk cache."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        device: str | None = None,
        cache_dir: str = ".cache/dinov2",
        batch_size: int = 16,
    ):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.to(self.device)
        self._model.eval()

    def _encode_batch(self, images: list[np.ndarray]) -> np.ndarray:
        import numpy as np  # noqa: F401
        from PIL import Image

        rgb = [
            Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))
            if img.ndim == 3 and img.shape[2] == 3
            else Image.fromarray(img).convert("RGB")
            for img in images
        ]
        inputs = self._processor(
            images=rgb, return_tensors="pt"
        ).to(self.device)
        with self.torch.no_grad():
            out = self._model(**inputs)
        feats = out.last_hidden_state[:, 0, :].float().cpu().numpy()
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.maximum(norms, 1e-12)

    def encode(self, images: list[np.ndarray], use_cache: bool = True) -> np.ndarray:
        results: list[np.ndarray | None] = [None] * len(images)
        pending: list[int] = []
        if use_cache:
            for i, img in enumerate(images):
                path = self._cache_dir / (image_key(img) + ".npz")
                if path.exists():
                    results[i] = np.load(path)["features"]
                else:
                    pending.append(i)
        else:
            pending = list(range(len(images)))

        for start in range(0, len(pending), self.batch_size):
            idxs = pending[start : start + self.batch_size]
            batch_imgs = [images[i] for i in idxs]
            batch_feats = self._encode_batch(batch_imgs)
            for i, feat in zip(idxs, batch_feats):
                results[i] = feat
                if use_cache:
                    path = self._cache_dir / (image_key(images[i]) + ".npz")
                    np.savez_compressed(path, features=feat)
        return np.vstack([r for r in results if r is not None])

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - np.dot(a, b))
