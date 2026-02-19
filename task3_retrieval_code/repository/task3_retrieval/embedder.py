import torch
import numpy as np
from dataclasses import dataclass
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


@dataclass
class EmbedConfig:
    model_id: str = "openai/clip-vit-base-patch32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    image_size: int = 224
    normalize: bool = True  # L2 normalize


class CLIPEmbedder:
    """Image + text embedding wrapper for retrieval."""

    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        self.processor = CLIPProcessor.from_pretrained(cfg.model_id)
        self.model = CLIPModel.from_pretrained(cfg.model_id)
        self.model.to(cfg.device)
        self.model.eval()

    def _l2(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    @torch.inference_mode()
    def encode_image(self, pil_img: Image.Image) -> np.ndarray:
        img = pil_img.convert("RGB").resize((self.cfg.image_size, self.cfg.image_size))
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
        feats = self.model.get_image_features(**inputs)
        if self.cfg.normalize:
            feats = self._l2(feats)
        return feats.detach().cpu().numpy().astype(np.float32)[0]

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
        feats = self.model.get_text_features(**inputs)
        if self.cfg.normalize:
            feats = self._l2(feats)
        return feats.detach().cpu().numpy().astype(np.float32)[0]
