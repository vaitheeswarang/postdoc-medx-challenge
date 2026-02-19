from __future__ import annotations

import numpy as np
from PIL import Image


def tensor_to_pil_rgb(t) -> Image.Image:
    arr = t.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    a_min, a_max = float(arr.min()), float(arr.max())
    if a_max - a_min < 1e-8:
        arr = np.zeros_like(arr)
    else:
        arr = (arr - a_min) / (a_max - a_min)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")
