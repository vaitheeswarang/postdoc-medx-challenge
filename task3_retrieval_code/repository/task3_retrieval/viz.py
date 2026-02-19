import os
from typing import List, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def save_retrieval_grid(
    query_img: Image.Image,
    retrieved: List[Tuple[Image.Image, int, float, int]],
    out_path: str,
    title: str = "Retrieval Results",
):
    """Save query + top-k grid to PNG."""
    k = len(retrieved)
    cols = min(6, k + 1)
    rows = 1 if (k + 1) <= cols else int(np.ceil((k + 1) / cols))

    fig = plt.figure(figsize=(2.2 * cols, 2.4 * rows))
    fig.suptitle(title)

    ax = fig.add_subplot(rows, cols, 1)
    ax.imshow(query_img.convert("RGB"))
    ax.set_title("Query")
    ax.axis("off")

    for i, (img, label, dist, idx) in enumerate(retrieved, start=2):
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(img.convert("RGB"))
        ax.set_title(f"idx={idx}\nlabel={label}\nd={dist:.3f}")
        ax.axis("off")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
