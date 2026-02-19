import os
import argparse
import numpy as np
from PIL import Image

from task3_retrieval.data import load_split, get_item
from task3_retrieval.embedder import EmbedConfig, CLIPEmbedder
from task3_retrieval.index_io import load_index
from task3_retrieval.viz import save_retrieval_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index_dir", type=str, default="task3_retrieval/index_store")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out_dir", type=str, default="reports/task3")
    p.add_argument("--query_idx", type=int, default=None)
    p.add_argument("--query_path", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    index, meta = load_index(args.index_dir)
    cfg = EmbedConfig(
        model_id=meta["model_id"],
        image_size=int(meta.get("image_size", 224)),
        normalize=bool(meta.get("normalized", True)),
    )
    embedder = CLIPEmbedder(cfg)
    ds = load_split(meta["split"], download=True)

    if (args.query_idx is None) == (args.query_path is None):
        raise ValueError("Exactly one of --query_idx or --query_path must be provided.")

    if args.query_idx is not None:
        q_item = get_item(ds, args.query_idx)
        q_img = q_item.image
        title = f"Query idx={args.query_idx} label={q_item.label}"
    else:
        q_img = Image.open(args.query_path)
        title = f"Query external: {os.path.basename(args.query_path)}"

    q_vec = embedder.encode_image(q_img).astype(np.float32)[None, :]
    D, I = index.search(q_vec, args.k)

    ds_indices = np.asarray(meta["indices"]).astype(int)
    labels = np.asarray(meta["labels"]).astype(int)

    retrieved = []
    for dist, idx_in_index in zip(D[0].tolist(), I[0].tolist()):
        ds_idx = int(ds_indices[idx_in_index])
        item = get_item(ds, ds_idx)
        retrieved.append((item.image, item.label, float(dist), int(ds_idx)))

    out_path = os.path.join(args.out_dir, f"retrieval_image_k{args.k}.png")
    save_retrieval_grid(
        query_img=q_img.convert("RGB").resize((224, 224)),
        retrieved=[(im.convert("RGB").resize((224, 224)), lab, dist, ds_i) for (im, lab, dist, ds_i) in retrieved],
        out_path=out_path,
        title=title,
    )
    print(f"Saved visualization: {out_path}")
    for r, (_, lab, dist, ds_i) in enumerate(retrieved, start=1):
        print(f"{r:02d}. idx={ds_i} label={lab} score={dist:.4f}")


if __name__ == "__main__":
    main()
