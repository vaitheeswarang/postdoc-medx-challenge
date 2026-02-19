import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import faiss

from task3_retrieval.data import load_split, get_item
from task3_retrieval.embedder import EmbedConfig, CLIPEmbedder
from task3_retrieval.index_io import save_index


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--index_dir", type=str, default="task3_retrieval/index_store")
    p.add_argument("--out_dir", type=str, default="reports/task3")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--no_normalize", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_split(args.split, download=True)

    cfg = EmbedConfig(
        model_id=args.model_id,
        image_size=args.image_size,
        normalize=(not args.no_normalize),
    )
    embedder = CLIPEmbedder(cfg)

    embs, labels, indices = [], [], []
    for i in tqdm(range(len(ds)), desc=f"Embedding {args.split}"):
        item = get_item(ds, i)
        embs.append(embedder.encode_image(item.image))
        labels.append(item.label)
        indices.append(item.index)

    embs = np.stack(embs).astype(np.float32)
    labels = np.asarray(labels).astype(int)
    indices = np.asarray(indices).astype(int)

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d) if cfg.normalize else faiss.IndexFlatL2(d)
    index.add(embs)

    meta = {
        "model_id": args.model_id,
        "split": args.split,
        "image_size": args.image_size,
        "normalized": cfg.normalize,
        "labels": labels.tolist(),
        "indices": indices.tolist(),
        "embedding_dim": int(d),
        "metric": "IP" if cfg.normalize else "L2",
    }

    save_index(args.index_dir, index, embs, meta)

    summary_path = os.path.join(args.out_dir, "index_build_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "index_dir": args.index_dir,
                "num_items": int(len(ds)),
                "embedding_dim": int(d),
                "model_id": args.model_id,
                "split": args.split,
            },
            f,
            indent=2,
        )

    print(f"Saved index to: {args.index_dir}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
