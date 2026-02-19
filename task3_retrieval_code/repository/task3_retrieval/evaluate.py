import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from task3_retrieval.data import load_split, get_item
from task3_retrieval.embedder import EmbedConfig, CLIPEmbedder
from task3_retrieval.index_io import load_index


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index_dir", type=str, default="task3_retrieval/index_store")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out_dir", type=str, default="reports/task3")
    p.add_argument("--max_queries", type=int, default=0)
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

    labels = np.asarray(meta["labels"]).astype(int)
    ds_indices = np.asarray(meta["indices"]).astype(int)

    n = len(ds_indices)
    qn = n if args.max_queries == 0 else min(n, args.max_queries)

    precisions = []
    per_label = {0: [], 1: []}

    for j in tqdm(range(qn), desc=f"Precision@{args.k}"):
        ds_idx = int(ds_indices[j])
        item = get_item(ds, ds_idx)
        q_label = item.label

        q_vec = embedder.encode_image(item.image).astype(np.float32)[None, :]
        _, I = index.search(q_vec, args.k)

        retrieved_labels = [int(labels[idx_in_index]) for idx_in_index in I[0].tolist()]
        prec = float(np.mean([1 if lab == q_label else 0 for lab in retrieved_labels]))

        precisions.append(prec)
        per_label[q_label].append(prec)

    result = {
        "k": args.k,
        "num_queries": int(qn),
        "mean_precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "std_precision_at_k": float(np.std(precisions)) if precisions else 0.0,
        "per_class_mean": {
            "normal(0)": float(np.mean(per_label[0])) if per_label[0] else 0.0,
            "pneumonia(1)": float(np.mean(per_label[1])) if per_label[1] else 0.0,
        },
        "model_id": meta["model_id"],
        "split_indexed": meta["split"],
        "metric": meta.get("metric", "IP"),
        "normalized": bool(meta.get("normalized", True)),
    }

    out_path = os.path.join(args.out_dir, f"precision_at_{args.k}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
