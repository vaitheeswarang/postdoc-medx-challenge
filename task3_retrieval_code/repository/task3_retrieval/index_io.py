import os
import json
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import faiss


@dataclass
class IndexPaths:
    index_dir: str

    @property
    def faiss_path(self) -> str:
        return os.path.join(self.index_dir, "faiss.index")

    @property
    def meta_path(self) -> str:
        return os.path.join(self.index_dir, "meta.json")

    @property
    def emb_path(self) -> str:
        return os.path.join(self.index_dir, "embeddings.npy")


def save_index(index_dir: str, index: faiss.Index, embeddings: np.ndarray, meta: Dict[str, Any]) -> None:
    os.makedirs(index_dir, exist_ok=True)
    p = IndexPaths(index_dir)
    faiss.write_index(index, p.faiss_path)
    np.save(p.emb_path, embeddings.astype(np.float32))
    with open(p.meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_index(index_dir: str):
    p = IndexPaths(index_dir)
    if not os.path.exists(p.faiss_path) or not os.path.exists(p.meta_path):
        raise FileNotFoundError("Index not found. Run build_index first.")
    index = faiss.read_index(p.faiss_path)
    with open(p.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta
