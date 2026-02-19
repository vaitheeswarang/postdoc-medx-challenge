# Task 3 – Semantic Image Retrieval (PneumoniaMNIST)

## Setup
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Build index (embeddings + FAISS)
```bash
python -m task3_retrieval.build_index --model_id openai/clip-vit-base-patch32 --split test --out_dir reports/task3 --index_dir task3_retrieval/index_store
```

## Evaluate Precision@k (image-to-image)
```bash
python -m task3_retrieval.evaluate --index_dir task3_retrieval/index_store --k 5 --out_dir reports/task3
```

## Image-to-image search
Query from dataset by index:
```bash
python -m task3_retrieval.search_image --index_dir task3_retrieval/index_store --query_idx 0 --k 5 --out_dir reports/task3
```

Query from external image path:
```bash
python -m task3_retrieval.search_image --index_dir task3_retrieval/index_store --query_path /path/to/image.png --k 5 --out_dir reports/task3
```

## Text-to-image search
```bash
python -m task3_retrieval.search_text --index_dir task3_retrieval/index_store --query "possible consolidation in right lower lobe" --k 5 --out_dir reports/task3
```

## Outputs
- `task3_retrieval/index_store/` contains FAISS index + metadata
- `reports/task3/` contains metrics JSON and retrieval visualizations
