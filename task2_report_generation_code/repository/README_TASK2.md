# Postdoc Challenge – Task 2 (Medical Report Generation)

Task 2 builds a pipeline that takes a PneumoniaMNIST chest X-ray image and generates a short, clinically-oriented description using an open-source Visual Language Model (VLM), as required by the challenge PDF.

## Install

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Run (generate 10 reports)

```bash
python -m task2_report_generation.generate_reports \
  --model_id <HF_MODEL_ID> \
  --split test \
  --num_images 10 \
  --out_dir reports/task2
```

## Optional: include Task-1 misclassified images

If a Task-1 checkpoint exists (example: `models/best.pt`):

```bash
python -m task2_report_generation.generate_reports \
  --model_id <HF_MODEL_ID> \
  --task1_checkpoint models/best.pt \
  --include_misclassified 1 \
  --out_dir reports/task2
```

## Outputs

- `reports/task2/reports_task2.md` (markdown with images)
- `reports/task2/reports_task2.jsonl` (structured outputs)
- `reports/task2/prompting_strategies.md` (prompt variants)
