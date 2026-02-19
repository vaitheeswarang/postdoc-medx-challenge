# PneumoniaMNIST – Task 1 (Classification)

This folder contains a complete **Task 1** implementation for the AlfaisalX postdoctoral technical challenge:
- CNN classifier training (PyTorch)
- Full evaluation (Accuracy / Precision / Recall / F1 / ROC-AUC)
- Confusion matrix + ROC curve
- Training curves
- Failure case visualization (misclassified images)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
python task1_classification/train.py --epochs 20 --batch_size 128 --lr 1e-3 --model simple_cnn
```

Optional: switch model
```bash
python task1_classification/train.py --model resnet18_small
```

## Evaluate (uses best checkpoint saved during training)
```bash
python task1_classification/eval.py --checkpoint models/task1/best.pt
```

Outputs are written to:
- `reports/task1/` (plots, metrics JSON, misclassified grids)

## Notes
- Dataset: `PneumoniaMNIST` from `medmnist` (MedMNIST v2).
- Input images are 28×28 grayscale.
- Augmentations are conservative (small rotations/translations) to preserve clinical plausibility.
