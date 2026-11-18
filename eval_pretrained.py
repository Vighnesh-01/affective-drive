"""
Affect-Drive — Pre-trained Model Evaluation
Evaluates the saved bigru_affective.pth weights on the full dataset
using both train.py's AffectiveDataset (sliding-window) approach.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), ".mpl_cache")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from src.model.bigru_model import AffectiveBiGRU
from src.model.train import AffectiveDataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
LABEL_NAMES = ["NORMAL", "ANXIOUS", "DISTRACTED"]
DEVICE = torch.device("cpu")

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "figure.dpi": 300, "pdf.fonttype": 42,
})

print("=" * 65)
print("  AFFECT-DRIVE — PRE-TRAINED MODEL EVALUATION")
print("=" * 65)

# Load pre-trained model
model = AffectiveBiGRU(input_size=41, hidden_size=64, num_classes=3).to(DEVICE)
model.load_state_dict(torch.load("data/bigru_affective.pth", map_location=DEVICE))
model.eval()
print(f"  Loaded pre-trained weights from data/bigru_affective.pth")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Load dataset (sliding window approach from train.py)
dataset = AffectiveDataset("data/phase2_dataset.csv", seq_len=30)
loader = DataLoader(dataset, batch_size=64, shuffle=False)
print(f"  Dataset: {len(dataset)} sequences (sliding window, seq_len=30)")

# Full-dataset evaluation
all_preds, all_trues, all_probs = [], [], []

with torch.inference_mode():
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        logits = model(X_batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_trues.extend(y_batch.numpy())
        all_probs.extend(probs)

all_preds = np.array(all_preds)
all_trues = np.array(all_trues)
all_probs = np.array(all_probs)

# Metrics
overall_acc = accuracy_score(all_trues, all_preds) * 100
macro_f1 = f1_score(all_trues, all_preds, average="macro") * 100
weighted_f1 = f1_score(all_trues, all_preds, average="weighted") * 100

y_bin = label_binarize(all_trues, classes=[0, 1, 2])
auroc_per = {}
for i, name in enumerate(LABEL_NAMES):
    try:
        auroc_per[name] = roc_auc_score(y_bin[:, i], all_probs[:, i])
    except ValueError:
        auroc_per[name] = float("nan")
macro_auroc = np.nanmean(list(auroc_per.values()))

print(f"\n  ═══ PRE-TRAINED MODEL RESULTS (Full Dataset) ═══")
print(f"  Accuracy           : {overall_acc:.2f}%")
print(f"  Macro-F1           : {macro_f1:.2f}%")
print(f"  Weighted-F1        : {weighted_f1:.2f}%")
print(f"  Macro AUROC        : {macro_auroc:.4f}")
for name, v in auroc_per.items():
    print(f"    AUROC ({name:>11s}): {v:.4f}")

# Classification report
print(f"\n  Per-class report:")
report = classification_report(all_trues, all_preds, target_names=LABEL_NAMES, output_dict=True)
print(f"  {'Class':<14s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
print("  " + "-" * 55)
for cls in LABEL_NAMES:
    r = report[cls]
    print(f"  {cls:<14s} {r['precision']*100:>9.2f}% {r['recall']*100:>9.2f}% {r['f1-score']*100:>9.2f}% {r['support']:>10.0f}")

# Save CSV
summary_rows = [
    {"Metric": "Accuracy (%)", "Value": round(overall_acc, 2)},
    {"Metric": "Macro-F1 (%)", "Value": round(macro_f1, 2)},
    {"Metric": "Weighted-F1 (%)", "Value": round(weighted_f1, 2)},
    {"Metric": "Macro AUROC", "Value": round(macro_auroc, 4)},
]
for name in LABEL_NAMES:
    summary_rows.append({"Metric": f"AUROC ({name})", "Value": round(auroc_per[name], 4)})
pd.DataFrame(summary_rows).to_csv(os.path.join(RESULTS_DIR, "pretrained_model_metrics.csv"), index=False)

per_class = []
for cls in LABEL_NAMES:
    r = report[cls]
    per_class.append({
        "Class": cls,
        "Precision (%)": round(r["precision"]*100, 2),
        "Recall (%)": round(r["recall"]*100, 2),
        "F1-Score (%)": round(r["f1-score"]*100, 2),
        "Support": int(r["support"]),
        "AUROC": round(auroc_per[cls], 4),
    })
pd.DataFrame(per_class).to_csv(os.path.join(RESULTS_DIR, "pretrained_per_class_metrics.csv"), index=False)

# Confusion matrix
cm = confusion_matrix(all_trues, all_preds)
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set(xticks=np.arange(3), yticks=np.arange(3),
       xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
       ylabel="True Label", xlabel="Predicted Label",
       title="Confusion Matrix — Pre-trained BiGRU Model")
thresh = cm.max() / 2.0
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14,
                fontweight="bold", color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pretrained_confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()

# AUROC curves
fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
colors = ["#2B5C8F", "#2A9D8F", "#E76F51"]
for i, (name, color) in enumerate(zip(LABEL_NAMES, colors)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
    roc_auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc_val:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
ax.set(xlim=[0, 1.01], ylim=[0, 1.01], xlabel="False Positive Rate",
       ylabel="True Positive Rate", title="AUROC Curves — Pre-trained BiGRU Model")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pretrained_auroc_curves.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"\n  [OK] All pre-trained model results saved to results/")
print("=" * 65)
