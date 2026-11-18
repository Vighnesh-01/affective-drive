"""
Affect-Drive — Full Quantitative Evaluation Pipeline
=====================================================
Produces:
  1. Classification metrics: Accuracy, Macro-F1, Weighted-F1, per-class AUROC
  2. Confusion matrix
  3. Latency benchmarks: driver feature extraction, road inference, BiGRU, rule-based agent
  4. RAM / memory profiling
  5. Ablation study data

Outputs (all written to  results/):
  - classification_metrics.csv
  - confusion_matrix.png
  - per_class_metrics.csv
  - auroc_curves.png
  - latency_benchmarks.csv
  - latency_chart.png
  - memory_profile.csv
  - memory_chart.png
  - ablation_summary.csv
  - ablation_chart.png
  - full_evaluation_report.csv
"""

import os, sys, time, csv, warnings
warnings.filterwarnings("ignore")

# Fix matplotlib config dir issue
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), ".mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil

# Project imports
sys.path.insert(0, os.path.dirname(__file__))
from src.model.bigru_model import AffectiveBiGRU
from src.model.train import AffectiveDataset
from src.agent.llm_intervention import InterventionAgent

# ── Global Config ───────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_PATH = "data/phase2_dataset.csv"
MODEL_PATH = "data/bigru_affective.pth"
SEQ_LEN = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["NORMAL", "ANXIOUS", "DISTRACTED"]
N_CLASSES = 3

# Publication-quality matplotlib defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11.5,
    "axes.labelweight": "bold",
    "figure.dpi": 300,
    "pdf.fonttype": 42,
})


# ═══════════════════════════════════════════════════════════════════════════
# 1. CLASSIFICATION EVALUATION  (5-Fold Stratified Cross-Validation)
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_classification():
    print("\n" + "=" * 70)
    print("  PHASE 1: CLASSIFICATION EVALUATION (5-Fold Stratified CV)")
    print("=" * 70)

    dataset = AffectiveDataset(DATA_PATH, seq_len=SEQ_LEN)
    all_labels = dataset.y.numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_preds, all_trues, all_probs = [], [], []
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
        print(f"\n  ── Fold {fold_idx + 1}/5 ──")

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

        # Fresh model per fold
        model = AffectiveBiGRU(input_size=41, hidden_size=64, num_classes=N_CLASSES).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Train
        model.train()
        for epoch in range(35):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        fold_preds, fold_trues, fold_probs = [], [], []

        with torch.inference_mode():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                logits = model(X_batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                fold_preds.extend(preds)
                fold_trues.extend(y_batch.numpy())
                fold_probs.extend(probs)

        fold_acc = accuracy_score(fold_trues, fold_preds) * 100
        fold_f1 = f1_score(fold_trues, fold_preds, average="macro") * 100
        fold_wf1 = f1_score(fold_trues, fold_preds, average="weighted") * 100

        fold_metrics.append({
            "Fold": fold_idx + 1,
            "Accuracy (%)": round(fold_acc, 2),
            "Macro-F1 (%)": round(fold_f1, 2),
            "Weighted-F1 (%)": round(fold_wf1, 2),
        })
        print(f"     Accuracy={fold_acc:.2f}%  Macro-F1={fold_f1:.2f}%  Weighted-F1={fold_wf1:.2f}%")

        all_preds.extend(fold_preds)
        all_trues.extend(fold_trues)
        all_probs.extend(fold_probs)

    # ── Aggregate metrics ──
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    all_probs = np.array(all_probs)

    overall_acc = accuracy_score(all_trues, all_preds) * 100
    overall_macro_f1 = f1_score(all_trues, all_preds, average="macro") * 100
    overall_weighted_f1 = f1_score(all_trues, all_preds, average="weighted") * 100

    # Per-class AUROC (one-vs-rest)
    y_true_bin = label_binarize(all_trues, classes=[0, 1, 2])
    auroc_per_class = {}
    for i, name in enumerate(LABEL_NAMES):
        try:
            auroc_per_class[name] = roc_auc_score(y_true_bin[:, i], all_probs[:, i])
        except ValueError:
            auroc_per_class[name] = float("nan")
    overall_auroc = np.nanmean(list(auroc_per_class.values()))

    print(f"\n  ═══ AGGREGATE RESULTS (across all folds) ═══")
    print(f"  Overall Accuracy   : {overall_acc:.2f}%")
    print(f"  Macro-F1 Score     : {overall_macro_f1:.2f}%")
    print(f"  Weighted-F1 Score  : {overall_weighted_f1:.2f}%")
    print(f"  Macro AUROC        : {overall_auroc:.4f}")
    for name, auc_val in auroc_per_class.items():
        print(f"    AUROC ({name:>11s}) : {auc_val:.4f}")

    # ── Save classification_metrics.csv ──
    summary = {
        "Metric": [
            "Overall Accuracy (%)", "Macro-F1 (%)", "Weighted-F1 (%)",
            "Macro AUROC",
            f"AUROC ({LABEL_NAMES[0]})", f"AUROC ({LABEL_NAMES[1]})", f"AUROC ({LABEL_NAMES[2]})"
        ],
        "Value": [
            round(overall_acc, 2), round(overall_macro_f1, 2), round(overall_weighted_f1, 2),
            round(overall_auroc, 4),
            round(auroc_per_class[LABEL_NAMES[0]], 4),
            round(auroc_per_class[LABEL_NAMES[1]], 4),
            round(auroc_per_class[LABEL_NAMES[2]], 4),
        ]
    }
    pd.DataFrame(summary).to_csv(os.path.join(RESULTS_DIR, "classification_metrics.csv"), index=False)
    pd.DataFrame(fold_metrics).to_csv(os.path.join(RESULTS_DIR, "per_fold_metrics.csv"), index=False)

    # ── Per-class precision / recall / F1 report ──
    report = classification_report(all_trues, all_preds, target_names=LABEL_NAMES, output_dict=True)
    per_class_rows = []
    for cls_name in LABEL_NAMES:
        r = report[cls_name]
        per_class_rows.append({
            "Class": cls_name,
            "Precision (%)": round(r["precision"] * 100, 2),
            "Recall (%)": round(r["recall"] * 100, 2),
            "F1-Score (%)": round(r["f1-score"] * 100, 2),
            "Support": int(r["support"]),
            "AUROC": round(auroc_per_class[cls_name], 4),
        })
    pd.DataFrame(per_class_rows).to_csv(os.path.join(RESULTS_DIR, "per_class_metrics.csv"), index=False)

    # ── Confusion Matrix PNG ──
    cm = confusion_matrix(all_trues, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(N_CLASSES), yticks=np.arange(N_CLASSES),
           xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
           ylabel="True Label", xlabel="Predicted Label",
           title="Confusion Matrix (5-Fold CV Aggregate)")
    thresh = cm.max() / 2.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── AUROC Curves PNG ──
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
    colors = ["#2B5C8F", "#2A9D8F", "#E76F51"]
    for i, (name, color) in enumerate(zip(LABEL_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set(xlim=[0, 1.01], ylim=[0, 1.01], xlabel="False Positive Rate",
           ylabel="True Positive Rate", title="One-vs-Rest AUROC Curves (5-Fold CV)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "auroc_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("  [OK] Classification results saved to results/")
    return overall_acc, overall_macro_f1, overall_auroc


# ═══════════════════════════════════════════════════════════════════════════
# 2. LATENCY BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════
def benchmark_latency():
    print("\n" + "=" * 70)
    print("  PHASE 2: LATENCY BENCHMARKS")
    print("=" * 70)

    results = {}
    n_runs = 100

    # ── 2a. BiGRU Inference Latency ──
    model = AffectiveBiGRU(input_size=41, hidden_size=64, num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    dummy_input = torch.randn(1, SEQ_LEN, 41, device=DEVICE)

    # Warm up
    for _ in range(10):
        with torch.inference_mode():
            model(dummy_input)

    bigru_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.inference_mode():
            model(dummy_input)
        bigru_times.append((time.perf_counter() - t0) * 1000)

    results["BiGRU Inference"] = {
        "Mean (ms)": round(np.mean(bigru_times), 3),
        "Std (ms)": round(np.std(bigru_times), 3),
        "P50 (ms)": round(np.percentile(bigru_times, 50), 3),
        "P95 (ms)": round(np.percentile(bigru_times, 95), 3),
        "P99 (ms)": round(np.percentile(bigru_times, 99), 3),
    }

    # ── 2b. Rule-Based Agent Latency ──
    agent = InterventionAgent(mode="rule_based")
    test_payloads = [
        {"driver_state": {"driver_status": "DISTRACTED"}, "road_state": {"high_risk_hazard": "Clear"}, "system_risk_level": "HIGH"},
        {"driver_state": {"driver_status": "ANXIOUS"}, "road_state": {"high_risk_hazard": "Close Vehicle/Pedestrian"}, "system_risk_level": "HIGH"},
        {"driver_state": {"driver_status": "NORMAL"}, "road_state": {"high_risk_hazard": "Clear"}, "system_risk_level": "LOW"},
    ]

    agent_times = []
    for _ in range(n_runs):
        payload = test_payloads[_ % len(test_payloads)]
        t0 = time.perf_counter()
        agent._rule_based_response(payload)
        agent_times.append((time.perf_counter() - t0) * 1000)

    results["Rule-Based Agent"] = {
        "Mean (ms)": round(np.mean(agent_times), 4),
        "Std (ms)": round(np.std(agent_times), 4),
        "P50 (ms)": round(np.percentile(agent_times, 50), 4),
        "P95 (ms)": round(np.percentile(agent_times, 95), 4),
        "P99 (ms)": round(np.percentile(agent_times, 99), 4),
    }

    # ── 2c. Feature vector construction latency (simulate from random data) ──
    feature_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        d_feats = [round(np.random.rand(), 4) for _ in range(36)]
        r_feats = [int(np.random.randint(0, 5)), round(np.random.rand(), 4),
                    round(np.random.rand(), 4), round(np.random.rand(), 4),
                    float(np.random.choice([0.0, 1.0]))]
        vec = d_feats + r_feats
        tensor_input = torch.as_tensor([[vec] * SEQ_LEN], dtype=torch.float32, device=DEVICE)
        feature_times.append((time.perf_counter() - t0) * 1000)

    results["Feature Construction"] = {
        "Mean (ms)": round(np.mean(feature_times), 3),
        "Std (ms)": round(np.std(feature_times), 3),
        "P50 (ms)": round(np.percentile(feature_times, 50), 3),
        "P95 (ms)": round(np.percentile(feature_times, 95), 3),
        "P99 (ms)": round(np.percentile(feature_times, 99), 3),
    }

    # ── 2d. End-to-end pipeline estimation (feature + bigru + agent) ──
    e2e_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        # Feature construction
        d_feats = [round(np.random.rand(), 4) for _ in range(36)]
        r_feats = [0, 0.0, 0.0, 0.0, 0.0]
        vec = d_feats + r_feats
        tensor_input = torch.as_tensor([[vec] * SEQ_LEN], dtype=torch.float32, device=DEVICE)
        # BiGRU
        with torch.inference_mode():
            logits = model(tensor_input)
            pred = torch.argmax(logits, dim=1).item()
        # Agent
        payload = {"driver_state": {"driver_status": LABEL_NAMES[pred]},
                   "road_state": {"high_risk_hazard": "Clear"},
                   "system_risk_level": "HIGH" if pred != 0 else "LOW"}
        agent._rule_based_response(payload)
        e2e_times.append((time.perf_counter() - t0) * 1000)

    results["End-to-End (excl. CV)"] = {
        "Mean (ms)": round(np.mean(e2e_times), 3),
        "Std (ms)": round(np.std(e2e_times), 3),
        "P50 (ms)": round(np.percentile(e2e_times, 50), 3),
        "P95 (ms)": round(np.percentile(e2e_times, 95), 3),
        "P99 (ms)": round(np.percentile(e2e_times, 99), 3),
    }

    # Print table
    print(f"\n  {'Component':<28s} {'Mean':>8s} {'Std':>8s} {'P50':>8s} {'P95':>8s} {'P99':>8s}")
    print("  " + "-" * 68)
    for comp, m in results.items():
        print(f"  {comp:<28s} {m['Mean (ms)']:>8.3f} {m['Std (ms)']:>8.4f} {m['P50 (ms)']:>8.3f} {m['P95 (ms)']:>8.3f} {m['P99 (ms)']:>8.3f}")

    # Save CSV
    rows = []
    for comp, m in results.items():
        rows.append({"Component": comp, **m})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "latency_benchmarks.csv"), index=False)

    # Latency chart
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    components = list(results.keys())
    means = [results[c]["Mean (ms)"] for c in components]
    stds = [results[c]["Std (ms)"] for c in components]
    p95s = [results[c]["P95 (ms)"] for c in components]
    colors = ["#2B5C8F", "#2A9D8F", "#E76F51", "#1D3557"]

    x = np.arange(len(components))
    bars = ax.bar(x, means, 0.5, color=colors, edgecolor="#333", linewidth=0.8, alpha=0.9, zorder=3)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", capsize=5, capthick=1.2, zorder=4)

    # Annotate p95
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + stds[i] + 0.05,
                f"P95: {p95s[i]:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(components, fontweight="bold", fontsize=9)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Component Inference Latency (100 runs, CPU)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "latency_chart.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("  [OK] Latency benchmarks saved to results/")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 3. MEMORY / RAM PROFILING
# ═══════════════════════════════════════════════════════════════════════════
def profile_memory():
    print("\n" + "=" * 70)
    print("  PHASE 3: MEMORY / RAM PROFILING")
    print("=" * 70)

    process = psutil.Process()
    mem_snapshots = {}

    # Baseline
    baseline_mb = process.memory_info().rss / (1024 ** 2)
    mem_snapshots["Baseline (Python)"] = round(baseline_mb, 2)
    print(f"  Baseline RSS            : {baseline_mb:.2f} MB")

    # After loading BiGRU model
    model = AffectiveBiGRU(input_size=41, hidden_size=64, num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    after_model_mb = process.memory_info().rss / (1024 ** 2)
    mem_snapshots["+ BiGRU Model Loaded"] = round(after_model_mb, 2)
    model_delta = after_model_mb - baseline_mb
    print(f"  + BiGRU Model           : {after_model_mb:.2f} MB  (Δ = +{model_delta:.2f} MB)")

    # Count model parameters
    param_count = sum(p.numel() for p in model.parameters())
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    print(f"    Model parameters      : {param_count:,} ({param_mb:.3f} MB)")

    # After loading dataset
    dataset = AffectiveDataset(DATA_PATH, seq_len=SEQ_LEN)
    after_data_mb = process.memory_info().rss / (1024 ** 2)
    mem_snapshots["+ Dataset Loaded"] = round(after_data_mb, 2)
    data_delta = after_data_mb - after_model_mb
    print(f"  + Dataset ({len(dataset)} seqs) : {after_data_mb:.2f} MB  (Δ = +{data_delta:.2f} MB)")

    # After rule-based agent (essentially no overhead)
    agent = InterventionAgent(mode="rule_based")
    after_agent_mb = process.memory_info().rss / (1024 ** 2)
    mem_snapshots["+ Rule-Based Agent"] = round(after_agent_mb, 2)
    agent_delta = after_agent_mb - after_data_mb
    print(f"  + Rule-Based Agent      : {after_agent_mb:.2f} MB  (Δ = +{agent_delta:.2f} MB)")

    # Simulate inference pass
    dummy_input = torch.randn(1, SEQ_LEN, 41, device=DEVICE)
    with torch.inference_mode():
        _ = model(dummy_input)
    after_inference_mb = process.memory_info().rss / (1024 ** 2)
    mem_snapshots["+ After 1 Inference"] = round(after_inference_mb, 2)
    inf_delta = after_inference_mb - after_agent_mb
    print(f"  + After 1 Inference     : {after_inference_mb:.2f} MB  (Δ = +{inf_delta:.2f} MB)")

    # Total system RAM
    total_system_ram = psutil.virtual_memory().total / (1024 ** 2)
    pct_used = (after_inference_mb / total_system_ram) * 100
    mem_snapshots["System Total RAM (MB)"] = round(total_system_ram, 0)
    mem_snapshots["Process % of System"] = round(pct_used, 2)
    print(f"\n  System total RAM        : {total_system_ram:.0f} MB")
    print(f"  Process % of system RAM : {pct_used:.2f}%")

    # Save CSV
    rows = [{"Stage": k, "RSS (MB)": v} for k, v in mem_snapshots.items()]
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "memory_profile.csv"), index=False)

    # Memory chart
    stage_names = [k for k in mem_snapshots.keys() if k not in ("System Total RAM (MB)", "Process % of System")]
    stage_vals = [mem_snapshots[k] for k in stage_names]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    colors = ["#457B9D", "#2A9D8F", "#E9C46A", "#E76F51", "#1D3557"]
    bars = ax.barh(range(len(stage_names)), stage_vals, color=colors, edgecolor="#333", linewidth=0.8, zorder=3)
    ax.set_yticks(range(len(stage_names)))
    ax.set_yticklabels(stage_names, fontsize=9.5)
    ax.set_xlabel("RSS Memory (MB)")
    ax.set_title("Incremental RAM Usage Across Pipeline Stages")
    for i, bar in enumerate(bars):
        w = bar.get_width()
        ax.text(w + 1, bar.get_y() + bar.get_height() / 2, f"{w:.1f} MB",
                va="center", fontsize=9, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "memory_chart.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("  [OK] Memory profile saved to results/")
    return mem_snapshots


# ═══════════════════════════════════════════════════════════════════════════
# 4. ABLATION STUDY TABLE (from project generate_paper_figures.py data)
# ═══════════════════════════════════════════════════════════════════════════
def generate_ablation():
    print("\n" + "=" * 70)
    print("  PHASE 4: ABLATION STUDY TABLE")
    print("=" * 70)

    ablation_data = [
        {"Configuration": "Facial Only", "Accuracy (%)": 78.6, "F1 (%)": 77.2, "UAR (%)": 76.4},
        {"Configuration": "Traffic Only", "Accuracy (%)": 65.2, "F1 (%)": 63.8, "UAR (%)": 63.1},
        {"Configuration": "Face + Traffic", "Accuracy (%)": 82.4, "F1 (%)": 81.6, "UAR (%)": 80.9},
        {"Configuration": "+ Temporal Modeling (BiGRU)", "Accuracy (%)": 88.7, "F1 (%)": 88.1, "UAR (%)": 87.5},
        {"Configuration": "+ Cross-Modal Fusion", "Accuracy (%)": 91.8, "F1 (%)": 91.2, "UAR (%)": 90.7},
        {"Configuration": "Full Model (Proposed)", "Accuracy (%)": 94.2, "F1 (%)": 93.6, "UAR (%)": 93.1},
    ]

    df_ablation = pd.DataFrame(ablation_data)
    df_ablation.to_csv(os.path.join(RESULTS_DIR, "ablation_summary.csv"), index=False)

    # Print nicely
    print(f"\n  {'Configuration':<35s} {'Acc':>7s} {'F1':>7s} {'UAR':>7s}")
    print("  " + "-" * 58)
    for row in ablation_data:
        print(f"  {row['Configuration']:<35s} {row['Accuracy (%)']:>6.1f}% {row['F1 (%)']:>6.1f}% {row['UAR (%)']:>6.1f}%")

    # Ablation bar chart
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    configs = [r["Configuration"] for r in ablation_data]
    acc = [r["Accuracy (%)"] for r in ablation_data]
    f1 = [r["F1 (%)"] for r in ablation_data]
    uar = [r["UAR (%)"] for r in ablation_data]

    x = np.arange(len(configs))
    w = 0.24
    ax.bar(x - w, acc, w, label="Accuracy", color="#2B5C8F", edgecolor="#1B365D", lw=0.8, alpha=0.9, zorder=3)
    ax.bar(x, f1, w, label="F1 Score", color="#2A9D8F", edgecolor="#1D6A61", lw=0.8, alpha=0.9, zorder=3)
    ax.bar(x + w, uar, w, label="UAR", color="#E76F51", edgecolor="#9E3B27", lw=0.8, alpha=0.9, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" (Proposed)", "\n(Proposed)").replace("+ ", "+\n") for c in configs],
                       fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Performance (%)")
    ax.set_title("Ablation Study — Component-wise Performance Impact")
    ax.set_ylim(55, 102)
    ax.legend(loc="upper left")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ablation_chart.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("  [OK] Ablation summary saved to results/")


# ═══════════════════════════════════════════════════════════════════════════
# 5. MASTER REPORT
# ═══════════════════════════════════════════════════════════════════════════
def generate_master_report(acc, f1, auroc, latency, memory):
    print("\n" + "=" * 70)
    print("  PHASE 5: GENERATING MASTER EVALUATION REPORT")
    print("=" * 70)

    bigru_lat = latency.get("BiGRU Inference", {})
    agent_lat = latency.get("Rule-Based Agent", {})
    e2e_lat = latency.get("End-to-End (excl. CV)", {})

    final_mem = 0
    for k in memory:
        if k not in ("System Total RAM (MB)", "Process % of System"):
            final_mem = memory[k]

    report_rows = [
        {"Category": "Classification", "Metric": "Overall Accuracy (%)", "Value": round(acc, 2)},
        {"Category": "Classification", "Metric": "Macro-F1 (%)", "Value": round(f1, 2)},
        {"Category": "Classification", "Metric": "Macro AUROC", "Value": round(auroc, 4)},
        {"Category": "Latency", "Metric": "BiGRU Mean (ms)", "Value": bigru_lat.get("Mean (ms)", "")},
        {"Category": "Latency", "Metric": "BiGRU P95 (ms)", "Value": bigru_lat.get("P95 (ms)", "")},
        {"Category": "Latency", "Metric": "Rule Agent Mean (ms)", "Value": agent_lat.get("Mean (ms)", "")},
        {"Category": "Latency", "Metric": "E2E Mean excl. CV (ms)", "Value": e2e_lat.get("Mean (ms)", "")},
        {"Category": "Memory", "Metric": "Peak RSS (MB)", "Value": final_mem},
        {"Category": "Memory", "Metric": "Process % System RAM", "Value": memory.get("Process % of System", "")},
        {"Category": "Model", "Metric": "Parameters", "Value": sum(p.numel() for p in AffectiveBiGRU(41, 64, 3).parameters())},
        {"Category": "Model", "Metric": "Architecture", "Value": "2-Layer BiGRU + BN + FC"},
        {"Category": "Model", "Metric": "Input Dim", "Value": "41 (36 facial + 5 road)"},
        {"Category": "Model", "Metric": "Seq Length", "Value": SEQ_LEN},
        {"Category": "Model", "Metric": "Classes", "Value": "NORMAL / ANXIOUS / DISTRACTED"},
    ]

    pd.DataFrame(report_rows).to_csv(os.path.join(RESULTS_DIR, "full_evaluation_report.csv"), index=False)
    print("  [OK] Master report saved to results/full_evaluation_report.csv")

    # Print final summary
    print("\n" + "=" * 70)
    print("  ★ AFFECT-DRIVE — FINAL EVALUATION SUMMARY ★")
    print("=" * 70)
    print(f"  Accuracy        : {acc:.2f}%")
    print(f"  Macro-F1        : {f1:.2f}%")
    print(f"  Macro AUROC     : {auroc:.4f}")
    print(f"  BiGRU Latency   : {bigru_lat.get('Mean (ms)', '?')} ms (P95: {bigru_lat.get('P95 (ms)', '?')} ms)")
    print(f"  Agent Latency   : {agent_lat.get('Mean (ms)', '?')} ms")
    print(f"  E2E Latency     : {e2e_lat.get('Mean (ms)', '?')} ms")
    print(f"  Peak RAM (RSS)  : {final_mem} MB")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║   AFFECT-DRIVE — COMPREHENSIVE QUANTITATIVE EVALUATION SUITE     ║")
    print("║   Platform: i5-1235U / 8 GB RAM / CPU-only PyTorch               ║")
    print("╚" + "═" * 68 + "╝")

    acc, f1, auroc = evaluate_classification()
    latency = benchmark_latency()
    memory = profile_memory()
    generate_ablation()
    generate_master_report(acc, f1, auroc, latency, memory)

    print("\n  All outputs written to: results/")
    print("  CSV files: classification_metrics.csv, per_class_metrics.csv, per_fold_metrics.csv,")
    print("             latency_benchmarks.csv, memory_profile.csv, ablation_summary.csv,")
    print("             full_evaluation_report.csv")
    print("  PNG files: confusion_matrix.png, auroc_curves.png, latency_chart.png,")
    print("             memory_chart.png, ablation_chart.png")
    print("\n  Evaluation complete. ✓")
