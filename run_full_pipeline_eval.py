"""
Affect-Drive — Remaining Full-Pipeline & Measured Ablation Evaluation
===================================================================
Runs the full dual-stream Affect-Drive pipeline (OpenCV + MediaPipe FaceMesh +
YOLOv8n + AffectiveBiGRU + Rule-Based InterventionAgent) headless on
data/sample_driver.mp4 and data/sample_road.mp4, plus measured ablation on
data/bigru_affective.pth & data/phase2_dataset.csv.

Outputs written to results/:
  - full_pipeline_latency.csv / full_pipeline_latency_chart.png
  - full_pipeline_memory.csv  / full_pipeline_memory_chart.png
  - full_pipeline_actuation_log.csv
  - full_pipeline_video_summary.csv
  - measured_ablation.csv / measured_ablation_chart.png
  - full_evaluation_report.csv (updated with accurate params & full pipeline metrics)
"""

import os
import sys
import csv
import time
import warnings
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), ".mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import psutil
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from src.model.bigru_model import AffectiveBiGRU
from src.model.train import AffectiveDataset
from src.agent.llm_intervention import InterventionAgent

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LABEL_NAMES = ["NORMAL", "ANXIOUS", "DISTRACTED"]
DEVICE = torch.device("cpu")
DATA_PATH = "data/phase2_dataset.csv"
MODEL_PATH = "data/bigru_affective.pth"
DRIVER_VIDEO = "data/sample_driver.mp4"
ROAD_VIDEO = "data/sample_road.mp4"
MAX_VIDEO_FRAMES = 300

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "figure.dpi": 300,
    "pdf.fonttype": 42,
})


def summarize_times(arr):
    a = np.asarray(arr, dtype=np.float64)
    return {
        "Mean (ms)": round(float(np.mean(a)), 3),
        "Std (ms)": round(float(np.std(a)), 3),
        "P50 (ms)": round(float(np.percentile(a, 50)), 3),
        "P95 (ms)": round(float(np.percentile(a, 95)), 3),
        "P99 (ms)": round(float(np.percentile(a, 99)), 3),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: MEASURED ABLATION ON PRE-TRAINED BiGRU + phase2_dataset.csv
# ═══════════════════════════════════════════════════════════════════════════
def run_measured_ablation():
    print("\n" + "=" * 70)
    print("  PART 1: MEASURED ABLATION ON PRE-TRAINED BiGRU (phase2_dataset.csv)")
    print("=" * 70)

    model = AffectiveBiGRU(input_size=41, hidden_size=64, num_layers=2, num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    dataset = AffectiveDataset(DATA_PATH, seq_len=30)
    X_all = dataset.X.clone()  # (271, 30, 41)
    y_all = dataset.y.numpy()
    y_bin = label_binarize(y_all, classes=[0, 1, 2])

    configs = {
        "Traffic Only (5-dim)": lambda x: x.clone().index_fill_(2, torch.arange(0, 36), 0.0),
        "Facial Only (36-dim)": lambda x: x.clone().index_fill_(2, torch.arange(36, 41), 0.0),
        "Single-Frame (t=-1)":  lambda x: x.clone().index_fill_(1, torch.arange(0, 29), 0.0),
        "Full BiGRU (41-dim)":  lambda x: x.clone(),
    }

    rows = []
    for name, transform in configs.items():
        X_mod = transform(X_all)
        with torch.inference_mode():
            logits = model(X_mod.to(DEVICE))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        acc = accuracy_score(y_all, preds) * 100.0
        mf1 = f1_score(y_all, preds, average="macro") * 100.0
        wf1 = f1_score(y_all, preds, average="weighted") * 100.0
        uar = recall_score(y_all, preds, average="macro") * 100.0
        auroc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")

        rows.append({
            "Configuration": name,
            "Accuracy (%)": round(acc, 2),
            "Macro-F1 (%)": round(mf1, 2),
            "Weighted-F1 (%)": round(wf1, 2),
            "UAR (%)": round(uar, 2),
            "Macro AUROC": round(auroc, 4),
        })
        print(f"  {name:<22s} | Acc={acc:6.2f}% | Macro-F1={mf1:6.2f}% | UAR={uar:6.2f}% | AUROC={auroc:.4f}")

    df_abl = pd.DataFrame(rows)
    df_abl.to_csv(os.path.join(RESULTS_DIR, "measured_ablation.csv"), index=False)

    # Plot measured ablation chart
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=300)
    names = [r["Configuration"].replace(" (", "\n(") for r in rows]
    acc_v = [r["Accuracy (%)"] for r in rows]
    f1_v = [r["Macro-F1 (%)"] for r in rows]
    uar_v = [r["UAR (%)"] for r in rows]

    x = np.arange(len(names))
    w = 0.25
    b1 = ax.bar(x - w, acc_v, w, label="Accuracy (%)", color="#2B5C8F", edgecolor="#1B365D", lw=0.8)
    b2 = ax.bar(x, f1_v, w, label="Macro-F1 (%)", color="#2A9D8F", edgecolor="#1D6A61", lw=0.8)
    b3 = ax.bar(x + w, uar_v, w, label="UAR (%)", color="#E76F51", edgecolor="#9E3B27", lw=0.8)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_title("Measured Feature & Temporal Ablation (Pre-trained BiGRU)")
    ax.set_ylim(0, 108)
    ax.legend(loc="upper left")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "measured_ablation_chart.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("  [OK] Measured ablation saved to results/measured_ablation.csv & measured_ablation_chart.png")
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: FULL-PIPELINE VIDEO RUN, CV COMPONENT LATENCY & REAL RAM PROFILING
# ═══════════════════════════════════════════════════════════════════════════
def run_full_pipeline_benchmark():
    print("\n" + "=" * 70)
    print("  PART 2: FULL-PIPELINE EXECUTION & CV LATENCY/RAM PROFILING")
    print("=" * 70)

    proc = psutil.Process()
    mem_stages = []

    def log_mem(stage_name):
        rss_mb = round(proc.memory_info().rss / (1024 ** 2), 2)
        mem_stages.append({"Stage": stage_name, "RSS (MB)": rss_mb})
        print(f"  [MEM] {stage_name:<35s}: {rss_mb:.2f} MB")
        return rss_mb

    log_mem("Baseline (Python + Torch)")

    # Import MediaPipe & YOLO cleanly and track memory step-by-step
    import mediapipe as mp
    try:
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
    except (ModuleNotFoundError, AttributeError):
        try:
            import mediapipe.solutions.face_mesh as mp_face_mesh
        except (ModuleNotFoundError, AttributeError):
            mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=False, min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    log_mem("+ MediaPipe FaceMesh")

    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")
    # Run 1 dummy warmup to materialize weights in RAM
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = yolo.predict(dummy_img, verbose=False, conf=0.40, iou=0.5, classes=[0, 2, 3, 5, 7], device="cpu", imgsz=640)
    log_mem("+ YOLOv8n (yolov8n.pt)")

    # Optional OpenVINO INT8 check
    ov_latencies = []
    ov_available = False
    if os.path.isdir("yolov8n_int8_openvino_model"):
        try:
            yolo_ov = YOLO("yolov8n_int8_openvino_model", task="detect")
            _ = yolo_ov.predict(dummy_img, verbose=False, conf=0.40, iou=0.5, classes=[0, 2, 3, 5, 7], imgsz=640)
            ov_available = True
            print("  [INFO] OpenVINO INT8 YOLOv8n model loaded successfully for comparison.")
        except Exception as e:
            print(f"  [INFO] OpenVINO INT8 YOLOv8n unavailable in current runtime ({e}).")

    # Initialize full AffectivePerceptionEngine (matching main_dataset_mode.py)
    from src.perception.fusion_pipeline import AffectivePerceptionEngine
    engine = AffectivePerceptionEngine(
        driver_src=DRIVER_VIDEO,
        road_src=ROAD_VIDEO,
        model_path=MODEL_PATH,
        seq_len=30,
        skip_road_frames=2,
        target_resolution=(640, 480)
    )
    log_mem("+ Full AffectivePerceptionEngine")

    agent = InterventionAgent(mode="rule_based")
    log_mem("+ Rule-Based InterventionAgent")

    # Clean up standalone warmup objects so RSS reflects actual runtime engine cleanly
    del face_mesh, yolo

    # Also measure standalone component latencies directly on real video frames (60 frames)
    cap_d = cv2.VideoCapture(DRIVER_VIDEO)
    cap_r = cv2.VideoCapture(ROAD_VIDEO)
    total_d_frames = int(cap_d.get(cv2.CAP_PROP_FRAME_COUNT))
    total_r_frames = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  [VIDEO] Driver video frames={total_d_frames}, Road video frames={total_r_frames}")

    mp_times, yolo_times = [], []
    n_comp = min(60, total_d_frames, total_r_frames)
    for idx in range(n_comp):
        rd, fd = cap_d.read()
        rr, fr = cap_r.read()
        if not rd or not rr:
            break
        fd = cv2.resize(fd, (640, 480), interpolation=cv2.INTER_LINEAR)
        fr = cv2.resize(fr, (640, 480), interpolation=cv2.INTER_LINEAR)

        t0 = time.perf_counter()
        _ = engine._extract_driver_features(fd)
        dt_mp = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        _ = engine._extract_road_features(fr)
        dt_yolo = (time.perf_counter() - t0) * 1000.0

        if ov_available:
            t0 = time.perf_counter()
            _ = yolo_ov.predict(fr, verbose=False, conf=0.40, iou=0.5, classes=[0, 2, 3, 5, 7], imgsz=640)
            dt_ov = (time.perf_counter() - t0) * 1000.0

        # Skip first 3 warmup frames
        if idx >= 3:
            mp_times.append(dt_mp)
            yolo_times.append(dt_yolo)
            if ov_available:
                ov_latencies.append(dt_ov)

    cap_d.release()
    cap_r.release()

    # Run full AffectivePerceptionEngine.step() + InterventionAgent headless loop
    n_run = min(MAX_VIDEO_FRAMES, total_d_frames) if total_d_frames > 0 else MAX_VIDEO_FRAMES
    step_times = []
    agent_times = []
    actuation_rows = []
    peak_rss_mb = proc.memory_info().rss / (1024 ** 2)

    last_risk = "LOW"
    last_interv_t = 0.0
    cooldown_s = 5.0
    t_loop_start = time.perf_counter()

    for frame_idx in range(1, n_run + 1):
        t0 = time.perf_counter()
        d_frame, r_frame, payload = engine.step()
        step_ms = (time.perf_counter() - t0) * 1000.0

        if d_frame is None or payload is None:
            break

        now = time.time()
        risk = payload.get("system_risk_level", "LOW")
        d_status = payload.get("driver_state", {}).get("driver_status", "NORMAL")
        r_hazard = payload.get("road_state", {}).get("high_risk_hazard", "Clear")

        ai_text = ""
        if risk == "HIGH" and (last_risk == "LOW" or (now - last_interv_t > cooldown_s)):
            t_a0 = time.perf_counter()
            ai_text = agent._rule_based_response(payload)
            agent_times.append((time.perf_counter() - t_a0) * 1000.0)
            last_interv_t = now
        else:
            t_a0 = time.perf_counter()
            _ = agent._rule_based_response(payload)
            agent_times.append((time.perf_counter() - t_a0) * 1000.0)

        last_risk = risk

        # Exclude first 5 warmup frames from step latency stats
        if frame_idx > 5:
            step_times.append(step_ms)

        if frame_idx % 20 == 0 or frame_idx == n_run:
            cur_rss = proc.memory_info().rss / (1024 ** 2)
            if cur_rss > peak_rss_mb:
                peak_rss_mb = cur_rss

        actuation_rows.append({
            "frame_id": frame_idx,
            "elapsed_sec": round(time.perf_counter() - t_loop_start, 3),
            "step_latency_ms": round(step_ms, 2),
            "risk_level": risk,
            "signal": "H" if risk == "HIGH" else "L",
            "driver_status": d_status,
            "road_hazard": r_hazard,
            "ai_intervention": ai_text,
        })

    total_wall_s = time.perf_counter() - t_loop_start
    engine.stop()

    peak_rss_mb = round(max(peak_rss_mb, proc.memory_info().rss / (1024 ** 2)), 2)
    mem_stages.append({"Stage": "Peak Active Video Processing", "RSS (MB)": peak_rss_mb})
    total_sys_ram = round(psutil.virtual_memory().total / (1024 ** 2), 1)
    pct_sys_ram = round((peak_rss_mb / total_sys_ram) * 100.0, 2)
    mem_stages.append({"Stage": "System Total RAM (MB)", "RSS (MB)": total_sys_ram})
    mem_stages.append({"Stage": "Peak Process % of System", "RSS (MB)": pct_sys_ram})
    print(f"  [MEM] Peak Active Video Processing     : {peak_rss_mb:.2f} MB ({pct_sys_ram:.2f}% of {total_sys_ram:.0f} MB)")

    # Also measure standalone BiGRU latency cleanly
    bigru = AffectiveBiGRU(input_size=41, hidden_size=64, num_layers=2, num_classes=3).to(DEVICE)
    bigru.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    bigru.eval()
    dummy_seq = torch.randn(1, 30, 41, device=DEVICE)
    for _ in range(10):
        with torch.inference_mode():
            _ = bigru(dummy_seq)
    bigru_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = bigru(dummy_seq)
        bigru_times.append((time.perf_counter() - t0) * 1000.0)

    # Save actuation log
    df_act = pd.DataFrame(actuation_rows)
    df_act.to_csv(os.path.join(RESULTS_DIR, "full_pipeline_actuation_log.csv"), index=False)

    # Video run summary
    frames_processed = len(df_act)
    eff_fps = round(frames_processed / max(total_wall_s, 1e-6), 2)
    status_counts = df_act["driver_status"].value_counts().to_dict()
    risk_counts = df_act["risk_level"].value_counts().to_dict()
    hazard_counts = df_act["road_hazard"].value_counts().to_dict()

    video_summary = [
        {"Metric": "Frames Processed", "Value": frames_processed},
        {"Metric": "Wall Time (s)", "Value": round(total_wall_s, 2)},
        {"Metric": "Effective Throughput (FPS)", "Value": eff_fps},
        {"Metric": "HIGH Risk Frames", "Value": risk_counts.get("HIGH", 0)},
        {"Metric": "LOW Risk Frames", "Value": risk_counts.get("LOW", 0)},
        {"Metric": "HIGH Risk (%)", "Value": round(risk_counts.get("HIGH", 0) / max(frames_processed, 1) * 100, 2)},
        {"Metric": "Driver NORMAL Frames", "Value": status_counts.get("NORMAL", 0)},
        {"Metric": "Driver ANXIOUS Frames", "Value": status_counts.get("ANXIOUS", 0)},
        {"Metric": "Driver DISTRACTED Frames", "Value": status_counts.get("DISTRACTED", 0)},
        {"Metric": "Close Hazard Frames", "Value": hazard_counts.get("Close Vehicle/Pedestrian", 0)},
        {"Metric": "AI Interventions Triggered", "Value": int((df_act["ai_intervention"] != "").sum())},
    ]
    pd.DataFrame(video_summary).to_csv(os.path.join(RESULTS_DIR, "full_pipeline_video_summary.csv"), index=False)

    print(f"\n  [VIDEO SUMMARY] Processed {frames_processed} frames in {total_wall_s:.2f}s ({eff_fps:.2f} FPS)")
    print(f"    Driver States: {status_counts}")
    print(f"    Risk Levels  : {risk_counts}")
    print(f"    Road Hazards : {hazard_counts}")

    # Compile full latency table
    latency_rows = [
        {"Component": "MediaPipe FaceMesh + Gaze", **summarize_times(mp_times)},
        {"Component": "YOLOv8n (CPU PyTorch)", **summarize_times(yolo_times)},
    ]
    if ov_available and ov_latencies:
        latency_rows.append({"Component": "YOLOv8n (OpenVINO INT8)", **summarize_times(ov_latencies)})
    latency_rows.extend([
        {"Component": "AffectiveBiGRU (30-frame)", **summarize_times(bigru_times)},
        {"Component": "Rule-Based Agent", **summarize_times(agent_times)},
        {"Component": "Full Pipeline Step (Async CV)", **summarize_times(step_times)},
    ])

    df_lat = pd.DataFrame(latency_rows)
    df_lat.to_csv(os.path.join(RESULTS_DIR, "full_pipeline_latency.csv"), index=False)

    print("\n  ═══ FULL-PIPELINE LATENCY BREAKDOWN ═══")
    print(f"  {'Component':<32s} {'Mean':>8s} {'Std':>8s} {'P50':>8s} {'P95':>8s} {'P99':>8s}")
    print("  " + "-" * 74)
    for r in latency_rows:
        print(f"  {r['Component']:<32s} {r['Mean (ms)']:>8.3f} {r['Std (ms)']:>8.3f} {r['P50 (ms)']:>8.3f} {r['P95 (ms)']:>8.3f} {r['P99 (ms)']:>8.3f}")

    # Save memory CSV
    df_mem = pd.DataFrame(mem_stages)
    df_mem.to_csv(os.path.join(RESULTS_DIR, "full_pipeline_memory.csv"), index=False)

    # Plot full latency chart
    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=300)
    comps = [r["Component"].replace(" (", "\n(") for r in latency_rows]
    means = [r["Mean (ms)"] for r in latency_rows]
    p95s = [r["P95 (ms)"] for r in latency_rows]
    colors = ["#457B9D", "#E76F51", "#F4A261", "#2A9D8F", "#8AB17D", "#1D3557"][:len(comps)]

    x = np.arange(len(comps))
    bars = ax.bar(x, means, 0.52, color=colors, edgecolor="#222", lw=0.8, alpha=0.92, zorder=3)
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max(means) * 0.02,
                f"{means[i]:.2f} ms\n(P95: {p95s[i]:.1f})", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(comps, fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Full End-to-End Pipeline Component Latency (Real Video Stream, CPU)")
    ax.set_ylim(0, max(means) * 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "full_pipeline_latency_chart.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot full memory chart
    plot_mem = [r for r in mem_stages if "System" not in r["Stage"]]
    fig, ax = plt.subplots(figsize=(9.5, 5), dpi=300)
    s_names = [r["Stage"] for r in plot_mem]
    s_vals = [r["RSS (MB)"] for r in plot_mem]
    m_colors = ["#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#1D3557"][:len(s_names)]

    bars = ax.barh(range(len(s_names)), s_vals, color=m_colors, edgecolor="#222", lw=0.8, zorder=3)
    ax.set_yticks(range(len(s_names)))
    ax.set_yticklabels(s_names, fontsize=9, fontweight="bold")
    ax.set_xlabel("Process RSS Memory (MB)")
    ax.set_title("Full-Pipeline Incremental RAM Footprint (OpenCV + MediaPipe + YOLOv8n + BiGRU)")
    ax.set_xlim(0, max(s_vals) * 1.18)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(s_vals) * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f} MB", va="center", fontsize=8.8, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "full_pipeline_memory_chart.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return latency_rows, mem_stages, video_summary


# ═══════════════════════════════════════════════════════════════════════════
# PART 3: UPDATE MASTER EVALUATION REPORT (ACCURATE & COMPLETE)
# ═══════════════════════════════════════════════════════════════════════════
def update_master_report(ablation_rows, latency_rows, mem_stages, video_summary):
    pre_df = pd.read_csv(os.path.join(RESULTS_DIR, "pretrained_model_metrics.csv"))
    pre_map = dict(zip(pre_df["Metric"], pre_df["Value"]))
    cv_df = pd.read_csv(os.path.join(RESULTS_DIR, "classification_metrics.csv"))
    cv_map = dict(zip(cv_df["Metric"], cv_df["Value"]))

    lat_map = {r["Component"]: r for r in latency_rows}
    mem_map = {r["Stage"]: r["RSS (MB)"] for r in mem_stages}
    vid_map = {r["Metric"]: r["Value"] for r in video_summary}

    param_count = sum(
        p.numel() for p in AffectiveBiGRU(input_size=41, hidden_size=64, num_layers=2, num_classes=3).parameters()
    )

    report = [
        {"Category": "Classification (Pretrained Checkpoint)", "Metric": "Accuracy (%)", "Value": pre_map.get("Accuracy (%)")},
        {"Category": "Classification (Pretrained Checkpoint)", "Metric": "Macro-F1 (%)", "Value": pre_map.get("Macro-F1 (%)")},
        {"Category": "Classification (Pretrained Checkpoint)", "Metric": "Weighted-F1 (%)", "Value": pre_map.get("Weighted-F1 (%)")},
        {"Category": "Classification (Pretrained Checkpoint)", "Metric": "Macro AUROC", "Value": pre_map.get("Macro AUROC")},
        {"Category": "Classification (5-Fold CV Scratch)", "Metric": "Accuracy (%)", "Value": cv_map.get("Overall Accuracy (%)")},
        {"Category": "Classification (5-Fold CV Scratch)", "Metric": "Macro-F1 (%)", "Value": cv_map.get("Macro-F1 (%)")},
        {"Category": "Classification (5-Fold CV Scratch)", "Metric": "Macro AUROC", "Value": cv_map.get("Macro AUROC")},
        {"Category": "Latency (Full CV Pipeline)", "Metric": "MediaPipe FaceMesh Mean (ms)", "Value": lat_map["MediaPipe FaceMesh + Gaze"]["Mean (ms)"]},
        {"Category": "Latency (Full CV Pipeline)", "Metric": "YOLOv8n CPU Mean (ms)", "Value": lat_map["YOLOv8n (CPU PyTorch)"]["Mean (ms)"]},
        {"Category": "Latency (Full CV Pipeline)", "Metric": "AffectiveBiGRU Mean (ms)", "Value": lat_map["AffectiveBiGRU (30-frame)"]["Mean (ms)"]},
        {"Category": "Latency (Full CV Pipeline)", "Metric": "Rule-Based Agent Mean (ms)", "Value": lat_map["Rule-Based Agent"]["Mean (ms)"]},
        {"Category": "Latency (Full CV Pipeline)", "Metric": "Full Async Step Mean (ms)", "Value": lat_map["Full Pipeline Step (Async CV)"]["Mean (ms)"]},
        {"Category": "Latency (Full CV Pipeline)", "Metric": "Full Async Step P95 (ms)", "Value": lat_map["Full Pipeline Step (Async CV)"]["P95 (ms)"]},
        {"Category": "Throughput", "Metric": "Effective Video FPS", "Value": vid_map.get("Effective Throughput (FPS)")},
        {"Category": "Memory (Full CV Stack)", "Metric": "Peak Active RSS (MB)", "Value": mem_map.get("Peak Active Video Processing")},
        {"Category": "Memory (Full CV Stack)", "Metric": "Peak % of System RAM", "Value": mem_map.get("Peak Process % of System")},
        {"Category": "Model", "Metric": "BiGRU Parameters", "Value": param_count},
        {"Category": "Model", "Metric": "Architecture", "Value": "2-Layer BiGRU (hidden=64) + BN + FC"},
        {"Category": "Model", "Metric": "Input / Sequence", "Value": "41 dims (36 facial + 5 road), seq_len=30"},
    ]
    if "YOLOv8n (OpenVINO INT8)" in lat_map:
        report.append({"Category": "Latency (Full CV Pipeline)", "Metric": "YOLOv8n OpenVINO INT8 Mean (ms)", "Value": lat_map["YOLOv8n (OpenVINO INT8)"]["Mean (ms)"]})

    pd.DataFrame(report).to_csv(os.path.join(RESULTS_DIR, "full_evaluation_report.csv"), index=False)
    print("\n  [OK] Updated results/full_evaluation_report.csv")


if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║  AFFECT-DRIVE — REMAINING FULL-PIPELINE & ABLATION EVALUATION    ║")
    print("╚" + "═" * 68 + "╝")
    abl = run_measured_ablation()
    lat, mem, vid = run_full_pipeline_benchmark()
    update_master_report(abl, lat, mem, vid)
    print("\n  All remaining evaluations complete. ✓")
