"""
Clean, non-duplicated RSS memory profiling for the production Affect-Drive stack:
  Python + Torch -> AffectivePerceptionEngine (OpenCV + MediaPipe + YOLOv8n + BiGRU)
  -> Rule-Based InterventionAgent + AudioAlertEngine -> Active 100-frame Video Run.
"""
import os, sys, gc, warnings
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), ".mpl_cache")

import psutil
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from src.perception.fusion_pipeline import AffectivePerceptionEngine
from src.agent.llm_intervention import InterventionAgent
from src.agent.audio_alerts import AudioAlertEngine

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
proc = psutil.Process()
stages = []

def snap(name):
    gc.collect()
    mb = round(proc.memory_info().rss / (1024 ** 2), 2)
    stages.append({"Stage": name, "RSS (MB)": mb})
    print(f"  [CLEAN MEM] {name:<42s}: {mb:.2f} MB")
    return mb

snap("Baseline (Python + Torch + OpenCV)")

engine = AffectivePerceptionEngine(
    driver_src="data/sample_driver.mp4",
    road_src="data/sample_road.mp4",
    model_path="data/bigru_affective.pth",
    seq_len=30,
    skip_road_frames=2,
    target_resolution=(640, 480)
)
snap("+ AffectivePerceptionEngine (CV + BiGRU)")

agent = InterventionAgent(mode="rule_based")
snap("+ Rule-Based InterventionAgent")

audio = AudioAlertEngine(volume=0.0)
import time
time.sleep(0.5)  # allow background pygame mixer init
snap("+ AudioAlertEngine (Pre-loaded WAVs)")

peak_mb = proc.memory_info().rss / (1024 ** 2)
for i in range(100):
    d_f, r_f, payload = engine.step()
    if d_f is None:
        break
    if payload:
        _ = agent._rule_based_response(payload)
    if i % 10 == 0 or i == 99:
        cur = proc.memory_info().rss / (1024 ** 2)
        if cur > peak_mb:
            peak_mb = cur

engine.stop()
audio.stop()

peak_mb = round(max(peak_mb, proc.memory_info().rss / (1024 ** 2)), 2)
stages.append({"Stage": "Peak Active Video Processing (100f)", "RSS (MB)": peak_mb})
sys_ram = round(psutil.virtual_memory().total / (1024 ** 2), 1)
pct = round((peak_mb / sys_ram) * 100.0, 2)
stages.append({"Stage": "System Total RAM (MB)", "RSS (MB)": sys_ram})
stages.append({"Stage": "Peak Process % of System", "RSS (MB)": pct})
print(f"  [CLEAN MEM] {'Peak Active Video Processing (100f)':<42s}: {peak_mb:.2f} MB ({pct:.2f}%)")

df_mem = pd.DataFrame(stages)
df_mem.to_csv(os.path.join(RESULTS_DIR, "full_pipeline_memory.csv"), index=False)

# Update chart
plt.rcParams.update({
    "font.family": "serif", "font.size": 10.5,
    "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.labelweight": "bold", "figure.dpi": 300,
})
plot_mem = [r for r in stages if "System" not in r["Stage"]]
fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=300)
s_names = [r["Stage"] for r in plot_mem]
s_vals = [r["RSS (MB)"] for r in plot_mem]
colors = ["#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#1D3557"]
bars = ax.barh(range(len(s_names)), s_vals, color=colors[:len(s_names)], edgecolor="#222", lw=0.8, zorder=3)
ax.set_yticks(range(len(s_names)))
ax.set_yticklabels(s_names, fontsize=9, fontweight="bold")
ax.set_xlabel("Process RSS Memory (MB)")
ax.set_title("Production Affect-Drive RAM Footprint (Clean Single-Instance Stack)")
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

# Update full_evaluation_report.csv memory rows
rep_path = os.path.join(RESULTS_DIR, "full_evaluation_report.csv")
if os.path.exists(rep_path):
    rep = pd.read_csv(rep_path)
    # Cast the Value column to object dtype so it can hold both numbers and strings
    rep["Value"] = rep["Value"].astype(object)
    rep.loc[rep["Metric"] == "Peak RSS (MB)", "Value"] = peak_mb
    rep.loc[rep["Metric"] == "Process % System RAM", "Value"] = pct
    rep.to_csv(rep_path, index=False)
print("  [OK] Clean RAM profile & full_evaluation_report.csv updated.")
