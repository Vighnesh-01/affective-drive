import os
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# Global Publication Styling Parameters (IEEE / Elsevier / Nature Standards)
# ==============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# Chart 1: Input Configuration Performance (Grouped Bar Chart)
# ==============================================================================
def plot_input_configuration_performance():
    metrics = ["Accuracy", "UAR", "WAR", "F1 Score"]
    
    # Quantitative benchmark data (%)
    configs = {
        "Facial Only": [78.6, 76.4, 78.6, 77.2],
        "Traffic Only": [65.2, 63.1, 65.2, 63.8],
        "Basic Fusion": [84.9, 83.5, 84.9, 84.1],
        "Proposed Model": [94.2, 93.1, 94.2, 93.6]
    }

    # Academic Color Palette (Color-blind safe, high-contrast print-friendly)
    palette = {
        "Facial Only": "#457B9D",      # Muted Steel Blue
        "Traffic Only": "#E76F51",     # Terracotta / Coral
        "Basic Fusion": "#D4A373",     # Ochre / Warm Taupe
        "Proposed Model": "#1D3557"    # Deep Navy Blue (Primary highlight)
    }

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=300)

    n_metrics = len(metrics)
    n_configs = len(configs)
    
    bar_width = 0.18
    group_spacing = 0.95
    indices = np.arange(n_metrics) * group_spacing

    # Offsets centered around 0
    offsets = np.linspace(-(n_configs - 1) / 2 * bar_width, (n_configs - 1) / 2 * bar_width, n_configs)

    for i, (name, values) in enumerate(configs.items()):
        is_proposed = name == "Proposed Model"
        color = palette[name]
        edge_color = "#0B1D3A" if is_proposed else "#333333"
        alpha = 1.0 if is_proposed else 0.88
        linewidth = 1.2 if is_proposed else 0.8
        hatch = "///" if is_proposed else None

        bars = ax.bar(
            indices + offsets[i],
            values,
            width=bar_width,
            label=name,
            color=color,
            edgecolor=edge_color,
            linewidth=linewidth,
            alpha=alpha,
            hatch=hatch,
            zorder=3
        )

        # Value labels above bars
        for bar in bars:
            height = bar.get_height()
            fontweight = "bold" if is_proposed else "normal"
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight=fontweight,
                color="#0E1C2E" if is_proposed else "#333333",
                rotation=0
            )

    # Aesthetics and styling
    ax.set_title("Input Configuration Performance Comparison on Driver Affect State", pad=14)
    ax.set_ylabel("Performance (%)", labelpad=8)
    ax.set_xticks(indices)
    ax.set_xticklabels(metrics, fontweight="bold")
    ax.set_ylim(50, 102)
    ax.set_yticks(np.arange(50, 105, 10))

    # Grid & Spines
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="#A0AAB2", zorder=0)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")

    # Legend
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#D0D0D0",
        framealpha=0.95
    )
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "input_configuration_performance.png")
    pdf_path = os.path.join(OUTPUT_DIR, "input_configuration_performance.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Chart 1 to: {png_path} and {pdf_path}")


# ==============================================================================
# Chart 2: Ablation Study (Grouped Bar + Progression Curve)
# ==============================================================================
def plot_ablation_study():
    stages = [
        "Face\nOnly",
        "Traffic\nOnly",
        "Face +\nTraffic",
        "+ Temporal\nModeling",
        "+ Cross-Modal\nFusion",
        "Full Model\n(Proposed)"
    ]

    # Metrics across progression stages (%)
    acc_scores = [78.6, 65.2, 82.4, 88.7, 91.8, 94.2]
    f1_scores  = [77.2, 63.8, 81.6, 88.1, 91.2, 93.6]
    uar_scores = [76.4, 63.1, 80.9, 87.5, 90.7, 93.1]

    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=300)

    n_stages = len(stages)
    x = np.arange(n_stages)
    bar_width = 0.24

    # Academic Palette: Navy / Teal / Slate
    c_acc = "#2B5C8F"   # Classic Academic Navy
    c_f1  = "#2A9D8F"   # Teal / Sage
    c_uar = "#E76F51"   # Rust / Warm Amber

    bars_acc = ax.bar(x - bar_width, acc_scores, width=bar_width, label="Accuracy", color=c_acc, edgecolor="#1B365D", linewidth=0.8, alpha=0.9, zorder=3)
    bars_f1  = ax.bar(x, f1_scores, width=bar_width, label="F1 Score", color=c_f1, edgecolor="#1D6A61", linewidth=0.8, alpha=0.9, zorder=3)
    bars_uar = ax.bar(x + bar_width, uar_scores, width=bar_width, label="UAR", color=c_uar, edgecolor="#9E3B27", linewidth=0.8, alpha=0.9, zorder=3)

    # Overlay Line Progression for Overall Accuracy
    line = ax.plot(
        x - bar_width,
        acc_scores,
        color="#0D233A",
        marker="o",
        markersize=6,
        linewidth=1.8,
        linestyle="-.",
        label="Accuracy Trajectory",
        zorder=5
    )

    # Value labels for bars
    def add_labels(bars, text_color, is_bold=False):
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            fontweight = "bold" if (is_bold or idx == n_stages - 1) else "normal"
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight=fontweight,
                color=text_color
            )

    add_labels(bars_acc, "#0F2D4A")
    add_labels(bars_f1,  "#124741")
    add_labels(bars_uar, "#6E2416")

    # Step gain callouts (highlighting impact of adding components from Face+Traffic -> Full)
    gains = [
        (2, 3, "+6.3% (Temporal BiGRU)", 88.7),
        (3, 4, "+3.1% (Cross-Modal)", 91.8),
        (4, 5, "+2.4% (Tuned Fusion)", 94.2)
    ]
    for start_idx, end_idx, text, peak_y in gains:
        mid_x = (x[start_idx] + x[end_idx]) / 2 - bar_width
        ax.annotate(
            text,
            xy=(x[end_idx] - bar_width, peak_y),
            xytext=(mid_x - 0.25, peak_y + 4.8),
            arrowprops=dict(arrowstyle="->", color="#1D3557", lw=1.0, shrinkA=3, shrinkB=3),
            fontsize=8.5,
            fontweight="bold",
            color="#1D3557",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F1FAEE", edgecolor="#457B9D", lw=0.7, alpha=0.95),
            zorder=6
        )

    # Aesthetics
    ax.set_title("Step-by-Step Architectural Ablation Study on System Performance", pad=14)
    ax.set_ylabel("Evaluation Metric (%)", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontweight="bold")
    ax.set_ylim(55, 105)
    ax.set_yticks(np.arange(60, 105, 10))

    # Grid & Spines
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="#A0AAB2", zorder=0)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")

    # Legend
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#D0D0D0",
        framealpha=0.95
    )
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "ablation_study.png")
    pdf_path = os.path.join(OUTPUT_DIR, "ablation_study.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Chart 2 to: {png_path} and {pdf_path}")


if __name__ == "__main__":
    print("Generating publication-quality figures for Affect-Drive...")
    plot_input_configuration_performance()
    plot_ablation_study()
    print("\nAll figures successfully generated in 300 DPI PNG & Vector PDF formats!")
