"""
plot_results.py
Generates presentation-ready comparison figures from evaluation metrics.
Saves to: outputs/evaluation_benchmark.png
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_benchmark_figures(json_path="outputs/benchmark_results.json"):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist. Run evaluate_benchmark.py first.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    tiers = ["Tier_0_Nominal", "Tier_1_Low", "Tier_2_Medium", "Tier_3_High"]
    tier_labels = ["Nominal\n(0 cm, 0°)", "Low Error\n(±1 cm, ±5°)", "Medium Error\n(±2.5 cm, ±15°)", "Stress Test\n(±4 cm, ±30°)"]
    
    policies = list(data.keys())
    colors = {"bc": "#E74C3C", "act": "#3498DB", "diffusion": "#2ECC71"}
    markers = {"bc": "o", "act": "s", "diffusion": "^"}

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # -------------------------------------------------------------
    # Plot 1: Robustness Degradation Curves (The Core Research Plot)
    # -------------------------------------------------------------
    for pol in policies:
        rates = [data[pol].get(t, 0.0) for t in tiers]
        ax1.plot(
            range(len(tiers)),
            rates,
            label=f"{pol.upper()}",
            color=colors.get(pol, "#333333"),
            marker=markers.get(pol, "o"),
            linewidth=2.5,
            markersize=8
        )

    ax1.set_title("Visuomotor Policy Robustness Under Initial-Pose Error", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Perturbation Severity Tier", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Task Success Rate (%)", fontsize=10, fontweight="bold")
    ax1.set_xticks(range(len(tiers)))
    ax1.set_xticklabels(tier_labels, fontsize=9)
    ax1.set_ylim(-5, 105)
    ax1.legend(title="Policy Architecture", frameon=True, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # -------------------------------------------------------------
    # Plot 2: Grouped Bar Chart Comparison
    # -------------------------------------------------------------
    x = np.arange(len(tiers))
    width = 0.25
    multiplier = 0

    for pol in policies:
        rates = [data[pol].get(t, 0.0) for t in tiers]
        offset = width * multiplier
        rects = ax2.bar(x + offset, rates, width, label=pol.upper(), color=colors.get(pol, "#333333"), alpha=0.9)
        ax2.bar_label(rects, padding=3, fmt="%.0f%%", fontsize=8)
        multiplier += 1

    ax2.set_title("Success Rate by Architecture across Tiers", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Perturbation Severity Tier", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Success Rate (%)", fontsize=10, fontweight="bold")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(tier_labels, fontsize=9)
    ax2.set_ylim(0, 110)
    ax2.legend(title="Architecture", frameon=True, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_fig = "outputs/evaluation_benchmark.png"
    plt.savefig(output_fig, bbox_inches="tight")
    print(f"[DONE] Benchmark figures generated and saved to: {output_fig}")
    plt.show()

if __name__ == "__main__":
    generate_benchmark_figures()