"""
Plot dictionary similarity trajectories from similarity_study.py results.

Generates a multi-panel figure showing:
1. Model vs its own initialization (with noise)
2. Model vs clean k-means centers (without noise)
3. Pairwise similarity: same init + same noise (A vs B)
4. Pairwise similarity: same init + different noise (A vs C, B vs C)

Usage:
    python plot_similarity.py path/to/similarity_study_results.json [--output fig.pdf]
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_similarity(results, output_path):
    config = results["config"]
    logs_a = results["model_A"]
    logs_b = results["model_B"]
    logs_c = results["model_C"]

    epochs_a = list(range(1, len(logs_a.get("sim_to_init", [])) + 1))
    epochs_b = list(range(1, len(logs_b.get("sim_to_init", [])) + 1))
    epochs_c = list(range(1, len(logs_c.get("sim_to_init", [])) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Dictionary Similarity During Training\n"
        f"(d_model={config['d_model']}, k={config['k']}, per_init={config['per_init']})",
        fontsize=14, fontweight='bold'
    )

    colors = {
        'A': '#1f77b4',
        'B': '#ff7f0e',
        'C': '#2ca02c',
    }

    # ── Panel 1: Model vs Init (with noise) ─────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs_a, logs_a["sim_to_init"], color=colors['A'],
            linewidth=2, label="A (seed=42)")
    ax.plot(epochs_b, logs_b["sim_to_init"], color=colors['B'],
            linewidth=2, label="B (seed=42)")
    ax.plot(epochs_c, logs_c["sim_to_init"], color=colors['C'],
            linewidth=2, label="C (seed=43)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Max Cosine Similarity")
    ax.set_title("Model vs Its Own Init (with noise)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # ── Panel 2: Model vs Clean Centers (without noise) ─────────────────────
    ax = axes[0, 1]
    ax.plot(epochs_a, logs_a["sim_to_clean_centers"], color=colors['A'],
            linewidth=2, label="A (seed=42)")
    ax.plot(epochs_b, logs_b["sim_to_clean_centers"], color=colors['B'],
            linewidth=2, label="B (seed=42)")
    ax.plot(epochs_c, logs_c["sim_to_clean_centers"], color=colors['C'],
            linewidth=2, label="C (seed=43)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Max Cosine Similarity")
    ax.set_title("Model vs Clean Centers (no noise)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # ── Panel 3: Same init + same noise (A ↔ B) ────────────────────────────
    ax = axes[1, 0]
    # B tracked similarity to A during its training
    if "sim_to_A" in logs_b:
        ax.plot(epochs_b, logs_b["sim_to_A"], color='#d62728',
                linewidth=2, label="B ↔ A (same init, same noise)")
    ax.set_xlabel("Epoch (of model B)")
    ax.set_ylabel("Mean Max Cosine Similarity")
    ax.set_title("Pairwise: Same Init + Same Noise (A ↔ B)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # ── Panel 4: Same init + different noise (C ↔ A, C ↔ B) ────────────────
    ax = axes[1, 1]
    if "sim_to_A" in logs_c:
        ax.plot(epochs_c, logs_c["sim_to_A"], color='#9467bd',
                linewidth=2, label="C ↔ A (different noise)")
    if "sim_to_B" in logs_c:
        ax.plot(epochs_c, logs_c["sim_to_B"], color='#8c564b',
                linewidth=2, label="C ↔ B (different noise)")
    ax.set_xlabel("Epoch (of model C)")
    ax.set_ylabel("Mean Max Cosine Similarity")
    ax.set_title("Pairwise: Same Init + Different Noise (C ↔ A/B)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    plt.close()

    # ── Also generate a combined overlay plot ────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.set_title(
        f"All Similarity Trajectories\n"
        f"(d_model={config['d_model']}, k={config['k']}, per_init={config['per_init']})",
        fontsize=13, fontweight='bold'
    )

    # Self-similarity to init
    ax2.plot(epochs_a, logs_a["sim_to_init"], color=colors['A'],
             linewidth=2, linestyle='-', label="A → own init")
    ax2.plot(epochs_b, logs_b["sim_to_init"], color=colors['B'],
             linewidth=2, linestyle='-', label="B → own init")
    ax2.plot(epochs_c, logs_c["sim_to_init"], color=colors['C'],
             linewidth=2, linestyle='-', label="C → own init")

    # To clean centers
    ax2.plot(epochs_a, logs_a["sim_to_clean_centers"], color=colors['A'],
             linewidth=2, linestyle='--', label="A → clean centers")
    ax2.plot(epochs_b, logs_b["sim_to_clean_centers"], color=colors['B'],
             linewidth=2, linestyle='--', label="B → clean centers")
    ax2.plot(epochs_c, logs_c["sim_to_clean_centers"], color=colors['C'],
             linewidth=2, linestyle='--', label="C → clean centers")

    # Pairwise
    if "sim_to_A" in logs_b:
        ax2.plot(epochs_b, logs_b["sim_to_A"], color='#d62728',
                 linewidth=2.5, linestyle=':', label="B ↔ A (same noise)")
    if "sim_to_A" in logs_c:
        ax2.plot(epochs_c, logs_c["sim_to_A"], color='#9467bd',
                 linewidth=2.5, linestyle=':', label="C ↔ A (diff noise)")

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Mean Max Cosine Similarity", fontsize=12)
    ax2.legend(fontsize=9, ncol=2, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    overlay_path = output_path.replace('.pdf', '_overlay.pdf').replace('.png', '_overlay.png')
    if overlay_path == output_path:
        overlay_path = output_path + '_overlay.pdf'
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
    print(f"Overlay figure saved to {overlay_path}")
    plt.close()

    # ── Print summary statistics ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, logs in [("A", logs_a), ("B", logs_b), ("C", logs_c)]:
        if not logs.get("sim_to_init"):
            continue
        print(f"\nModel {name}:")
        print(f"  Sim to init:    {logs['sim_to_init'][0]:.4f} → {logs['sim_to_init'][-1]:.4f}")
        print(f"  Sim to clean:   {logs['sim_to_clean_centers'][0]:.4f} → {logs['sim_to_clean_centers'][-1]:.4f}")
        print(f"  Final loss:     {logs['avg_loss'][-1]:.4f}")

    if "sim_to_A" in logs_b:
        print(f"\nPairwise (same noise):")
        print(f"  B↔A: {logs_b['sim_to_A'][0]:.4f} → {logs_b['sim_to_A'][-1]:.4f}")
    if "sim_to_A" in logs_c:
        print(f"\nPairwise (different noise):")
        print(f"  C↔A: {logs_c['sim_to_A'][0]:.4f} → {logs_c['sim_to_A'][-1]:.4f}")
    if "sim_to_B" in logs_c:
        print(f"  C↔B: {logs_c['sim_to_B'][0]:.4f} → {logs_c['sim_to_B'][-1]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot similarity study results")
    parser.add_argument("results_json", type=str,
                        help="Path to similarity_study_results.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output figure path (default: same dir as results, .pdf)")
    args = parser.parse_args()

    results = load_results(args.results_json)

    if args.output is None:
        import os
        base_dir = os.path.dirname(args.results_json)
        args.output = os.path.join(base_dir, "similarity_study.pdf")

    plot_similarity(results, args.output)


if __name__ == "__main__":
    main()
