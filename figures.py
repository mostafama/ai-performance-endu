"""
figures.py
----------
Regenerates Figures 2, 3, and 4 from data/scores.csv, using the arithmetic-mean
Overall (Section 3.5). Writes PNGs to figures/.

Usage:
    python figures.py
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIMS = ["correctness", "completeness", "clarity", "cognitive_alignment"]
HO = [4, 5, 6]
ORDER = ["claude-sonnet-4.5", "deepseek-v3.2", "gemini-2.5-flash",
         "gpt-4o-mini", "llama-3.3-70b", "llama-3.1-8b"]
LABEL = dict(zip(ORDER, ["Claude S4.5", "DeepSeek V3.2", "Gemini 2.5F",
                         "GPT-4o-mini", "LLaMA 3.3 70B", "LLaMA 3.1 8B"]))
COLOR = dict(zip(ORDER, ["#2E86C1", "#E74C3C", "#27AE60",
                         "#F39C12", "#8E44AD", "#5D6D7E"]))


def load(path):
    df = pd.read_csv(path)
    if "error" in df.columns:
        df = df[df["error"] != True]  # noqa: E712
    df = df.dropna(subset=DIMS)
    df["overall"] = df[DIMS].mean(axis=1)
    return df


def fig2(df, out):
    diffs = ["Easy", "Medium", "Hard"]
    x = np.arange(3)
    w = 0.13
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(ORDER):
        s = df[df["model_name"] == m].groupby("difficulty")["overall"].mean()
        vals = [s[d] for d in diffs]
        bars = ax.bar(x + (i - 2.5) * w, vals, w, label=LABEL[m], color=COLOR[m])
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.02,
                    f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=5.2)
    ax.set_xticks(x)
    ax.set_xticklabels(diffs)
    ax.set_ylim(6, 10.2)
    ax.set_ylabel("Mean Score (0-10)")
    ax.set_xlabel("Difficulty Level")
    ax.set_title("Model Performance by Difficulty Level")
    ax.legend(ncol=3, fontsize=7, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def fig3(df, out):
    xt = ["L1\nRemember", "L2\nUnderstand", "L3\nApply", "L4-6\nHigher-Order"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in ORDER:
        lv = df[df["model_name"] == m].groupby("bloom_level")["overall"].mean()
        vals = [lv[1], lv[2], lv[3], lv.loc[HO].mean()]
        ax.plot(range(4), vals, marker="o", label=LABEL[m], color=COLOR[m], linewidth=1.8)
    ax.axhline(5.0, ls="--", color="red", alpha=0.7, label="Midpoint (5.0)")
    ax.set_xticks(range(4))
    ax.set_xticklabels(xt)
    ax.set_ylim(4, 10.3)
    ax.set_ylabel("Mean Overall Score (0-10)")
    ax.set_title("Model Performance Across Bloom's Cognitive Levels")
    ax.legend(ncol=2, fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def fig4(df, out):
    d = df.copy()
    d["bg"] = np.where(d["bloom_level"].isin(HO), "HO",
                       "L" + d["bloom_level"].astype(int).astype(str))
    piv = (d.pivot_table(index="difficulty", columns="bg", values="overall", aggfunc="mean")
           .reindex(["Easy", "Medium", "Hard"])[["L1", "L2", "L3", "HO"]])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=6.5, vmax=10, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["L1 Remember", "L2 Understand", "L3 Apply", "L4-6 Higher-Order"])
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Easy", "Medium", "Hard"])
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontweight="bold")
    ax.set_title("Difficulty x Bloom's Level Interaction (All Models)")
    plt.colorbar(im, label="Mean Score (0-10)")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/scores.csv")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    import os
    os.makedirs(args.outdir, exist_ok=True)
    df = load(args.scores)
    fig2(df, f"{args.outdir}/fig2_difficulty.png")
    fig3(df, f"{args.outdir}/fig3_bloom.png")
    fig4(df, f"{args.outdir}/fig4_interaction.png")
    print(f"Wrote fig2_difficulty.png, fig3_bloom.png, fig4_interaction.png to {args.outdir}/")


if __name__ == "__main__":
    main()