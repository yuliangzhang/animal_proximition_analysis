import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = "data/dating_analysis/proximition_distance_analysis_6_split.csv"
OUT_DIR = "data/dating_analysis/proximition_distance_relation_plot"


def load_and_filter(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"group", "time_split", "value_sum", "distance_cnt", "distance_media"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in input CSV: {missing}")

    # Coerce types
    df["value_sum"] = pd.to_numeric(df["value_sum"], errors="coerce")
    df["distance_cnt"] = pd.to_numeric(df["distance_cnt"], errors="coerce")
    df["distance_media_num"] = pd.to_numeric(df["distance_media"], errors="coerce")

    # Base filters
    base = (
        df["value_sum"].notna()
        & df["distance_cnt"].notna()
        & df["distance_media_num"].notna()
        & (df["value_sum"] > 75)
        & (df["distance_cnt"] > 24)
    )
    df = df[base].copy()

    # Optional: clamp extreme distances for more stable fits
    df = df[df["distance_media_num"] <= 150.0]

    # Extract 2-digit time bin from time_split (YYYY-MM-DD_0i)
    df["time_bin"] = df["time_split"].astype(str).str[-2:]
    return df


def plot_group(df: pd.DataFrame, group_label: str, colors: List[str]) -> str:
    sub = df[df["group"] == group_label]
    if sub.empty:
        return ""

    plt.figure(figsize=(10, 7))
    bins = ["01", "02", "03", "04", "05", "06"]

    for i, b in enumerate(bins):
        sb = sub[sub["time_bin"] == b]
        if sb.empty:
            continue
        x = sb["value_sum"].to_numpy()
        y = sb["distance_media_num"].to_numpy()
        c = colors[i % len(colors)]
        # Scatter
        plt.scatter(x, y, s=14, alpha=0.5, color=c, label=f"bin {b} (n={len(x)})")
        # Linear fit
        if len(x) >= 2:
            coef = np.polyfit(x, y, 1)
            xx = np.linspace(float(np.min(x)), float(np.max(x)), 50)
            yy = coef[0] * xx + coef[1]
            plt.plot(xx, yy, color=c, linewidth=2)

    plt.xlabel("value_sum")
    plt.ylabel("distance_median (m)")
    names = {"G": "High Shade", "Y": "Medium Shade", "P": "No Shade"}
    title = names.get(group_label, group_label)
    plt.title(f"Group {group_label} ({title}) — 6 time bins (linear fit)")
    plt.legend(frameon=False, fontsize=9, ncol=2)
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"relation_group_{group_label}.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def main() -> None:
    df = load_and_filter(INPUT_CSV)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    created = []
    for g in ["G", "Y", "P"]:
        out = plot_group(df, g, palette)
        if out:
            created.append(out)

    print("Created plots:")
    for p in created:
        print(" -", p)


if __name__ == "__main__":
    main()

