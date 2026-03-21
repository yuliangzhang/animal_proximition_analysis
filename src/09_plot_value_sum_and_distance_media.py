import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = "data/dating_analysis/proximition_distance_analysis_6_split.csv"
OUT_DIR = "data/dating_analysis/proximition_distance_media_plot"


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", s)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"ram_eid", "ewe_eid", "time_split", "value_sum", "distance_media", "distance_cnt"}
    missing = req - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in input CSV: {missing}")

    # Ensure numeric types
    df["value_sum"] = pd.to_numeric(df["value_sum"], errors="coerce").fillna(0)
    df["distance_cnt"] = pd.to_numeric(df["distance_cnt"], errors="coerce").fillna(0).astype(int)
    # Convert distance_media: 'INF' -> NaN; numeric otherwise
    df["distance_media_num"] = pd.to_numeric(df["distance_media"], errors="coerce")
    # Valid distance points: finite, <=150m, and distance_cnt >= 20
    mask_valid_dist = (
        np.isfinite(df["distance_media_num"]) &
        (df["distance_media_num"] <= 150.0) &
        (df["distance_cnt"] >= 20)
    )
    df["distance_media_valid"] = mask_valid_dist
    return df


def plot_pair(ram: str, ewe: str, sub: pd.DataFrame) -> str:
    # Sort by time_split lexicographically (matches chronological for this format)
    sub = sub.sort_values("time_split").reset_index(drop=True)
    x_labels = sub["time_split"].tolist()
    x = np.arange(len(x_labels))

    y_val = sub["value_sum"].to_numpy()

    # For distance media, only plot valid points
    y_dist = sub["distance_media_num"].to_numpy()
    valid = sub["distance_media_valid"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax1.plot(
        x,
        y_val,
        color="#1f77b4",
        linewidth=1.2,
        marker="o",
        markersize=3,
        markerfacecolor="#1f77b4",
        markeredgewidth=0.0,
    )
    ax1.set_ylabel("value_sum")
    ax1.set_title(f"Pair: {ram} vs {ewe}")

    # Plot only valid distance_media points
    ax2.plot(
        x[valid],
        y_dist[valid],
        color="#ff7f0e",
        linewidth=1.2,
        marker="o",
        markersize=3,
        markerfacecolor="#ff7f0e",
        markeredgewidth=0.0,
    )
    ax2.set_ylabel("distance_median (m)")
    ax2.set_xlabel("time_split")

    # Limit x tick labels for readability (max ~30)
    step = max(1, len(x_labels) // 30)
    ticks = np.arange(0, len(x_labels), step)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([x_labels[i] for i in ticks], rotation=90)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fname = f"trend_{sanitize(ram)}_{sanitize(ewe)}.png"
    out_path = os.path.join(OUT_DIR, fname)
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def main(limit_pairs: int = 50) -> None:
    df = load_data(INPUT_CSV)
    # Select first N unique pairs deterministically
    pairs = df[["ram_eid", "ewe_eid"]].drop_duplicates().head(limit_pairs).values.tolist()

    outs: List[str] = []
    for ram, ewe in pairs:
        sub = df[(df["ram_eid"] == ram) & (df["ewe_eid"] == ewe)]
        if sub.empty:
            continue
        out = plot_pair(ram, ewe, sub)
        outs.append(out)

    print(f"Plotted {len(outs)} pairs to {OUT_DIR}")


if __name__ == "__main__":
    main(limit_pairs=50)
