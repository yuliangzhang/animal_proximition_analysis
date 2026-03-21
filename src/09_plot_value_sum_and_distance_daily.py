import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = "data/dating_analysis/proximition_distance_analysis_daily_split.csv"
OUT_DIR_MEDIA = "data/dating_analysis/daily_compare/proximition_distance_media_plot"
OUT_DIR_AVG = "data/dating_analysis/daily_compare/proximition_distance_avg_plot"
OUT_DIR_Q1 = "data/dating_analysis/daily_compare/proximition_distance_q1_plot"


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", s)


def load_daily(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {
        "ram_eid",
        "ewe_eid",
        "time_split",
        "value_sum",
        "distance_cnt",
        "distance_media",
        "distance_avg",
        "distance_q1",
    }
    missing = req - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in input CSV: {missing}")

    df["value_sum"] = pd.to_numeric(df["value_sum"], errors="coerce").fillna(0)
    df["distance_cnt"] = pd.to_numeric(df["distance_cnt"], errors="coerce").fillna(0).astype(int)
    df["distance_media_num"] = pd.to_numeric(df["distance_media"], errors="coerce")
    df["distance_avg_num"] = pd.to_numeric(df["distance_avg"], errors="coerce")
    df["distance_q1_num"] = pd.to_numeric(df["distance_q1"], errors="coerce")
    return df


def select_top_pairs(df: pd.DataFrame, limit_pairs: int = 50) -> List[Tuple[str, str]]:
    # Quality rows: distance_cnt >= 150 and finite distances within 150 m (for either metric)
    valid_media = np.isfinite(df["distance_media_num"]) & (df["distance_media_num"] <= 150.0)
    valid_avg = np.isfinite(df["distance_avg_num"]) & (df["distance_avg_num"] <= 150.0)
    quality = (df["distance_cnt"] >= 150) & (valid_media | valid_avg)
    qual_df = df[quality]
    counts = (
        qual_df.groupby(["ram_eid", "ewe_eid"]).size().reset_index(name="qual_n")
    )
    counts = counts.sort_values(["qual_n"], ascending=False)
    pairs = counts[["ram_eid", "ewe_eid"]].head(limit_pairs).values.tolist()
    return [tuple(p) for p in pairs]


def plot_pair(df: pd.DataFrame, ram: str, ewe: str, out_dir: str, which: str) -> str:
    # which in {"media", "avg", "q1"}
    os.makedirs(out_dir, exist_ok=True)
    sub = df[(df["ram_eid"] == ram) & (df["ewe_eid"] == ewe)].copy()
    if sub.empty:
        return ""
    sub = sub.sort_values("time_split").reset_index(drop=True)

    x_labels = sub["time_split"].tolist()
    x = np.arange(len(x_labels))
    y_val = sub["value_sum"].to_numpy()

    if which == "media":
        y_dist = sub["distance_media_num"].to_numpy()
        valid = (
            np.isfinite(y_dist) & (y_dist <= 150.0) & (sub["distance_cnt"].to_numpy() >= 150)
        )
        color = "#ff7f0e"
        ylabel = "distance_median (m)"
        suffix = "media"
    elif which == "avg":
        y_dist = sub["distance_avg_num"].to_numpy()
        valid = (
            np.isfinite(y_dist) & (y_dist <= 150.0) & (sub["distance_cnt"].to_numpy() >= 150)
        )
        color = "#2ca02c"
        ylabel = "distance_avg (m)"
        suffix = "avg"
    else:
        y_dist = sub["distance_q1_num"].to_numpy()
        valid = (
            np.isfinite(y_dist) & (y_dist <= 150.0) & (sub["distance_cnt"].to_numpy() >= 150)
        )
        color = "#9467bd"
        ylabel = "distance_q1 (m)"
        suffix = "q1"

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
    ax1.set_ylabel("value_sum (daily)")
    ax1.set_title(f"Pair: {ram} vs {ewe}")

    ax2.plot(
        x[valid],
        y_dist[valid],
        color=color,
        linewidth=1.2,
        marker="o",
        markersize=3,
        markerfacecolor=color,
        markeredgewidth=0.0,
    )
    ax2.set_ylabel(ylabel)
    ax2.set_xlabel("date")

    step = max(1, len(x_labels) // 30)
    ticks = np.arange(0, len(x_labels), step)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([x_labels[i] for i in ticks], rotation=90)

    plt.tight_layout()
    fname = f"trend_daily_{suffix}_{sanitize(ram)}_{sanitize(ewe)}.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def main(limit_pairs: int = 50) -> None:
    df = load_daily(INPUT_CSV)
    pairs = select_top_pairs(df, limit_pairs=limit_pairs)

    outs_media: List[str] = []
    outs_avg: List[str] = []
    for ram, ewe in pairs:
        p1 = plot_pair(df, ram, ewe, OUT_DIR_MEDIA, which="media")
        if p1:
            outs_media.append(p1)
        p2 = plot_pair(df, ram, ewe, OUT_DIR_AVG, which="avg")
        if p2:
            outs_avg.append(p2)
        p3 = plot_pair(df, ram, ewe, OUT_DIR_Q1, which="q1")
        

    print(f"Pairs plotted: {len(pairs)}")
    print(f"Media plots: {len(outs_media)} saved to {OUT_DIR_MEDIA}")
    print(f"Avg plots: {len(outs_avg)} saved to {OUT_DIR_AVG}")
    print(f"Q1 plots saved to {OUT_DIR_Q1}")


if __name__ == "__main__":
    main(limit_pairs=50)
