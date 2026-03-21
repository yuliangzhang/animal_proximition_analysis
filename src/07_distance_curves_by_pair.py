import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = "data/dating_analysis/proximition_distance_analysis_6_split.csv"
OUT_DIR = "data/dating_analysis/plot"


def main(limit_pairs: int = 0) -> None:
    df = pd.read_csv(INPUT_CSV)
    req = {"ram_eid", "ewe_eid", "distance_avg", "distance_cnt"}
    missing = req - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in input: {missing}")

    # Filter out rows with distance_cnt < 20 and non-finite distance_avg
    df = df[df["distance_cnt"] >= 20].copy()
    df["distance_avg_num"] = pd.to_numeric(df["distance_avg"], errors="coerce")
    df = df[np.isfinite(df["distance_avg_num"])]
    # New filter: remove entries where distance > 150 m
    df = df[df["distance_avg_num"] <= 150.0]

    # Group by pair and require at least 80 remaining rows per pair
    grouped = df.groupby(["ram_eid", "ewe_eid"], as_index=False)
    counts = grouped.size().rename(columns={"size": "n"})
    keep_pairs = set(map(tuple, counts[counts["n"] >= 80][["ram_eid", "ewe_eid"]].values.tolist()))

    if limit_pairs:
        # deterministically pick first N pairs
        keep_pairs = set(list(keep_pairs)[:limit_pairs])

    lines = 0
    plt.figure(figsize=(10, 6))
    for (ram, ewe), sub in df.groupby(["ram_eid", "ewe_eid"]):
        if (ram, ewe) not in keep_pairs:
            continue
        y = sub.sort_values("distance_avg_num")["distance_avg_num"].to_numpy()
        x = np.arange(1, len(y) + 1)
        plt.plot(x, y, alpha=0.35, linewidth=1)
        lines += 1

    plt.xlabel("Rank (by distance ascending)")
    plt.ylabel("Average distance (m)")
    plt.title(f"Distance curves (all pairs meeting filters) | pairs={lines}")
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "curves_all_pairs.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Plotted pairs: {lines}, Output: {out_path}")


if __name__ == "__main__":
    # Plot all qualifying pairs on a single figure
    main(limit_pairs=0)
