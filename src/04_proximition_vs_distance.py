import os
import random
from typing import List, Tuple

import pandas as pd
import numpy as np


INPUT_DIR = "data/proximition_distance_split"
OUTPUT_DIR = "data/proximition_distance_stat"
# Exclude distances greater than this threshold (meters)
DISTANCE_MAX_M = 500.0


def sample_files(n: int = 100) -> List[str]:
    files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(".csv")]
    if len(files) <= n:
        return sorted(files)
    return random.sample(files, n)


def compute_stats(files: List[str], max_distance_m: float = DISTANCE_MAX_M) -> Tuple[float, int, float, int]:
    sum_nonzero = 0.0
    cnt_nonzero = 0
    sum_zero = 0.0
    cnt_zero = 0

    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "distance" not in df.columns:
            continue

        # Coerce types
        df["distance_num"] = pd.to_numeric(df["distance"], errors="coerce")
        df["value_num"] = pd.to_numeric(df.get("value", 0), errors="coerce").fillna(0)

        # Keep only finite distances (exclude NaN and +/-inf)
        dist = df["distance_num"].astype(float)
        mask = np.isfinite(dist)
        sub = df[mask]
        # Apply max distance filter
        sub = sub[sub["distance_num"] <= max_distance_m]
        if sub.empty:
            continue

        sub_zero = sub[sub["value_num"] == 0]
        sub_nz = sub[sub["value_num"] != 0]

        sum_zero += float(sub_zero["distance_num"].sum())
        cnt_zero += int(sub_zero.shape[0])
        sum_nonzero += float(sub_nz["distance_num"].sum())
        cnt_nonzero += int(sub_nz.shape[0])

    mean_nonzero = (sum_nonzero / cnt_nonzero) if cnt_nonzero else float("nan")
    mean_zero = (sum_zero / cnt_zero) if cnt_zero else float("nan")
    return mean_nonzero, cnt_nonzero, mean_zero, cnt_zero


def main(n_files: int = 100) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sample_files(n_files)
    print(f"Sampled files: {len(files)}")

    mean_nz, cnt_nz, mean_z, cnt_z = compute_stats(files, max_distance_m=DISTANCE_MAX_M)

    # Save results
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    pd.DataFrame(
        [
            {"group": "value_nonzero", "mean_distance": mean_nz, "count": cnt_nz},
            {"group": "value_zero", "mean_distance": mean_z, "count": cnt_z},
        ]
    ).to_csv(summary_path, index=False)

    sample_list_path = os.path.join(OUTPUT_DIR, "files_sampled.txt")
    with open(sample_list_path, "w") as f:
        for p in sorted(files):
            f.write(os.path.basename(p) + "\n")

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {sample_list_path}")


if __name__ == "__main__":
    main(n_files=100)
