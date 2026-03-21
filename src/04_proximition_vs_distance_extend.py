import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd


INPUT_DIR = "data/proximition_distance_split"
OUTPUT_DIR = "data/proximition_distance_stat"
DISTANCE_MAX_M = 500.0


def sample_files(n: int = 100) -> List[str]:
    files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(".csv")]
    if len(files) <= n:
        return sorted(files)
    return random.sample(files, n)


def compute_stats_extended(files: List[str], max_distance_m: float = DISTANCE_MAX_M) -> Tuple[float, int, float, int]:
    sum_ext_nonzero = 0.0
    cnt_ext_nonzero = 0
    sum_zero = 0.0
    cnt_zero = 0

    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "distance" not in df.columns:
            continue

        # Coerce and filter distance
        df["distance_num"] = pd.to_numeric(df["distance"], errors="coerce")
        dist = df["distance_num"].astype(float)
        mask = np.isfinite(dist) & (dist <= max_distance_m)
        sub = df[mask].copy()
        if sub.empty:
            continue

        # Prepare value and timestamps
        sub["value_num"] = pd.to_numeric(sub.get("value", 0), errors="coerce").fillna(0)
        sub["ts"] = pd.to_datetime(sub["timestamp"], errors="coerce")
        sub = sub.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        if sub.empty:
            continue

        # Original nonzero mask
        nz = sub["value_num"] != 0
        # Neighbor relationship only when exactly 5 minutes apart
        dt = sub["ts"]
        prev_is_5 = (dt - dt.shift(1)) == pd.Timedelta(minutes=5)
        next_is_5 = (dt.shift(-1) - dt) == pd.Timedelta(minutes=5)

        include_prev = nz.shift(1).fillna(False) & prev_is_5
        include_next = nz.shift(-1).fillna(False) & next_is_5
        ext_nz = nz | include_prev | include_next

        # Aggregate distances
        ext_group = sub.loc[ext_nz, "distance_num"]
        zero_group = sub.loc[~ext_nz, "distance_num"]

        sum_ext_nonzero += float(ext_group.sum())
        cnt_ext_nonzero += int(ext_group.shape[0])
        sum_zero += float(zero_group.sum())
        cnt_zero += int(zero_group.shape[0])

    mean_ext_nz = (sum_ext_nonzero / cnt_ext_nonzero) if cnt_ext_nonzero else float("nan")
    mean_zero = (sum_zero / cnt_zero) if cnt_zero else float("nan")
    return mean_ext_nz, cnt_ext_nonzero, mean_zero, cnt_zero


def main(n_files: int = 100) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sample_files(n_files)
    print(f"Sampled files: {len(files)}")

    mean_ext_nz, cnt_ext_nz, mean_z, cnt_z = compute_stats_extended(files, max_distance_m=DISTANCE_MAX_M)

    # Save results (extended)
    summary_path = os.path.join(OUTPUT_DIR, "summary_extend.csv")
    pd.DataFrame(
        [
            {"group": "value_nonzero_extended", "mean_distance": mean_ext_nz, "count": cnt_ext_nz},
            {"group": "value_zero_remaining", "mean_distance": mean_z, "count": cnt_z},
        ]
    ).to_csv(summary_path, index=False)

    sample_list_path = os.path.join(OUTPUT_DIR, "files_sampled_extend.txt")
    with open(sample_list_path, "w") as f:
        for p in sorted(files):
            f.write(os.path.basename(p) + "\n")

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {sample_list_path}")


if __name__ == "__main__":
    main(n_files=100)

