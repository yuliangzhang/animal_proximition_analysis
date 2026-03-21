import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


INPUT_DIR = "data/proximition_distance_split_full"
OUTPUT_DIR = "data/dating_analysis"
OUTPUT_FILE = "proximition_distance_analysis_daily_split.csv"
DISTANCE_MAX_M = 500.0


def parse_filename(fname: str):
    base = os.path.basename(fname)
    if base.lower().endswith(".csv"):
        base = base[:-4]
    parts = base.split("_")
    if len(parts) < 5:
        return None
    group = parts[0]
    ram_eid = parts[3]
    ewe_eid = parts[4]
    return group, ram_eid, ewe_eid


def process_one_file(path: str) -> List[Dict[str, object]]:
    parsed = parse_filename(path)
    if not parsed:
        return []
    group, ram_eid, ewe_eid = parsed

    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if df.empty or "timestamp" not in df.columns:
        return []

    # Prepare timestamps and values
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["ts"]).copy()
    df["date"] = df["ts"].dt.date.astype(str)

    # Value as numeric
    df["value_num"] = pd.to_numeric(df.get("value", 0), errors="coerce").fillna(0)

    # Distance as numeric
    df["distance_num"] = pd.to_numeric(df.get("distance"), errors="coerce")

    results: List[Dict[str, object]] = []
    for date_key, grp in df.groupby("date", sort=True):
        value_sum = float(grp["value_num"].sum())

        # Filter valid distances for stats
        dist = grp["distance_num"].astype(float)
        mask = np.isfinite(dist) & (dist <= DISTANCE_MAX_M)
        dist_valid = dist[mask]
        distance_cnt = int(dist_valid.shape[0])
        if distance_cnt > 0:
            distance_avg = float(dist_valid.mean())
            distance_media = float(dist_valid.median())
            distance_q1 = float(dist_valid.quantile(0.25))
        else:
            distance_avg = "INF"
            distance_media = "INF"
            distance_q1 = "INF"

        results.append(
            {
                "ram_eid": ram_eid,
                "ewe_eid": ewe_eid,
                "group": group,
                # Keep column name as time_split for compatibility; value is date string
                "time_split": date_key,
                "value_sum": value_sum,
                "distance_avg": distance_avg,
                "distance_cnt": distance_cnt,
                "distance_media": distance_media,
                "distance_q1": distance_q1,
            }
        )

    return results


def main(limit_files: Optional[int] = None) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(".csv")]
    files.sort()
    if limit_files is not None:
        files = files[:limit_files]

    all_rows: List[Dict[str, object]] = []
    for path in files:
        all_rows.extend(process_one_file(path))

    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Processed files: {len(files)}")
    print(f"Rows written: {len(all_rows)} -> {out_path}")


if __name__ == "__main__":
    # Default: process all files
    main(limit_files=None)

