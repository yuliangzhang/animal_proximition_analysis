import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd


INPUT_DIR = "data/proximition_distance_split_full"
OUTPUT_DIR = "data/dating_analysis"
OUTPUT_FILE = "proximition_distance_analysis_6_split.csv"
DISTANCE_MAX_M = 500.0


def parse_filename(fname: str) -> Optional[Tuple[str, str, str]]:
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


def split_index(hour: int) -> int:
    """Return 1..6 for 4-hour bins: [0,4),[4,8),[8,12),[12,16),[16,20),[20,24)."""
    if 0 <= hour < 4:
        return 1
    if 4 <= hour < 8:
        return 2
    if 8 <= hour < 12:
        return 3
    if 12 <= hour < 16:
        return 4
    if 16 <= hour < 20:
        return 5
    return 6


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
    df["hour"] = df["ts"].dt.hour
    df["split"] = df["hour"].apply(split_index)
    df["time_split"] = df["date"] + "_" + df["split"].astype(str).str.zfill(2)

    # Value as numeric
    df["value_num"] = pd.to_numeric(df.get("value", 0), errors="coerce").fillna(0)

    # Distance as numeric; drop non-finite and > 500 for averaging
    df["distance_num"] = pd.to_numeric(df.get("distance"), errors="coerce")

    results: List[Dict[str, object]] = []
    for time_split, grp in df.groupby("time_split", sort=True):
        value_sum = float(grp["value_num"].sum())

        # Filter distances for averaging
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
                "time_split": time_split,
                "value_sum": value_sum,
                "distance_avg": distance_avg,
                "distance_cnt": distance_cnt,
                "distance_media": distance_media,
                "distance_q1": distance_q1,
            }
        )

    return results


def main(limit_files: Optional[int] = 10) -> None:
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
    # Test run on 10 files by default
    main(limit_files=10)
