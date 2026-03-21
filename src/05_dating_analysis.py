import os
from typing import List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt


PROX_DIR = "data/proximition_split"
OUT_DIR = "data/dating_analysis"
PLOT_DIR = os.path.join(OUT_DIR, "plot")


def parse_filename(fname: str) -> Optional[Tuple[str, str, str]]:
    """
    Example: G_GPS0398_GPS0008_940-110012305898_940-110009540691.csv
    -> (group='G', ram_eid='940-110012305898', ewe_eid='940-110009540691')
    """
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


def detect_dating_times(df: pd.DataFrame) -> List[pd.Timestamp]:
    """
    Dating rule: a timestamp t is selected if
    - value(t) >= 4, and
    - For each offset in {5,10,15} minutes, both value(t - offset) > 0 and value(t + offset) > 0
    Returns list of timestamps (as pandas Timestamp) that satisfy the rule.
    """
    if df.empty or "timestamp" not in df.columns or "value" not in df.columns:
        return []
    tmp = df.copy()
    tmp["ts"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
    tmp = tmp.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    if tmp.empty:
        return []

    # Map ts -> value
    s = tmp.set_index("ts")["value"]
    # Ensure numeric
    s = pd.to_numeric(s, errors="coerce").fillna(0)

    candidates = s[s >= 4].index
    results: List[pd.Timestamp] = []
    offsets = [5, 10, 15]
    for t in candidates:
        ok = True
        for m in offsets:
            prev_t = t - pd.Timedelta(minutes=m)
            next_t = t + pd.Timedelta(minutes=m)
            if prev_t not in s.index or next_t not in s.index:
                ok = False
                break
            if not (s.loc[prev_t] > 0 and s.loc[next_t] > 0):
                ok = False
                break
        if ok:
            results.append(t)
    return results


def build_dating_table(limit: Optional[int] = None) -> pd.DataFrame:
    rows: List[dict] = []
    files = [os.path.join(PROX_DIR, f) for f in os.listdir(PROX_DIR) if f.lower().endswith(".csv")]
    files.sort()
    if limit is not None:
        files = files[:limit]

    for f in files:
        parsed = parse_filename(f)
        if not parsed:
            continue
        group, ram_eid, ewe_eid = parsed
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        dts = detect_dating_times(df)
        for t in dts:
            rows.append({
                "ram_eid": ram_eid,
                "ewe_eid": ewe_eid,
                "dating_time": t.strftime("%Y-%m-%d %H:%M:%S"),
                "group": group,
            })

    return pd.DataFrame(rows)


def plot_bar_counts(df: pd.DataFrame, title: str, out_path: str) -> None:
    plt.figure(figsize=(12, 6))
    ax = df.plot(kind="bar", legend=False)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(limit: Optional[int] = None) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    dating_df = build_dating_table(limit=limit)
    dating_path = os.path.join(OUT_DIR, "dating_time.csv")
    dating_df.to_csv(dating_path, index=False)

    if dating_df.empty:
        print("No dating times found.")
        return

    # Parse hour
    dating_df["dating_ts"] = pd.to_datetime(dating_df["dating_time"], errors="coerce")
    dating_df.dropna(subset=["dating_ts"], inplace=True)
    dating_df["hour"] = dating_df["dating_ts"].dt.hour

    # Overall bar
    overall_counts = dating_df.groupby("hour").size().reindex(range(24), fill_value=0)
    plot_bar_counts(overall_counts, "Overall Dating Time Distribution (by hour)", os.path.join(PLOT_DIR, "overall_by_hour.png"))

    # Per-group bars
    for grp, name in [("G", "High Shade"), ("Y", "Medium Shade"), ("P", "No Shade")]:
        sub = dating_df[dating_df["group"] == grp]
        if sub.empty:
            continue
        counts = sub.groupby("hour").size().reindex(range(24), fill_value=0)
        plot_bar_counts(counts, f"{name} (group {grp}) Dating Time Distribution", os.path.join(PLOT_DIR, f"group_{grp}_by_hour.png"))

    print(f"Wrote dating table: {dating_path}")
    print(f"Plots in: {PLOT_DIR}")


if __name__ == "__main__":
    # Run full dataset (set limit=None). For a quick test, pass a small number.
    main(limit=None)

