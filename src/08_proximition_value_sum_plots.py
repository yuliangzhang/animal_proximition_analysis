import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = "data/dating_analysis/proximition_distance_analysis_6_split.csv"
OUT_DIR = "data/dating_analysis/proximition_plot"


def prepare_long(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure numeric value_sum
    df = df.copy()
    df["value_sum"] = pd.to_numeric(df.get("value_sum", 0), errors="coerce").fillna(0)

    ram = df[["ram_eid", "time_split", "value_sum"]].rename(columns={"ram_eid": "eid"})
    ewe = df[["ewe_eid", "time_split", "value_sum"]].rename(columns={"ewe_eid": "eid"})
    long_df = pd.concat([ram, ewe], ignore_index=True)
    # Aggregate in case of duplicates
    agg = long_df.groupby(["eid", "time_split"], as_index=False)["value_sum"].sum()
    # Sort time_split lexicographically (YYYY-MM-DD_0i format sorts correctly)
    agg = agg.sort_values(["eid", "time_split"]) 
    return agg


def plot_eid_series(eid: str, sub: pd.DataFrame) -> str:
    if sub.empty:
        return ""
    x = sub["time_split"].tolist()
    y = sub["value_sum"].tolist()

    plt.figure(figsize=(12, 4))
    plt.plot(range(len(x)), y, marker="o", linewidth=1)
    plt.title(f"Proximition value_sum over time | EID={eid}")
    plt.xlabel("time_split (chronological)")
    plt.ylabel("value_sum")
    # Ticks: keep it readable by showing at most 30 ticks
    step = max(1, len(x) // 30)
    idxs = list(range(0, len(x), step))
    plt.xticks(idxs, [x[i] for i in idxs], rotation=90)
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"value_sum_{eid}.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    # Quick sanity checks
    req = {"ram_eid", "ewe_eid", "time_split", "value_sum"}
    missing = req - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in input CSV: {missing}")

    agg = prepare_long(df)

    made = 0
    for eid, sub in agg.groupby("eid"):
        out = plot_eid_series(eid, sub)
        if out:
            made += 1

    print(f"Plotted {made} EIDs to {OUT_DIR}")


if __name__ == "__main__":
    main()

