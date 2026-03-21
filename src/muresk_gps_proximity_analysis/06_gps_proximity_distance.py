from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROXIMITY_FILE = Path("data/muresk_farm_proximity_data/muresk_proximity_data/muresk_proximity_5min.csv")
GPS_FILE = Path("data/muresk_farm_proximity_data/muresk_gps_data/muresk_gps_raw_data.csv")
OUTPUT_DIR = Path("data/muresk_farm_proximity_data/gps_proximition_data")
SUMMARY_FILE = OUTPUT_DIR / "pair_distance_summary.csv"

START_DATE = "2025-12-10"
END_DATE = "2026-01-03"
MAX_INTERP_POINTS = 3  # 3 x 5min = 15min
MIN_VALID_RATIO = 0.5  # exclude devices with valid ratio < 50% (loss rate > 50%)
PROXIMITY_TIME_SHIFT_MINUTES = 0  # positive: move proximity timeline later

META_COLS = [
    "receiver_gps_id",
    "receiver_actigraph_id",
    "receiver_serial_id",
    "beacon_gps_id",
    "beacon_actigraph_id",
    "beacon_serial_id",
    "beacon_collar_design",
]

def normalize_gps_code(value: object) -> str:
    if pd.isna(value):
        return ""
    text = "".join(str(value).strip().split()).upper()
    if not text:
        return ""
    return text


def get_time_columns(df: pd.DataFrame) -> list[str]:
    time_cols: list[str] = []
    for col in df.columns:
        if col in META_COLS:
            continue
        ts = pd.to_datetime(col, errors="coerce")
        if pd.notna(ts):
            time_cols.append(col)
    return time_cols


def haversine_distance_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance in meters."""
    r = 6_371_000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def spearman_corr(x: pd.Series, y: pd.Series) -> float | object:
    """Spearman correlation without scipy dependency."""
    x_rank = x.rank(method="average")
    y_rank = y.rank(method="average")
    corr = x_rank.corr(y_rank, method="pearson")
    if pd.isna(corr):
        return pd.NA
    return float(corr)


def load_proximity_pairs(proximity_file: Path) -> tuple[pd.DataFrame, list[pd.Timestamp], pd.Timestamp, pd.Timestamp]:
    df = pd.read_csv(proximity_file)
    missing = set(META_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in proximity file: {sorted(missing)}")

    time_cols = get_time_columns(df)
    if not time_cols:
        raise ValueError("No time slot columns found in proximity file.")

    all_slots = pd.to_datetime(time_cols, errors="coerce").sort_values()
    start_ts = pd.Timestamp(f"{START_DATE} 00:00:00")
    end_ts = pd.Timestamp(f"{END_DATE} 23:59:59")
    all_slots = all_slots[(all_slots >= start_ts) & (all_slots <= end_ts)]
    if all_slots.empty:
        raise ValueError("No proximity time slots after applying START_DATE/END_DATE.")

    work_df = df.copy()
    work_df["beacon_collar_design_norm"] = (
        work_df["beacon_collar_design"].astype("string").str.strip().str.lower()
    )
    work_df["receiver_gps_norm"] = work_df["receiver_gps_id"].apply(normalize_gps_code)
    work_df["beacon_gps_norm"] = work_df["beacon_gps_id"].apply(normalize_gps_code)

    pair_df = work_df[
        (work_df["receiver_gps_norm"] != "")
        & (work_df["beacon_gps_norm"] != "")
    ].copy()

    if pair_df.empty:
        raise ValueError("No receiver-beacon pairs with valid receiver/beacon GPS IDs.")

    pair_df = pair_df.drop_duplicates(subset=["receiver_gps_norm", "beacon_gps_norm"], keep="first")
    return pair_df, list(all_slots), start_ts, end_ts


def load_gps_5min(gps_file: Path, needed_codes: set[str], start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(gps_file)
    required = {"gps_code", "timestamp", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in GPS file: {sorted(missing)}")

    df["gps_code_norm"] = df["gps_code"].apply(normalize_gps_code)
    df = df[df["gps_code_norm"].isin(needed_codes)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].copy()

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Mark invalid positions as missing before aggregation.
    invalid = (
        df["latitude"].isna()
        | df["longitude"].isna()
        | ((df["latitude"] == 0) & (df["longitude"] == 0))
        | (~df["latitude"].between(-90, 90))
        | (~df["longitude"].between(-180, 180))
    )
    df.loc[invalid, ["latitude", "longitude"]] = np.nan

    df["time_slot"] = df["timestamp"].dt.floor("5min")
    agg = (
        df.groupby(["gps_code_norm", "time_slot"], as_index=False)
        .agg(
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            valid_point_count=("latitude", lambda s: int(s.notna().sum())),
        )
        .sort_values(["gps_code_norm", "time_slot"], kind="stable")
    )
    return agg


def compute_valid_ratio_by_code(gps_agg: pd.DataFrame, all_slots: list[pd.Timestamp]) -> dict[str, float]:
    """Compute raw valid 5-min slot ratio for each GPS code."""
    total_slots = max(len(all_slots), 1)
    slot_set = set(all_slots)
    in_window = gps_agg["time_slot"].isin(slot_set)
    valid_df = gps_agg[in_window & gps_agg["latitude"].notna() & gps_agg["longitude"].notna()]
    counts = valid_df.groupby("gps_code_norm").size().to_dict()
    return {str(code): float(cnt) / float(total_slots) for code, cnt in counts.items()}


def build_code_timeseries(
    gps_agg: pd.DataFrame,
    gps_code: str,
    all_slots: list[pd.Timestamp],
) -> pd.DataFrame:
    code_df = gps_agg[gps_agg["gps_code_norm"] == gps_code][
        ["time_slot", "latitude", "longitude", "valid_point_count"]
    ].copy()
    code_df = code_df.drop_duplicates(subset=["time_slot"], keep="first").set_index("time_slot")

    out = code_df.reindex(all_slots)
    out.index.name = "time_slot"
    out["valid_point_count"] = out["valid_point_count"].fillna(0).astype("int64")
    out["raw_available"] = out["latitude"].notna() & out["longitude"].notna()

    out["latitude"] = out["latitude"].interpolate(
        method="linear",
        limit=MAX_INTERP_POINTS,
        limit_area="inside",
    )
    out["longitude"] = out["longitude"].interpolate(
        method="linear",
        limit=MAX_INTERP_POINTS,
        limit_area="inside",
    )

    after_interp = out["latitude"].notna() & out["longitude"].notna()
    out["point_source"] = np.where(
        out["raw_available"],
        "raw",
        np.where(after_interp, "interpolated", "missing"),
    )
    return out.reset_index()


def extract_proximity_series(
    row: pd.Series,
    all_slots: list[pd.Timestamp],
    shift_minutes: int,
) -> pd.DataFrame:
    slot_index = pd.DatetimeIndex(all_slots, name="time_slot")
    values = [pd.to_numeric(row.get(slot.strftime("%Y-%m-%d %H:%M:%S")), errors="coerce") for slot in all_slots]
    ser = pd.Series(values, index=slot_index, name="proximity_count").astype("Float64")

    if shift_minutes != 0:
        ser = ser.shift(freq=pd.Timedelta(minutes=shift_minutes))
        ser = ser.groupby(level=0).sum(min_count=1).reindex(slot_index)

    df = ser.reset_index()
    df.columns = ["time_slot", "proximity_count"]
    return df


def process_pair(
    row: pd.Series,
    gps_agg: pd.DataFrame,
    all_slots: list[pd.Timestamp],
    proximity_shift_minutes: int,
) -> tuple[pd.DataFrame, dict]:
    receiver = row["receiver_gps_norm"]
    beacon = row["beacon_gps_norm"]
    beacon_collar_design = str(row.get("beacon_collar_design_norm", "")).strip().lower()

    recv_ts = build_code_timeseries(gps_agg, receiver, all_slots).rename(
        columns={
            "latitude": "receiver_latitude",
            "longitude": "receiver_longitude",
            "valid_point_count": "receiver_valid_point_count",
            "point_source": "receiver_point_source",
        }
    )
    beacon_ts = build_code_timeseries(gps_agg, beacon, all_slots).rename(
        columns={
            "latitude": "beacon_latitude",
            "longitude": "beacon_longitude",
            "valid_point_count": "beacon_valid_point_count",
            "point_source": "beacon_point_source",
        }
    )
    prox_ts = extract_proximity_series(row, all_slots, proximity_shift_minutes)

    out = prox_ts.merge(recv_ts, on="time_slot", how="left").merge(beacon_ts, on="time_slot", how="left")
    out = out.drop(columns=["raw_available_x", "raw_available_y"], errors="ignore")

    valid = (
        out["receiver_latitude"].notna()
        & out["receiver_longitude"].notna()
        & out["beacon_latitude"].notna()
        & out["beacon_longitude"].notna()
    )
    out["distance_m"] = np.nan
    if valid.any():
        out.loc[valid, "distance_m"] = haversine_distance_m(
            out.loc[valid, "receiver_latitude"].to_numpy(dtype=float),
            out.loc[valid, "receiver_longitude"].to_numpy(dtype=float),
            out.loc[valid, "beacon_latitude"].to_numpy(dtype=float),
            out.loc[valid, "beacon_longitude"].to_numpy(dtype=float),
        )

    out.insert(1, "receiver_gps_id", receiver)
    out.insert(2, "beacon_gps_id", beacon)
    out.insert(3, "beacon_collar_design", beacon_collar_design)

    valid_pair = out[out["proximity_count"].notna() & out["distance_m"].notna()].copy()
    spearman = pd.NA
    pearson = pd.NA
    if len(valid_pair) >= 3 and valid_pair["proximity_count"].nunique() > 1 and valid_pair["distance_m"].nunique() > 1:
        spearman = spearman_corr(valid_pair["proximity_count"], valid_pair["distance_m"])
        pearson_val = valid_pair["proximity_count"].corr(valid_pair["distance_m"], method="pearson")
        pearson = float(pearson_val) if pd.notna(pearson_val) else pd.NA

    summary = {
        "receiver_gps_id": receiver,
        "beacon_gps_id": beacon,
        "beacon_collar_design": beacon_collar_design,
        "proximity_time_shift_minutes": proximity_shift_minutes,
        "total_time_slots": len(out),
        "proximity_non_missing_slots": int(out["proximity_count"].notna().sum()),
        "proximity_positive_slots": int((out["proximity_count"].fillna(0) > 0).sum()),
        "distance_non_missing_slots": int(out["distance_m"].notna().sum()),
        "receiver_raw_slots": int((out["receiver_point_source"] == "raw").sum()),
        "receiver_interpolated_slots": int((out["receiver_point_source"] == "interpolated").sum()),
        "beacon_raw_slots": int((out["beacon_point_source"] == "raw").sum()),
        "beacon_interpolated_slots": int((out["beacon_point_source"] == "interpolated").sum()),
        "median_distance_m": float(out["distance_m"].median()) if out["distance_m"].notna().any() else pd.NA,
        "spearman_corr_proximity_vs_distance": spearman,
        "pearson_corr_proximity_vs_distance": pearson,
    }
    return out, summary


def main() -> None:
    pair_df, all_slots, start_ts, end_ts = load_proximity_pairs(PROXIMITY_FILE)
    needed_codes = set(pair_df["receiver_gps_norm"].tolist()) | set(pair_df["beacon_gps_norm"].tolist())
    gps_agg = load_gps_5min(GPS_FILE, needed_codes, start_ts, end_ts)
    valid_ratio_map = compute_valid_ratio_by_code(gps_agg, all_slots)

    pair_df["receiver_valid_ratio"] = pair_df["receiver_gps_norm"].map(valid_ratio_map).fillna(0.0)
    pair_df["beacon_valid_ratio"] = pair_df["beacon_gps_norm"].map(valid_ratio_map).fillna(0.0)
    excluded_pairs = pair_df[
        (pair_df["receiver_valid_ratio"] < MIN_VALID_RATIO) | (pair_df["beacon_valid_ratio"] < MIN_VALID_RATIO)
    ][["receiver_gps_norm", "beacon_gps_norm", "receiver_valid_ratio", "beacon_valid_ratio"]].copy()
    pair_df = pair_df[
        (pair_df["receiver_valid_ratio"] >= MIN_VALID_RATIO) & (pair_df["beacon_valid_ratio"] >= MIN_VALID_RATIO)
    ].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for _, row in pair_df.iterrows():
        out_df, summary = process_pair(row, gps_agg, all_slots, PROXIMITY_TIME_SHIFT_MINUTES)
        file_name = f"{summary['receiver_gps_id']}_{summary['beacon_gps_id']}_distance.csv"
        out_path = OUTPUT_DIR / file_name
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries).sort_values(["receiver_gps_id", "beacon_gps_id"], kind="stable")
    summary_df.to_csv(SUMMARY_FILE, index=False, encoding="utf-8")

    print(f"Pairs processed: {len(summary_df)}")
    print(f"Date filter: {start_ts} to {end_ts}")
    print(f"Applied proximity time shift: {PROXIMITY_TIME_SHIFT_MINUTES} minutes")
    print(f"Min GPS valid ratio per device: {MIN_VALID_RATIO:.2%}")
    if not excluded_pairs.empty:
        print("Excluded pairs due to low GPS valid ratio:")
        for _, r in excluded_pairs.iterrows():
            print(
                f"  {r['receiver_gps_norm']}-{r['beacon_gps_norm']} | "
                f"receiver={r['receiver_valid_ratio']:.2%}, beacon={r['beacon_valid_ratio']:.2%}"
            )
    print(f"Saved pair files to: {OUTPUT_DIR}")
    print(f"Saved summary to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
