from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/muresk_farm_proximity_data/muresk_gps_data/muresk_gps_raw_data.csv")
LOSS_OUTPUT_FILE = Path("data/muresk_farm_proximity_data/muresk_gps_data/gps_loss_stat.csv")
QUALITY_OUTPUT_FILE = Path("data/muresk_farm_proximity_data/muresk_gps_data/gps_quality_stat.csv")
DEVICE_FILTER_OUTPUT_FILE = Path("data/muresk_farm_proximity_data/muresk_gps_data/gps_device_loss_filter_stat.csv")

START_DATE = "2025-12-10"
END_DATE = "2026-01-03"
DEVICE_LOSS_FILTER_THRESHOLD = 50.0


def build_time_grid(designs: list[str]) -> pd.DataFrame:
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    zone_starts = [0, 4, 8, 12, 16, 20]

    rows: list[dict[str, str]] = []
    for design in designs:
        for date in dates:
            for hour in zone_starts:
                zone_dt = date + pd.Timedelta(hours=hour)
                rows.append(
                    {
                        "coller_design": design,
                        "stat_date": date.strftime("%Y-%m-%d"),
                        "stat_time_zone": zone_dt.strftime("%Y-%m-%d %H:%M"),
                    }
                )
    return pd.DataFrame(rows)


def filter_extreme_devices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    working_df = df.copy()
    working_df["is_non_zero"] = ((working_df["latitude"] != 0) & (working_df["longitude"] != 0)).astype(int)
    working_df["is_zero"] = ((working_df["latitude"] == 0) & (working_df["longitude"] == 0)).astype(int)

    device_stat = (
        working_df.groupby(["gps_code", "coller_design"], as_index=False)
        .agg(none_zero_num=("is_non_zero", "sum"), zero_num=("is_zero", "sum"))
    )
    device_stat["total_num"] = device_stat["none_zero_num"] + device_stat["zero_num"]
    device_stat = device_stat[device_stat["total_num"] > 0].copy()
    device_stat["loss_rate"] = (device_stat["zero_num"] / device_stat["total_num"] * 100).round(2)
    device_stat["excluded"] = device_stat["loss_rate"] > DEVICE_LOSS_FILTER_THRESHOLD

    exclude_codes = sorted(device_stat.loc[device_stat["excluded"], "gps_code"].unique().tolist())
    filtered_df = working_df[~working_df["gps_code"].isin(exclude_codes)].copy()
    filtered_df = filtered_df.drop(columns=["is_non_zero", "is_zero"])

    return filtered_df, device_stat, exclude_codes


def prepare_base_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)
    required_cols = {"gps_code", "timestamp", "coller_design", "latitude", "longitude", "HDOP"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input file: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["stat_date_dt"] = df["timestamp"].dt.floor("D")
    end_exclusive = pd.to_datetime(END_DATE) + pd.Timedelta(days=1)

    df = df[
        (df["timestamp"] >= pd.to_datetime(START_DATE))
        & (df["timestamp"] < end_exclusive)
        & df["coller_design"].notna()
    ].copy()

    df["time_zone_start_dt"] = df["stat_date_dt"] + pd.to_timedelta((df["timestamp"].dt.hour // 4) * 4, unit="h")
    df["stat_date"] = df["stat_date_dt"].dt.strftime("%Y-%m-%d")
    df["stat_time_zone"] = df["time_zone_start_dt"].dt.strftime("%Y-%m-%d %H:%M")

    filtered_df, device_stat, excluded_codes = filter_extreme_devices(df)
    device_stat.to_csv(DEVICE_FILTER_OUTPUT_FILE, index=False, encoding="utf-8")

    print(
        f"Device filtering threshold: loss_rate > {DEVICE_LOSS_FILTER_THRESHOLD:.2f}% | "
        f"excluded devices: {len(excluded_codes)}"
    )
    if excluded_codes:
        print(f"Excluded gps_code: {', '.join(excluded_codes)}")
    print(f"Saved device filter stat to: {DEVICE_FILTER_OUTPUT_FILE}")

    return filtered_df


def generate_gps_loss_stat(df: pd.DataFrame, full_grid: pd.DataFrame) -> pd.DataFrame:
    grouped_df = df.copy()
    grouped_df["is_non_zero"] = ((grouped_df["latitude"] != 0) & (grouped_df["longitude"] != 0)).astype(int)
    grouped_df["is_zero"] = ((grouped_df["latitude"] == 0) & (grouped_df["longitude"] == 0)).astype(int)
    loss_stat = (
        grouped_df.groupby(["coller_design", "stat_date", "stat_time_zone"], as_index=False)
        .agg(none_zero_num=("is_non_zero", "sum"), zero_num=("is_zero", "sum"))
    )

    loss_stat = full_grid.merge(loss_stat, on=["coller_design", "stat_date", "stat_time_zone"], how="left")
    loss_stat["none_zero_num"] = loss_stat["none_zero_num"].fillna(0).astype(int)
    loss_stat["zero_num"] = loss_stat["zero_num"].fillna(0).astype(int)
    return loss_stat[["coller_design", "stat_date", "stat_time_zone", "none_zero_num", "zero_num"]]


def generate_gps_quality_stat(df: pd.DataFrame, full_grid: pd.DataFrame) -> pd.DataFrame:
    valid_df = df[(df["latitude"] != 0) & (df["longitude"] != 0)].copy()
    valid_df["HDOP"] = pd.to_numeric(valid_df["HDOP"], errors="coerce")
    valid_df = valid_df.dropna(subset=["HDOP"])

    grouped = (
        valid_df.groupby(["coller_design", "stat_date", "stat_time_zone"])["HDOP"]
        .agg(["count", "mean", "min", "max", "median"])
        .reset_index()
    )
    grouped = grouped.rename(
        columns={
            "mean": "avg_hdop",
            "min": "min_hdop",
            "max": "max_hdop",
            "median": "median_hdop",
        }
    )

    quality_stat = full_grid.merge(grouped, on=["coller_design", "stat_date", "stat_time_zone"], how="left")

    default_hdop = 25.50
    less_than_three = quality_stat["count"].fillna(0) < 3
    for col in ["avg_hdop", "min_hdop", "max_hdop", "median_hdop"]:
        quality_stat.loc[less_than_three, col] = default_hdop

    quality_stat = quality_stat[["coller_design", "stat_date", "stat_time_zone", "avg_hdop", "min_hdop", "max_hdop", "median_hdop"]]
    for col in ["avg_hdop", "min_hdop", "max_hdop", "median_hdop"]:
        quality_stat[col] = quality_stat[col].round(2)

    return quality_stat


def main() -> None:
    base_df = prepare_base_data()
    designs = sorted(base_df["coller_design"].unique().tolist())
    full_grid = build_time_grid(designs)

    loss_stat = generate_gps_loss_stat(base_df, full_grid)
    quality_stat = generate_gps_quality_stat(base_df, full_grid)

    loss_stat.to_csv(LOSS_OUTPUT_FILE, index=False, encoding="utf-8")
    quality_stat.to_csv(QUALITY_OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"Saved loss stat to: {LOSS_OUTPUT_FILE} (rows={len(loss_stat)})")
    print(f"Saved quality stat to: {QUALITY_OUTPUT_FILE} (rows={len(quality_stat)})")


if __name__ == "__main__":
    main()
