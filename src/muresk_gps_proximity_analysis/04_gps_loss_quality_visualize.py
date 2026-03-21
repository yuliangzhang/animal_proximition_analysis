from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOSS_STAT_FILE = Path("data/muresk_farm_proximity_data/muresk_gps_data/gps_loss_stat.csv")
QUALITY_STAT_FILE = Path("data/muresk_farm_proximity_data/muresk_gps_data/gps_quality_stat.csv")
OUTPUT_DIR = Path("data/muresk_farm_proximity_data/gps_quality_figure")

DAILY_LOSS_FIG = OUTPUT_DIR / "gps_daily_loss_rate.png"
DAILY_QUALITY_FIG = OUTPUT_DIR / "gps_daily_quality_hdop.png"
TIME_ZONE_LOSS_FIG = OUTPUT_DIR / "gps_time_zone_loss_rate.png"
TIME_ZONE_QUALITY_FIG = OUTPUT_DIR / "gps_time_zone_quality_hdop.png"


def plot_lines(df: pd.DataFrame, x_col: str, y_col: str, title: str, y_label: str, output_file: Path) -> None:
    plt.figure(figsize=(14, 6))
    for design in sorted(df["coller_design"].dropna().unique().tolist()):
        subset = df[df["coller_design"] == design].sort_values(x_col)
        plt.plot(subset[x_col], subset[y_col], marker="o", markersize=3, linewidth=1.5, label=design)

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_label)
    plt.grid(alpha=0.3)
    plt.legend(title="coller_design")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def build_daily_loss(loss_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        loss_df.groupby(["coller_design", "stat_date"], as_index=False)[["none_zero_num", "zero_num"]]
        .sum()
    )
    denominator = grouped["none_zero_num"] + grouped["zero_num"]
    grouped["loss_rate"] = (grouped["zero_num"] / denominator * 100).fillna(0).round(2)
    grouped["stat_date"] = pd.to_datetime(grouped["stat_date"], errors="coerce")
    return grouped


def build_daily_quality(quality_df: pd.DataFrame) -> pd.DataFrame:
    grouped = quality_df.groupby(["coller_design", "stat_date"], as_index=False)["median_hdop"].mean()
    grouped["median_hdop"] = grouped["median_hdop"].round(2)
    grouped["stat_date"] = pd.to_datetime(grouped["stat_date"], errors="coerce")
    return grouped


def build_time_zone_loss(loss_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        loss_df.groupby(["coller_design", "stat_time_zone"], as_index=False)[["none_zero_num", "zero_num"]]
        .sum()
    )
    denominator = grouped["none_zero_num"] + grouped["zero_num"]
    grouped["loss_rate"] = (grouped["zero_num"] / denominator * 100).fillna(0).round(2)
    grouped["stat_time_zone"] = pd.to_datetime(grouped["stat_time_zone"], errors="coerce")
    return grouped


def build_time_zone_quality(quality_df: pd.DataFrame) -> pd.DataFrame:
    grouped = quality_df.groupby(["coller_design", "stat_time_zone"], as_index=False)["median_hdop"].mean()
    grouped["median_hdop"] = grouped["median_hdop"].round(2)
    grouped["stat_time_zone"] = pd.to_datetime(grouped["stat_time_zone"], errors="coerce")
    return grouped


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loss_df = pd.read_csv(LOSS_STAT_FILE)
    quality_df = pd.read_csv(QUALITY_STAT_FILE)

    daily_loss = build_daily_loss(loss_df)
    daily_quality = build_daily_quality(quality_df)
    time_zone_loss = build_time_zone_loss(loss_df)
    time_zone_quality = build_time_zone_quality(quality_df)

    plot_lines(
        df=daily_loss,
        x_col="stat_date",
        y_col="loss_rate",
        title="Daily GPS Loss Rate by Collar Design",
        y_label="Loss Rate (%)",
        output_file=DAILY_LOSS_FIG,
    )
    plot_lines(
        df=daily_quality,
        x_col="stat_date",
        y_col="median_hdop",
        title="Daily GPS Quality (Mean of Median HDOP) by Collar Design",
        y_label="HDOP",
        output_file=DAILY_QUALITY_FIG,
    )
    plot_lines(
        df=time_zone_loss,
        x_col="stat_time_zone",
        y_col="loss_rate",
        title="Time-Zone GPS Loss Rate by Collar Design",
        y_label="Loss Rate (%)",
        output_file=TIME_ZONE_LOSS_FIG,
    )
    plot_lines(
        df=time_zone_quality,
        x_col="stat_time_zone",
        y_col="median_hdop",
        title="Time-Zone GPS Quality (Median HDOP) by Collar Design",
        y_label="HDOP",
        output_file=TIME_ZONE_QUALITY_FIG,
    )

    print(f"Saved: {DAILY_LOSS_FIG}")
    print(f"Saved: {DAILY_QUALITY_FIG}")
    print(f"Saved: {TIME_ZONE_LOSS_FIG}")
    print(f"Saved: {TIME_ZONE_QUALITY_FIG}")


if __name__ == "__main__":
    main()
