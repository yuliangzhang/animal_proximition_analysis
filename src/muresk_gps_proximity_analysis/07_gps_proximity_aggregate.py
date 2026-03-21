from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_DIR = Path("data/muresk_farm_proximity_data/gps_proximition_data")
SUMMARY_FILE = INPUT_DIR / "pair_distance_summary.csv"
FIGURE_ROOT = INPUT_DIR / "figures"
ANALYSIS_DIR = INPUT_DIR / "analysis"

WINDOW_MINUTES = [30, 60, 90, 120, 150, 180, 210, 240]
DISTANCE_THRESHOLDS_M = [10, 15, 20, 25, 30, 35, 40, 45, 50]

LAG_STEP_MINUTES = 5
MAX_LAG_MINUTES = 24 * 60
MIN_OVERLAP_POINTS = 10
BASE_SLOT_MINUTES = 5.0


@dataclass
class CorrStat:
    pearson: float | None
    spearman: float | None
    n_points: int


def load_pair_files(summary_file: Path, input_dir: Path) -> list[tuple[str, Path]]:
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    summary_df = pd.read_csv(summary_file)
    required = {"receiver_gps_id", "beacon_gps_id"}
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"Missing columns in summary file: {sorted(missing)}")

    pair_files: list[tuple[str, Path]] = []
    for row in summary_df.itertuples(index=False):
        pair_id = f"{row.receiver_gps_id}_{row.beacon_gps_id}"
        file_path = input_dir / f"{pair_id}_distance.csv"
        if file_path.exists():
            pair_files.append((pair_id, file_path))
    if not pair_files:
        raise FileNotFoundError(f"No pair distance files found in {input_dir}")
    return pair_files


def load_pair_timeseries(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, usecols=["time_slot", "proximity_count", "distance_m"])
    df["time_slot"] = pd.to_datetime(df["time_slot"], errors="coerce")
    df["proximity_count"] = pd.to_numeric(df["proximity_count"], errors="coerce")
    df["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")
    df = df.dropna(subset=["time_slot"]).sort_values("time_slot")
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["time_slot"], keep="first").set_index("time_slot")
    return df


def shift_series_index(series: pd.Series, minutes: int) -> pd.Series:
    if minutes == 0:
        return series
    shifted = series.copy()
    shifted.index = shifted.index + pd.Timedelta(minutes=minutes)
    return shifted


def aggregate_proximity(
    proximity_minutes: pd.Series,
    proximity_valid_minutes: pd.Series,
    window_minutes: int,
    shift_minutes: int,
) -> pd.DataFrame:
    freq = f"{window_minutes}min"
    p_minutes = shift_series_index(proximity_minutes, shift_minutes)
    p_valid = shift_series_index(proximity_valid_minutes, shift_minutes)

    prox_sum = p_minutes.resample(freq).sum(min_count=1)
    prox_valid_sum = p_valid.resample(freq).sum(min_count=1)
    prox_rate = prox_sum / prox_valid_sum
    prox_rate = prox_rate.where(prox_valid_sum > 0)

    out = pd.DataFrame(
        {
            "proximity_sum": prox_sum,
            "proximity_valid_minutes": prox_valid_sum,
            "proximity_rate": prox_rate,
        }
    )
    out.index.name = "time_slot"
    return out


def aggregate_distance(distance_m: pd.Series, window_minutes: int, threshold_m: int) -> pd.DataFrame:
    freq = f"{window_minutes}min"
    valid_count = distance_m.notna().astype("int64").resample(freq).sum(min_count=1)
    in_range_count = ((distance_m <= threshold_m) & distance_m.notna()).astype("int64").resample(freq).sum(min_count=1)
    in_range_ratio = in_range_count / valid_count
    in_range_ratio = in_range_ratio.where(valid_count > 0)

    out = pd.DataFrame(
        {
            "distance_valid_count": valid_count,
            "distance_in_range_count": in_range_count,
            "distance_in_range_ratio": in_range_ratio,
        }
    )
    out.index.name = "time_slot"
    return out


def compute_corr_stat(proximity_rate: pd.Series, gps_ratio: pd.Series) -> CorrStat:
    aligned = pd.DataFrame({"p": proximity_rate, "g": gps_ratio}).dropna()
    if len(aligned) < MIN_OVERLAP_POINTS:
        return CorrStat(pearson=None, spearman=None, n_points=int(len(aligned)))
    if aligned["p"].nunique() < 2 or aligned["g"].nunique() < 2:
        return CorrStat(pearson=None, spearman=None, n_points=int(len(aligned)))

    pearson_val = aligned["p"].corr(aligned["g"], method="pearson")
    spearman_val = aligned["p"].rank(method="average").corr(aligned["g"].rank(method="average"), method="pearson")
    return CorrStat(
        pearson=float(pearson_val) if pd.notna(pearson_val) else None,
        spearman=float(spearman_val) if pd.notna(spearman_val) else None,
        n_points=int(len(aligned)),
    )


def pick_best_lag(stats_by_lag: dict[int, CorrStat]) -> int | None:
    candidates: list[tuple[int, CorrStat]] = [
        (lag, stat) for lag, stat in stats_by_lag.items() if stat.pearson is not None
    ]
    if not candidates:
        return None

    # maximize positive consistency (proximity high with GPS-close ratio high)
    candidates.sort(
        key=lambda item: (
            item[1].pearson,
            item[1].spearman if item[1].spearman is not None else -1.0,
            item[1].n_points,
        ),
        reverse=True,
    )
    return candidates[0][0]


def plot_raw_vs_aligned(
    pair_id: str,
    window_minutes: int,
    threshold_m: int,
    combo_df: pd.DataFrame,
    output_path: Path,
    best_lag_minutes: int | None,
    lag0_corr: float | None,
    best_corr: float | None,
) -> None:
    plot_df = combo_df.sort_values("time_slot")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax_top, ax_bottom = axes

    # Top: raw
    top_left = plot_df.dropna(subset=["proximity_rate_raw"])
    top_right = plot_df.dropna(subset=["distance_in_range_ratio"])
    ax_top.plot(
        top_left["time_slot"],
        top_left["proximity_rate_raw"],
        color="#b45309",
        linewidth=1.5,
        label="Proximity Rate (Raw)",
    )
    ax_top_b = ax_top.twinx()
    ax_top_b.plot(
        top_right["time_slot"],
        top_right["distance_in_range_ratio"],
        color="#1d4ed8",
        linewidth=1.5,
        label=f"GPS Close Ratio (<= {threshold_m}m)",
    )
    ax_top.set_ylabel("Proximity Rate")
    ax_top_b.set_ylabel("GPS Close Ratio")
    ax_top.set_ylim(0, 1.0)
    ax_top_b.set_ylim(0, 1.0)
    ax_top.grid(alpha=0.25)
    top_lines = ax_top.get_lines() + ax_top_b.get_lines()
    ax_top.legend(top_lines, [line.get_label() for line in top_lines], loc="upper right")

    # Bottom: aligned by best lag
    bottom_left = plot_df.dropna(subset=["proximity_rate_aligned"])
    bottom_right = top_right
    ax_bottom.plot(
        bottom_left["time_slot"],
        bottom_left["proximity_rate_aligned"],
        color="#b45309",
        linewidth=1.5,
        label="Proximity Rate (Lag Adjusted)",
    )
    ax_bottom_b = ax_bottom.twinx()
    ax_bottom_b.plot(
        bottom_right["time_slot"],
        bottom_right["distance_in_range_ratio"],
        color="#1d4ed8",
        linewidth=1.5,
        label=f"GPS Close Ratio (<= {threshold_m}m)",
    )
    ax_bottom.set_ylabel("Proximity Rate")
    ax_bottom_b.set_ylabel("GPS Close Ratio")
    ax_bottom.set_ylim(0, 1.0)
    ax_bottom_b.set_ylim(0, 1.0)
    ax_bottom.grid(alpha=0.25)
    bottom_lines = ax_bottom.get_lines() + ax_bottom_b.get_lines()
    ax_bottom.legend(bottom_lines, [line.get_label() for line in bottom_lines], loc="upper right")

    lag_txt = "N/A" if best_lag_minutes is None else f"{best_lag_minutes:+d} min"
    lag0_txt = "N/A" if lag0_corr is None else f"{lag0_corr:.3f}"
    best_txt = "N/A" if best_corr is None else f"{best_corr:.3f}"

    fig.suptitle(
        f"{pair_id} | Window={window_minutes}min | Threshold<={threshold_m}m\n"
        f"Lag0 Corr={lag0_txt} | Best Corr={best_txt} | Best Lag={lag_txt}",
        fontsize=11,
    )
    ax_bottom.set_xlabel("Time")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    pair_files = load_pair_files(SUMMARY_FILE, INPUT_DIR)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    lags = list(range(-MAX_LAG_MINUTES, MAX_LAG_MINUTES + 1, LAG_STEP_MINUTES))

    lag_scan_rows: list[dict] = []
    lag_summary_rows: list[dict] = []
    timeseries_rows: list[pd.DataFrame] = []

    for pair_id, file_path in pair_files:
        base_df = load_pair_timeseries(file_path)
        if base_df.empty:
            print(f"Skip empty pair file: {file_path}")
            continue

        proximity_minutes = base_df["proximity_count"]
        proximity_valid_minutes = base_df["proximity_count"].notna().astype("float64") * BASE_SLOT_MINUTES
        distance_m = base_df["distance_m"]

        prox_cache: dict[tuple[int, int], pd.DataFrame] = {}

        for window_minutes in WINDOW_MINUTES:
            # precompute lagged proximity aggregation once per lag/window
            for lag_minutes in lags:
                prox_cache[(window_minutes, lag_minutes)] = aggregate_proximity(
                    proximity_minutes=proximity_minutes,
                    proximity_valid_minutes=proximity_valid_minutes,
                    window_minutes=window_minutes,
                    shift_minutes=lag_minutes,
                )

            for threshold_m in DISTANCE_THRESHOLDS_M:
                gps_agg = aggregate_distance(
                    distance_m=distance_m,
                    window_minutes=window_minutes,
                    threshold_m=threshold_m,
                )

                stats_by_lag: dict[int, CorrStat] = {}
                for lag_minutes in lags:
                    prox_agg = prox_cache[(window_minutes, lag_minutes)]
                    stat = compute_corr_stat(
                        proximity_rate=prox_agg["proximity_rate"],
                        gps_ratio=gps_agg["distance_in_range_ratio"],
                    )
                    stats_by_lag[lag_minutes] = stat
                    lag_scan_rows.append(
                        {
                            "pair_id": pair_id,
                            "aggregate_window_min": window_minutes,
                            "distance_threshold_m": threshold_m,
                            "lag_minutes": lag_minutes,
                            "pearson_corr": stat.pearson,
                            "spearman_corr": stat.spearman,
                            "overlap_points": stat.n_points,
                        }
                    )

                best_lag = pick_best_lag(stats_by_lag)
                lag0_stat = stats_by_lag.get(0, CorrStat(pearson=None, spearman=None, n_points=0))
                best_stat = stats_by_lag.get(best_lag, CorrStat(pearson=None, spearman=None, n_points=0)) if best_lag is not None else CorrStat(pearson=None, spearman=None, n_points=0)

                lag_summary_rows.append(
                    {
                        "pair_id": pair_id,
                        "aggregate_window_min": window_minutes,
                        "distance_threshold_m": threshold_m,
                        "lag0_pearson_corr": lag0_stat.pearson,
                        "lag0_spearman_corr": lag0_stat.spearman,
                        "lag0_overlap_points": lag0_stat.n_points,
                        "best_lag_minutes": best_lag,
                        "best_pearson_corr": best_stat.pearson,
                        "best_spearman_corr": best_stat.spearman,
                        "best_overlap_points": best_stat.n_points,
                        "corr_improvement": (
                            (best_stat.pearson - lag0_stat.pearson)
                            if (best_stat.pearson is not None and lag0_stat.pearson is not None)
                            else None
                        ),
                    }
                )

                prox_raw = prox_cache[(window_minutes, 0)].rename(
                    columns={
                        "proximity_sum": "proximity_sum_raw",
                        "proximity_valid_minutes": "proximity_valid_minutes_raw",
                        "proximity_rate": "proximity_rate_raw",
                    }
                )
                prox_best = prox_cache[(window_minutes, best_lag)].rename(
                    columns={
                        "proximity_sum": "proximity_sum_aligned",
                        "proximity_valid_minutes": "proximity_valid_minutes_aligned",
                        "proximity_rate": "proximity_rate_aligned",
                    }
                ) if best_lag is not None else prox_raw.rename(
                    columns={
                        "proximity_sum_raw": "proximity_sum_aligned",
                        "proximity_valid_minutes_raw": "proximity_valid_minutes_aligned",
                        "proximity_rate_raw": "proximity_rate_aligned",
                    }
                )

                combo = (
                    gps_agg.join(prox_raw, how="outer")
                    .join(prox_best, how="outer")
                    .reset_index()
                )
                combo["pair_id"] = pair_id
                combo["aggregate_window_min"] = window_minutes
                combo["distance_threshold_m"] = threshold_m
                combo["best_lag_minutes"] = best_lag
                combo["lag0_pearson_corr"] = lag0_stat.pearson
                combo["best_pearson_corr"] = best_stat.pearson

                ordered_cols = [
                    "pair_id",
                    "aggregate_window_min",
                    "distance_threshold_m",
                    "time_slot",
                    "proximity_sum_raw",
                    "proximity_valid_minutes_raw",
                    "proximity_rate_raw",
                    "distance_valid_count",
                    "distance_in_range_count",
                    "distance_in_range_ratio",
                    "proximity_sum_aligned",
                    "proximity_valid_minutes_aligned",
                    "proximity_rate_aligned",
                    "best_lag_minutes",
                    "lag0_pearson_corr",
                    "best_pearson_corr",
                ]
                combo = combo[ordered_cols]
                timeseries_rows.append(combo)

                fig_dir = FIGURE_ROOT / f"window_{window_minutes}min"
                fig_file = fig_dir / f"{pair_id}_le_{threshold_m}m.png"
                plot_raw_vs_aligned(
                    pair_id=pair_id,
                    window_minutes=window_minutes,
                    threshold_m=threshold_m,
                    combo_df=combo,
                    output_path=fig_file,
                    best_lag_minutes=best_lag,
                    lag0_corr=lag0_stat.pearson,
                    best_corr=best_stat.pearson,
                )

            print(f"[{pair_id}] completed window={window_minutes}min")

    lag_scan_df = pd.DataFrame(lag_scan_rows).sort_values(
        ["pair_id", "aggregate_window_min", "distance_threshold_m", "lag_minutes"],
        kind="stable",
    )
    lag_summary_df = pd.DataFrame(lag_summary_rows).sort_values(
        ["pair_id", "aggregate_window_min", "distance_threshold_m"],
        kind="stable",
    )
    all_timeseries_df = pd.concat(timeseries_rows, ignore_index=True).sort_values(
        ["pair_id", "aggregate_window_min", "distance_threshold_m", "time_slot"],
        kind="stable",
    )

    lag_scan_path = INPUT_DIR / "gps_proximity_grid_lag_scan.csv"
    lag_summary_path = INPUT_DIR / "gps_proximity_grid_lag_summary.csv"
    timeseries_path = INPUT_DIR / "gps_proximity_grid_timeseries.csv"

    lag_scan_df.to_csv(lag_scan_path, index=False, encoding="utf-8")
    lag_summary_df.to_csv(lag_summary_path, index=False, encoding="utf-8")
    all_timeseries_df.to_csv(timeseries_path, index=False, encoding="utf-8")

    # analysis helpers for report script
    best_by_pair = (
        lag_summary_df.dropna(subset=["best_pearson_corr"])
        .sort_values(["pair_id", "best_pearson_corr"], ascending=[True, False], kind="stable")
        .groupby("pair_id", as_index=False)
        .head(1)
    )
    best_by_pair_path = ANALYSIS_DIR / "gps_proximity_grid_best_by_pair.csv"
    best_by_pair.to_csv(best_by_pair_path, index=False, encoding="utf-8")

    print(f"Pairs analyzed: {lag_summary_df['pair_id'].nunique()}")
    print(f"Grid size: windows={len(WINDOW_MINUTES)} x thresholds={len(DISTANCE_THRESHOLDS_M)}")
    print(f"Lag search: {min(lags)} to {max(lags)} min, step={LAG_STEP_MINUTES} min")
    print(f"Saved lag scan: {lag_scan_path}")
    print(f"Saved lag summary: {lag_summary_path}")
    print(f"Saved timeseries: {timeseries_path}")
    print(f"Saved best-by-pair: {best_by_pair_path}")
    print(f"Saved figures root: {FIGURE_ROOT}")


if __name__ == "__main__":
    main()
