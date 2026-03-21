from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path("data/muresk_farm_proximity_data/gps_proximition_data")
OUTPUT_DIR = BASE_DIR / "figures" / "corr_figures"

PAIRS = ["N172_N171", "N172_N182", "N172_N189"]
AGGREGATE_WINDOW_MIN = 240
DISTANCE_THRESHOLD_M = 15
BASE_SAMPLE_MIN = 5.0

LAG_MAX_MIN = 1440
LAG_STEP_MIN = 5

ANOMALY_HOURS = {12, 13, 14, 15}

MIN_POINTS_FOR_FIT = 8
MIN_DISTANCE_VALID_RATIO = 0.50


def load_pair_raw(pair_id: str) -> pd.DataFrame:
    path = BASE_DIR / f"{pair_id}_distance.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing pair file: {path}")

    df = pd.read_csv(path, usecols=["time_slot", "proximity_count", "distance_m"])
    df["time_slot"] = pd.to_datetime(df["time_slot"], errors="coerce")
    df["proximity_count"] = pd.to_numeric(df["proximity_count"], errors="coerce")
    df["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")
    df = df.dropna(subset=["time_slot"]).sort_values("time_slot")
    if df.empty:
        raise ValueError(f"No valid rows for pair: {pair_id}")
    return df


def aggregate_pair(df_raw: pd.DataFrame, lag_min: int) -> pd.DataFrame:
    idx = pd.DatetimeIndex(df_raw["time_slot"])
    proximity = pd.Series(df_raw["proximity_count"].to_numpy(), index=idx)
    proximity_valid_min = pd.Series(
        np.where(proximity.notna(), BASE_SAMPLE_MIN, 0.0),
        index=idx,
        dtype="float64",
    )
    distance = pd.Series(df_raw["distance_m"].to_numpy(), index=idx)

    if lag_min != 0:
        shift = pd.Timedelta(minutes=lag_min)
        proximity.index = proximity.index + shift
        proximity_valid_min.index = proximity_valid_min.index + shift

    freq = f"{AGGREGATE_WINDOW_MIN}min"
    proximity_sum = proximity.resample(freq).sum(min_count=1)
    proximity_valid_minutes = proximity_valid_min.resample(freq).sum(min_count=1)
    proximity_rate = (proximity_sum / proximity_valid_minutes).where(proximity_valid_minutes > 0)

    distance_valid_count = distance.notna().astype("int64").resample(freq).sum(min_count=1)
    distance_in_range_count = (
        ((distance <= DISTANCE_THRESHOLD_M) & distance.notna())
        .astype("int64")
        .resample(freq)
        .sum(min_count=1)
    )
    gps_close_ratio = (distance_in_range_count / distance_valid_count).where(distance_valid_count > 0)
    expected_points = int(AGGREGATE_WINDOW_MIN / BASE_SAMPLE_MIN)
    distance_valid_ratio = distance_valid_count / expected_points
    distance_missing_ratio = 1.0 - distance_valid_ratio

    out = pd.DataFrame(
        {
            "time_slot": proximity_sum.index,
            "proximity_rate": proximity_rate.values,
            "gps_close_ratio": gps_close_ratio.reindex(proximity_sum.index).values,
            "distance_valid_ratio": distance_valid_ratio.reindex(proximity_sum.index).values,
            "distance_missing_ratio": distance_missing_ratio.reindex(proximity_sum.index).values,
        }
    )
    return out


def pearson_corr(a: pd.Series, b: pd.Series) -> tuple[float | None, int]:
    t = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(t) < MIN_POINTS_FOR_FIT or t["a"].nunique() < 2 or t["b"].nunique() < 2:
        return None, int(len(t))
    val = t["a"].corr(t["b"], method="pearson")
    return (float(val) if pd.notna(val) else None), int(len(t))


def scan_best_lag(df_raw: pd.DataFrame) -> dict:
    lags = list(range(-LAG_MAX_MIN, LAG_MAX_MIN + 1, LAG_STEP_MIN))
    best_lag = 0
    best_corr = None
    best_n = 0

    for lag in lags:
        agg = aggregate_pair(df_raw, lag)
        agg = agg[agg["distance_valid_ratio"] >= MIN_DISTANCE_VALID_RATIO]
        corr, n = pearson_corr(agg["proximity_rate"], agg["gps_close_ratio"])
        if corr is None:
            continue
        if (best_corr is None) or (corr > best_corr):
            best_corr = corr
            best_lag = lag
            best_n = n

    return {"best_lag_min": best_lag, "best_corr": best_corr, "best_n": best_n}


def build_mode_table(pair_raw: dict[str, pd.DataFrame], pair_lags: dict[str, int]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for pair_id in PAIRS:
        lag = pair_lags[pair_id]
        agg = aggregate_pair(pair_raw[pair_id], lag)
        agg = agg[agg["distance_valid_ratio"] >= MIN_DISTANCE_VALID_RATIO]
        agg = agg.dropna(subset=["proximity_rate", "gps_close_ratio"]).copy()
        agg["pair_id"] = pair_id
        agg["lag_min"] = lag
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def fit_linear(df: pd.DataFrame) -> dict:
    t = df.dropna(subset=["proximity_rate", "gps_close_ratio"]).copy()
    if len(t) < MIN_POINTS_FOR_FIT or t["gps_close_ratio"].nunique() < 2:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "pearson_r": np.nan,
            "n": int(len(t)),
        }

    x = t["gps_close_ratio"].to_numpy(dtype=float)
    y = t["proximity_rate"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    pearson_r = float(pd.Series(x).corr(pd.Series(y), method="pearson"))

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
        "pearson_r": pearson_r,
        "n": int(len(t)),
    }


def plot_mode(
    df_mode: pd.DataFrame,
    mode_name: str,
    out_path: Path,
    pair_lags: dict[str, int],
    drop_midday: bool,
) -> dict:
    colors = {
        "N172_N171": "#1d4ed8",
        "N172_N182": "#059669",
        "N172_N189": "#7c3aed",
    }

    df = df_mode.copy()

    if drop_midday:
        df = df[~df["time_slot"].dt.hour.isin(ANOMALY_HOURS)].copy()

    clean = df.copy()
    fit = fit_linear(clean)

    fig, ax = plt.subplots(figsize=(10, 7))
    for pair_id in PAIRS:
        t = df[df["pair_id"] == pair_id]
        if t.empty:
            continue
        pair_label = pair_id
        if mode_name.lower().startswith("best"):
            pair_label = f"{pair_id} (lag {pair_lags[pair_id]:+d}m)"
        ax.scatter(
            t["gps_close_ratio"],
            t["proximity_rate"],
            s=42,
            alpha=0.72,
            color=colors.get(pair_id, "#374151"),
            label=pair_label,
        )

    if np.isfinite(fit["slope"]) and np.isfinite(fit["intercept"]):
        x_min = float(clean["gps_close_ratio"].min())
        x_max = float(clean["gps_close_ratio"].max())
        x_line = np.linspace(x_min, x_max, 200)
        y_line = fit["slope"] * x_line + fit["intercept"]
        ax.plot(x_line, y_line, color="black", linewidth=2.0, label="Linear fit (anomaly-removed)")

    view_tag = " (12:00-16:00 removed)" if drop_midday else ""
    title = (
        f"{mode_name}: Proximity Rate vs GPS Close Ratio "
        f"(distance <= {DISTANCE_THRESHOLD_M}m, window = {AGGREGATE_WINDOW_MIN}min){view_tag}"
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("GPS close ratio", fontsize=11)
    ax.set_ylabel("Proximity rate", fontsize=11)
    ax.grid(alpha=0.25)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=-0.01)

    txt = (
        f"Fit (after GPS missing-rate filtering): y = {fit['slope']:.4f}x + {fit['intercept']:.4f}\n"
        f"R^2 = {fit['r2']:.4f} | Pearson r = {fit['pearson_r']:.4f}\n"
        f"n = {fit['n']} | keep buckets with distance_missing_ratio <= 50%"
    )
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#6b7280"),
    )
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    return {
        "mode": mode_name,
        "drop_midday_12_16": drop_midday,
        "slope": fit["slope"],
        "intercept": fit["intercept"],
        "r2": fit["r2"],
        "pearson_r": fit["pearson_r"],
        "n_after_dropna": int(len(df)),
        "n_used_for_fit": fit["n"],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pair_raw: dict[str, pd.DataFrame] = {pair_id: load_pair_raw(pair_id) for pair_id in PAIRS}

    best_lag_rows: list[dict] = []
    best_lags: dict[str, int] = {}
    for pair_id in PAIRS:
        best = scan_best_lag(pair_raw[pair_id])
        lag = int(best["best_lag_min"])
        best_lags[pair_id] = lag
        best_lag_rows.append(
            {
                "pair_id": pair_id,
                "best_lag_min": lag,
                "best_corr": best["best_corr"],
                "best_points": best["best_n"],
            }
        )

    lag0_lags = {pair_id: 0 for pair_id in PAIRS}
    lag0_table = build_mode_table(pair_raw=pair_raw, pair_lags=lag0_lags)
    best_table = build_mode_table(pair_raw=pair_raw, pair_lags=best_lags)

    lag0_stats = plot_mode(
        df_mode=lag0_table,
        mode_name="Lag0",
        out_path=OUTPUT_DIR / "lag0_correlation_scatter_fit.png",
        pair_lags=lag0_lags,
        drop_midday=False,
    )
    lag0_drop_midday_stats = plot_mode(
        df_mode=lag0_table,
        mode_name="Lag0",
        out_path=OUTPUT_DIR / "lag0_correlation_scatter_fit_no_12_16.png",
        pair_lags=lag0_lags,
        drop_midday=True,
    )
    best_stats = plot_mode(
        df_mode=best_table,
        mode_name="Best Lag",
        out_path=OUTPUT_DIR / "best_lag_correlation_scatter_fit.png",
        pair_lags=best_lags,
        drop_midday=False,
    )
    best_drop_midday_stats = plot_mode(
        df_mode=best_table,
        mode_name="Best Lag",
        out_path=OUTPUT_DIR / "best_lag_correlation_scatter_fit_no_12_16.png",
        pair_lags=best_lags,
        drop_midday=True,
    )

    pd.DataFrame(best_lag_rows).sort_values("pair_id").to_csv(
        OUTPUT_DIR / "best_lag_by_pair.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame([lag0_stats, lag0_drop_midday_stats, best_stats, best_drop_midday_stats]).to_csv(
        OUTPUT_DIR / "correlation_fit_summary.csv", index=False, encoding="utf-8"
    )

    print(f"Saved figure: {OUTPUT_DIR / 'lag0_correlation_scatter_fit.png'}")
    print(f"Saved figure: {OUTPUT_DIR / 'lag0_correlation_scatter_fit_no_12_16.png'}")
    print(f"Saved figure: {OUTPUT_DIR / 'best_lag_correlation_scatter_fit.png'}")
    print(f"Saved figure: {OUTPUT_DIR / 'best_lag_correlation_scatter_fit_no_12_16.png'}")
    print(f"Saved lag table: {OUTPUT_DIR / 'best_lag_by_pair.csv'}")
    print(f"Saved fit summary: {OUTPUT_DIR / 'correlation_fit_summary.csv'}")


if __name__ == "__main__":
    main()
