"""UWA proximity exploratory analysis and mating inference pipeline.

This script performs:
1) Pair-level pre-filtering before all anomaly actions.
2) Data quality screening and low-signal pair detection.
3) Plot-day management anomaly mining using robust statistics.
4) Pair-level mating inference with biologically constrained filtering.
5) Year + paddock result export (6 files in total).
6) EDA tables, figures, and bilingual Markdown reports.

Run with:
    /opt/miniconda3/bin/python src/proximity_data_analysis/02_eda_uwa_mating_analysis.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------
# Paths and constants
# ----------------------------
DATA_DIR = Path("data/uwa_proximity_data")
EDA_DIR = Path("data/uwa_proximity_eda")
RES_DIR = Path("data/uwa_proximity_res")
FIG_DIR = EDA_DIR / "figures"
TABLE_DIR = EDA_DIR / "tables"
REPORT_DIR = EDA_DIR / "reports"

YEARS = [2023, 2024]
PLOTS = ["High_shade", "Medium_shade", "No_shade"]
PLOT_MAP = {
    "High Shade": "High_shade",
    "Medium Shade": "Medium_shade",
    "No Shade": "No_shade",
}

# Pair-level low-signal rule
LOW_SIGNAL_ZERO_RATIO_THRESHOLD = 0.50
LOW_SIGNAL_TOTAL_QUANTILE = 0.20

# Plot-day management anomaly rule
MANAGEMENT_RZ_THRESHOLD = 3.0

# Pair pre-filter rule (requested upgrade)
PAIR_MIN_NONZERO_DAYS = 7
PAIR_TRIM_N = 2
PAIR_RZ_THRESHOLD = 3.0

# Pair-level mating candidate rule
MATING_SCORE_THRESHOLD = 4.0
MATING_FC_THRESHOLD = 3.0
MATING_DOMINANCE_THRESHOLD = 2.0
MATING_ABS_COUNT_THRESHOLD = 80.0

# Biological plausibility caps
MAX_EWE_PER_RAM_PER_DAY = 30
MAX_RAM_PER_EWE_PER_DAY = 3


@dataclass
class YearData:
    """Container for one year's day-level and 4-hour-level data."""

    year: int
    day_df: pd.DataFrame
    h4_df: pd.DataFrame
    full_day_cols: List[str]
    day_to_zone_cols: Dict[str, List[str]]


def ensure_dirs(paths: Iterable[Path]) -> None:
    """Create output folders if missing."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def robust_z(series: pd.Series) -> pd.Series:
    """Compute robust z-score using median and MAD."""
    median = float(series.median())
    mad = float((series - median).abs().median())
    scale = 1.4826 * mad
    if scale <= 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - median) / scale


def normalize_treatment(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize plot naming format."""
    out = df.copy()
    out["Plot"] = out["Plot"].replace(PLOT_MAP)
    return out


def parse_day_columns(columns: List[str]) -> Tuple[List[str], pd.Series]:
    """Parse day columns and return both original labels and timestamps."""
    day_cols = columns[5:]
    day_ts = pd.to_datetime(day_cols, dayfirst=True, errors="coerce")
    return list(day_cols), pd.Series(day_ts, index=day_cols)


def full_day_columns(day_cols: List[str], day_ts: pd.Series) -> List[str]:
    """Keep only complete days: exclude first and last day in each year file."""
    min_ts = day_ts.min()
    max_ts = day_ts.max()
    return [col for col in day_cols if day_ts[col] > min_ts and day_ts[col] < max_ts]


def build_day_to_zone_cols(h4_cols: List[str]) -> Dict[str, List[str]]:
    """Map each day label to its 4-hour columns."""
    mapping: Dict[str, List[str]] = {}
    for col in h4_cols:
        ts = pd.to_datetime(col, dayfirst=True, errors="coerce")
        if pd.isna(ts):
            continue
        day_label = f"{ts.day}/{ts.month}/{ts.year}"
        mapping.setdefault(day_label, []).append(col)

    # Keep chronological order for deterministic max selection on ties.
    for day in mapping:
        mapping[day] = sorted(mapping[day], key=lambda c: pd.to_datetime(c, dayfirst=True))
    return mapping


def load_year_data(year: int) -> YearData:
    """Load one year input files and derive day/time metadata."""
    day_path = DATA_DIR / f"HS_proximity_{year}_day_UWA.csv"
    h4_path = DATA_DIR / f"HS_proximity_{year}_4h_UWA.csv"

    if not day_path.exists():
        raise FileNotFoundError(f"Missing day-level file: {day_path}")
    if not h4_path.exists():
        raise FileNotFoundError(f"Missing 4-hour file: {h4_path}")

    day_df = normalize_treatment(pd.read_csv(day_path))
    h4_df = normalize_treatment(pd.read_csv(h4_path))

    # Validate pair row alignment.
    key_cols = ["Ram_EID", "Ewe_EID", "Plot"]
    if not day_df[key_cols].equals(h4_df[key_cols]):
        raise ValueError(f"Row alignment mismatch between day and 4h files for year {year}.")

    day_cols, day_ts = parse_day_columns(list(day_df.columns))
    full_cols = full_day_columns(day_cols, day_ts)
    h4_cols = list(h4_df.columns[5:])
    mapping = build_day_to_zone_cols(h4_cols)
    return YearData(year=year, day_df=day_df, h4_df=h4_df, full_day_cols=full_cols, day_to_zone_cols=mapping)


def trimmed_nonzero_mean(values: pd.Series, trim_n: int = PAIR_TRIM_N) -> float:
    """Compute mean of non-zero values after removing both tails."""
    non_zero = values[values > 0].astype(float).sort_values().to_numpy()
    if len(non_zero) < (2 * trim_n + 1):
        return np.nan
    trimmed = non_zero[trim_n:-trim_n]
    if len(trimmed) == 0:
        return np.nan
    return float(np.mean(trimmed))


def prefilter_pairs_before_anomaly(bundle: YearData) -> Tuple[YearData, pd.DataFrame]:
    """Apply pair-level pre-filter rules before management anomaly detection.

    Requested logic:
    Step 1: count non-zero experiment days per pair.
    Step 2: remove pairs with non-zero day count < 7.
    Step 3: for remaining pairs, compute trimmed non-zero mean
            (remove two largest and two smallest non-zero values).
    Step 4: robust-z filter on Step 3 metric with threshold 3.
    """
    full_cols = bundle.full_day_cols
    day_numeric = bundle.day_df[full_cols].fillna(0).astype(float)
    non_zero_days = (day_numeric > 0).sum(axis=1)
    pass_step2 = non_zero_days >= PAIR_MIN_NONZERO_DAYS

    # Step 3: trimmed mean on non-zero values after Step 2.
    trimmed_mean = pd.Series(np.nan, index=bundle.day_df.index, dtype=float)
    step2_idx = bundle.day_df.index[pass_step2]
    if len(step2_idx) > 0:
        trimmed_mean.loc[step2_idx] = day_numeric.loc[step2_idx].apply(trimmed_nonzero_mean, axis=1)

    # Step 4: robust z-score filter by treatment to avoid paddock-level scale bias.
    rz = pd.Series(np.nan, index=bundle.day_df.index, dtype=float)
    pass_step4 = pd.Series(False, index=bundle.day_df.index, dtype=bool)
    for treatment, g in bundle.day_df.groupby("Plot"):
        idx = g.index.intersection(step2_idx)
        vals = trimmed_mean.loc[idx].dropna()
        if vals.empty:
            continue
        rz_vals = robust_z(vals)
        rz.loc[vals.index] = rz_vals
        pass_step4.loc[vals.index] = rz_vals.abs() <= PAIR_RZ_THRESHOLD

    keep = pass_step2 & pass_step4

    # Build audit table for traceability.
    audit = bundle.day_df[["Ram_EID", "Ewe_EID", "Plot"]].copy()
    audit = audit.rename(columns={"Plot": "treatment"})
    audit["year"] = bundle.year
    audit["row_id_original"] = audit.index
    audit["non_zero_days"] = non_zero_days
    audit["pass_step2_non_zero_days"] = pass_step2
    audit["trimmed_nonzero_mean"] = trimmed_mean
    audit["trimmed_nonzero_mean_rz"] = rz
    audit["pass_step4_robust_z"] = pass_step4
    audit["pair_prefilter_keep"] = keep
    audit["prefilter_removed_reason"] = np.where(
        ~pass_step2,
        f"non_zero_days_lt_{PAIR_MIN_NONZERO_DAYS}",
        np.where(~pass_step4, "trimmed_mean_robust_z_outlier", "kept"),
    )

    # Keep day and 4h rows aligned after filtering.
    keep_idx = bundle.day_df.index[keep]
    day_filtered = bundle.day_df.loc[keep_idx].reset_index(drop=True)
    h4_filtered = bundle.h4_df.loc[keep_idx].reset_index(drop=True)

    filtered_bundle = YearData(
        year=bundle.year,
        day_df=day_filtered,
        h4_df=h4_filtered,
        full_day_cols=bundle.full_day_cols,
        day_to_zone_cols=bundle.day_to_zone_cols,
    )
    return filtered_bundle, audit.reset_index(drop=True)


def detect_management_anomalies(bundle: YearData) -> pd.DataFrame:
    """Detect plot-day management anomalies using total and sync robust z-scores."""
    rows: List[dict] = []
    full_cols = bundle.full_day_cols
    if not full_cols:
        raise ValueError(f"No full-day columns available for year {bundle.year}")

    for treatment, g in bundle.day_df.groupby("Plot"):
        x = g[full_cols].fillna(0).astype(float)
        day_total = x.sum(axis=0)
        active_pairs = (x > 0).sum(axis=0)

        # Daily synchronization ratio:
        # fraction of pairs above each pair's own 90th percentile on that day.
        pair_q90 = x.quantile(0.9, axis=1)
        sync_prop = ((x.T > pair_q90.values).T).mean(axis=0)

        rz_total = robust_z(day_total)
        rz_sync = robust_z(sync_prop)
        flag = (rz_total >= MANAGEMENT_RZ_THRESHOLD) | (rz_sync >= MANAGEMENT_RZ_THRESHOLD)

        for day_col in full_cols:
            day_ts = pd.to_datetime(day_col, dayfirst=True)
            rows.append(
                {
                    "year": bundle.year,
                    "treatment": treatment,
                    "day_col": day_col,
                    "date": day_ts.date().isoformat(),
                    "day_total": float(day_total[day_col]),
                    "active_pairs": int(active_pairs[day_col]),
                    "sync_prop": float(sync_prop[day_col]),
                    "rz_total": float(rz_total[day_col]),
                    "rz_sync": float(rz_sync[day_col]),
                    "management_anomaly": bool(flag[day_col]),
                }
            )

    out = pd.DataFrame(rows).sort_values(["year", "treatment", "date"]).reset_index(drop=True)
    return out


def infer_pair_candidates(
    bundle: YearData,
    anomaly_df: pd.DataFrame,
) -> pd.DataFrame:
    """Infer one strongest candidate day per pair and compute quality metrics."""
    full_cols = bundle.full_day_cols
    n_days = len(full_cols)
    rows: List[dict] = []

    management_by_treatment = (
        anomaly_df[anomaly_df["management_anomaly"]]
        .groupby("treatment")["day_col"]
        .apply(set)
        .to_dict()
    )

    day_numeric = bundle.day_df[full_cols].fillna(0).astype(float)

    for treatment, g in bundle.day_df.groupby("Plot"):
        idx = g.index
        x = day_numeric.loc[idx]
        period_total = x.sum(axis=1)
        zero_ratio = (x == 0).sum(axis=1) / n_days
        low_signal_threshold = float(period_total.quantile(LOW_SIGNAL_TOTAL_QUANTILE))
        management_days = management_by_treatment.get(treatment, set())
        baseline_days = [c for c in full_cols if c not in management_days]
        if not baseline_days:
            baseline_days = full_cols

        for row_id in idx:
            row = bundle.day_df.loc[row_id]
            series = day_numeric.loc[row_id]

            total = float(period_total.loc[row_id])
            zratio = float(zero_ratio.loc[row_id])
            low_signal = (zratio > LOW_SIGNAL_ZERO_RATIO_THRESHOLD) and (total < low_signal_threshold)

            base = series[baseline_days]
            baseline_median = float(base.median())
            baseline_mad = float((base - baseline_median).abs().median())
            baseline_scale = 1.4826 * baseline_mad + 1.0

            candidate_series = series.drop(labels=list(management_days), errors="ignore")
            if candidate_series.empty:
                best_day = ""
                best_count = 0.0
                best_z = -1e9
                best_fc = 0.0
                best_dom = 0.0
            else:
                best_day = str(candidate_series.idxmax())
                best_count = float(candidate_series.max())
                sorted_vals = candidate_series.sort_values()
                second = float(sorted_vals.iloc[-2]) if len(sorted_vals) > 1 else 0.0
                best_z = (best_count - baseline_median) / baseline_scale
                best_fc = (best_count + 1.0) / (baseline_median + 1.0)
                best_dom = (best_count + 1.0) / (second + 1.0)

            candidate_pre_cap = (
                (not low_signal)
                and (best_count >= MATING_ABS_COUNT_THRESHOLD)
                and (best_z >= MATING_SCORE_THRESHOLD)
                and (best_fc >= MATING_FC_THRESHOLD)
                and (best_dom >= MATING_DOMINANCE_THRESHOLD)
            )

            rows.append(
                {
                    "year": bundle.year,
                    "row_id": int(row_id),
                    "Ewe_EID": row["Ewe_EID"],
                    "Ram_EID": row["Ram_EID"],
                    "treatment": treatment,
                    "period_total": total,
                    "zero_ratio": zratio,
                    "low_signal_threshold": low_signal_threshold,
                    "low_signal_pair": bool(low_signal),
                    "baseline_median": baseline_median,
                    "baseline_mad": baseline_mad,
                    "best_day_col": best_day,
                    "best_day_count": best_count,
                    "best_day_z": best_z,
                    "best_day_fc": best_fc,
                    "best_day_dominance": best_dom,
                    "n_full_days": n_days,
                    "day_proximity_avg": total / n_days if n_days > 0 else 0.0,
                    "candidate_pre_cap": bool(candidate_pre_cap),
                }
            )

    out = pd.DataFrame(rows).sort_values(["year", "treatment", "row_id"]).reset_index(drop=True)
    return out


def apply_biological_caps(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    """Apply ram/day and ewe/day caps to candidate events."""
    cand = pair_metrics[pair_metrics["candidate_pre_cap"]].copy()
    if cand.empty:
        pair_metrics["candidate_selected"] = False
        pair_metrics["candidate_removed_by_cap"] = False
        pair_metrics["ram_day_rank"] = np.nan
        pair_metrics["ewe_day_rank"] = np.nan
        return pair_metrics

    cand = cand.sort_values(
        ["year", "treatment", "Ram_EID", "best_day_col", "best_day_z", "best_day_count"],
        ascending=[True, True, True, True, False, False],
    )
    cand["ram_day_rank"] = cand.groupby(["year", "treatment", "Ram_EID", "best_day_col"]).cumcount() + 1
    cand = cand[cand["ram_day_rank"] <= MAX_EWE_PER_RAM_PER_DAY].copy()

    cand = cand.sort_values(
        ["year", "treatment", "Ewe_EID", "best_day_col", "best_day_z", "best_day_count"],
        ascending=[True, True, True, True, False, False],
    )
    cand["ewe_day_rank"] = cand.groupby(["year", "treatment", "Ewe_EID", "best_day_col"]).cumcount() + 1
    selected = cand[cand["ewe_day_rank"] <= MAX_RAM_PER_EWE_PER_DAY].copy()

    selected_keys = set(zip(selected["year"], selected["row_id"]))

    out = pair_metrics.copy()
    out["candidate_selected"] = out.apply(lambda r: (r["year"], r["row_id"]) in selected_keys, axis=1)
    out["candidate_removed_by_cap"] = out["candidate_pre_cap"] & (~out["candidate_selected"])

    rank_cols = selected[["year", "row_id", "ram_day_rank", "ewe_day_rank"]]
    out = out.merge(rank_cols, on=["year", "row_id"], how="left")
    return out


def zone_label_from_col(col: str) -> str:
    """Convert 4-hour start timestamp to readable zone label."""
    ts = pd.to_datetime(col, dayfirst=True)
    start = ts.hour
    end = (start + 4) % 24
    return f"{start:02d}:00-{end:02d}:00"


def build_final_results(
    bundles: Dict[int, YearData],
    pair_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build final pair-level output with required columns."""
    rows: List[dict] = []

    for _, r in pair_metrics.iterrows():
        year = int(r["year"])
        row_id = int(r["row_id"])
        bundle = bundles[year]
        n_days = int(r["n_full_days"])
        day_avg = float(r["day_proximity_avg"])
        is_mating = bool(r["candidate_selected"])

        if is_mating:
            day_col = str(r["best_day_col"])
            day_count = float(r["best_day_count"])
            dt = pd.to_datetime(day_col, dayfirst=True)
            date_label = dt.date().isoformat()

            if n_days > 1:
                day_avg_exp = float((r["period_total"] - day_count) / (n_days - 1))
            else:
                day_avg_exp = day_avg

            zone_cols = bundle.day_to_zone_cols.get(day_col, [])
            zone_values = bundle.h4_df.loc[row_id, zone_cols].fillna(0).astype(float) if zone_cols else pd.Series(dtype=float)
            if zone_values.empty:
                best_zone_col = ""
                best_zone_count = 0.0
                best_zone_label = ""
            else:
                best_zone_col = str(zone_values.idxmax())
                best_zone_count = float(zone_values.max())
                best_zone_label = zone_label_from_col(best_zone_col)
        else:
            date_label = ""
            best_zone_label = ""
            best_zone_count = 0.0
            day_count = 0.0
            day_avg_exp = day_avg

        rows.append(
            {
                "year": year,
                "Ewe_EID": r["Ewe_EID"],
                "Ram_EID": r["Ram_EID"],
                "treatment": r["treatment"],
                "Mating": "Y" if is_mating else "N",
                "date": date_label,
                "time_zone": best_zone_label,
                "day_proximity_times": day_count,
                "day_proximity_avg": day_avg,
                "day_proximity_avg_exp": day_avg_exp,
                "time_zone_proximity_times": best_zone_count,
            }
        )

    out = pd.DataFrame(rows)

    # Keep the requested final column order.
    final_cols = [
        "Ewe_EID",
        "Ram_EID",
        "treatment",
        "Mating",
        "date",
        "time_zone",
        "day_proximity_times",
        "day_proximity_avg",
        "day_proximity_avg_exp",
        "time_zone_proximity_times",
    ]
    out = out[["year"] + final_cols]
    return out


def save_split_result_files(final_df: pd.DataFrame) -> pd.DataFrame:
    """Save 2 x 3 final result files and return their index table."""
    rows: List[dict] = []
    for year in YEARS:
        for treatment in PLOTS:
            subset = final_df[(final_df["year"] == year) & (final_df["treatment"] == treatment)].copy()
            subset = subset.drop(columns=["year"]).sort_values(["Ewe_EID", "Ram_EID"]).reset_index(drop=True)
            out_path = RES_DIR / f"uwa_proximity_mating_{year}_{treatment}.csv"
            subset.to_csv(out_path, index=False)
            rows.append(
                {
                    "year": year,
                    "treatment": treatment,
                    "file": str(out_path),
                    "rows": int(len(subset)),
                    "mating_Y": int((subset["Mating"] == "Y").sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["year", "treatment"]).reset_index(drop=True)


def build_summary_tables(
    final_df: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    prefilter_audit: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Create summary tables for report and plotting."""
    result_core = final_df.copy()
    result_core["is_mating"] = result_core["Mating"].eq("Y")

    # Year-treatment overview.
    overview = (
        pair_metrics.groupby(["year", "treatment"], as_index=False)
        .agg(
            pairs=("row_id", "count"),
            low_signal_pairs=("low_signal_pair", "sum"),
            candidates_pre_cap=("candidate_pre_cap", "sum"),
            candidates_selected=("candidate_selected", "sum"),
        )
        .sort_values(["year", "treatment"])
    )

    # Plot-day anomaly summary.
    anomaly_summary = (
        anomaly_df.groupby(["year", "treatment"], as_index=False)
        .agg(
            days=("date", "count"),
            management_anomaly_days=("management_anomaly", "sum"),
        )
        .sort_values(["year", "treatment"])
    )

    flagged_days = anomaly_df[anomaly_df["management_anomaly"]].copy()
    flagged_days = flagged_days.sort_values(["year", "treatment", "date"]).reset_index(drop=True)

    # Pair pre-filter summary.
    prefilter_summary = (
        prefilter_audit.groupby(["year", "treatment"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "pairs_total": int(len(g)),
                    "removed_step2_nonzero_lt7": int((~g["pass_step2_non_zero_days"]).sum()),
                    "removed_step4_robust_z": int(
                        (g["pass_step2_non_zero_days"] & (~g["pass_step4_robust_z"])).sum()
                    ),
                    "kept_pairs": int(g["pair_prefilter_keep"].sum()),
                }
            )
        )
        .reset_index(drop=True)
        .sort_values(["year", "treatment"])
    )

    # Ram daily mating partner counts.
    mating_only = result_core[result_core["is_mating"]].copy()
    ram_day = (
        mating_only.groupby(["year", "treatment", "date", "Ram_EID"], as_index=False)
        .size()
        .rename(columns={"size": "mating_ewe_count"})
    )

    # Ewe mating summary.
    ewe_summary = (
        mating_only.groupby(["year", "treatment", "Ewe_EID"], as_index=False)
        .agg(
            mating_count=("Ram_EID", "count"),
            unique_rams=("Ram_EID", "nunique"),
        )
        .sort_values(["year", "treatment", "Ewe_EID"])
    )

    # Time-zone preference.
    tz_summary = (
        mating_only.groupby(["year", "treatment", "time_zone"], as_index=False)
        .size()
        .rename(columns={"size": "mating_events"})
        .sort_values(["year", "treatment", "time_zone"])
    )

    # Focus table for user-observed day.
    focus_date = "2023-03-02"
    focus_table = anomaly_df[
        (anomaly_df["year"] == 2023)
        & (anomaly_df["treatment"] == "High_shade")
        & (anomaly_df["date"].isin(["2023-02-15", focus_date]))
    ].copy()

    focus_candidates = (
        pair_metrics[
            (pair_metrics["year"] == 2023)
            & (pair_metrics["treatment"] == "High_shade")
            & (pair_metrics["best_day_col"].isin(["15/2/2023", "2/3/2023"]))
        ]
        .groupby("best_day_col", as_index=False)
        .agg(
            candidates_pre_cap=("candidate_pre_cap", "sum"),
            selected_after_cap=("candidate_selected", "sum"),
        )
    )
    if not focus_candidates.empty:
        focus_candidates["date"] = pd.to_datetime(focus_candidates["best_day_col"], dayfirst=True).dt.date.astype(str)
        focus_table = focus_table.merge(focus_candidates[["date", "candidates_pre_cap", "selected_after_cap"]], on="date", how="left")

    return {
        "overview": overview,
        "prefilter_summary": prefilter_summary,
        "anomaly_summary": anomaly_summary,
        "flagged_days": flagged_days,
        "ram_day": ram_day,
        "ewe_summary": ewe_summary,
        "tz_summary": tz_summary,
        "focus_table": focus_table,
    }


def save_tables(
    final_df: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    prefilter_audit: pd.DataFrame,
    table_dict: Dict[str, pd.DataFrame],
    split_index: pd.DataFrame,
) -> None:
    """Save all table outputs to EDA folder."""
    final_df.to_csv(TABLE_DIR / "mating_results_all.csv", index=False)
    pair_metrics.to_csv(TABLE_DIR / "pair_signal_metrics.csv", index=False)
    anomaly_df.to_csv(TABLE_DIR / "plot_day_anomaly_metrics.csv", index=False)
    prefilter_audit.to_csv(TABLE_DIR / "pair_prefilter_audit.csv", index=False)
    split_index.to_csv(TABLE_DIR / "final_result_file_index.csv", index=False)

    for name, df in table_dict.items():
        df.to_csv(TABLE_DIR / f"{name}.csv", index=False)


def setup_plot_style() -> None:
    """Configure plotting style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 160


def plot_day_heatmaps(anomaly_df: pd.DataFrame) -> None:
    """Plot day-total and sync heatmaps by year."""
    for year in YEARS:
        one = anomaly_df[anomaly_df["year"] == year].copy()
        if one.empty:
            continue

        one["date"] = pd.to_datetime(one["date"])
        one = one.sort_values("date")
        date_labels = [d.strftime("%m-%d") for d in sorted(one["date"].unique())]

        total_pivot = one.pivot(index="treatment", columns="date", values="day_total").reindex(PLOTS)
        sync_pivot = one.pivot(index="treatment", columns="date", values="sync_prop").reindex(PLOTS)

        fig, axes = plt.subplots(2, 1, figsize=(16, 6), constrained_layout=True)
        sns.heatmap(total_pivot, cmap="YlOrRd", ax=axes[0], cbar_kws={"label": "Daily total proximity"})
        axes[0].set_title(f"{year} Plot-Day Total Proximity Heatmap")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Treatment")
        axes[0].set_xticklabels(date_labels, rotation=45, ha="right")

        sns.heatmap(sync_pivot, cmap="Blues", ax=axes[1], cbar_kws={"label": "Sync proportion (>pair q90)"})
        axes[1].set_title(f"{year} Plot-Day Synchronization Heatmap")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Treatment")
        axes[1].set_xticklabels(date_labels, rotation=45, ha="right")

        fig.savefig(FIG_DIR / f"01_day_heatmaps_{year}.png")
        plt.close(fig)


def plot_management_focus(anomaly_df: pd.DataFrame) -> None:
    """Plot focus timeline for High_shade in 2023 with highlighted date."""
    focus = anomaly_df[(anomaly_df["year"] == 2023) & (anomaly_df["treatment"] == "High_shade")].copy()
    if focus.empty:
        return
    focus = focus.sort_values("date")
    focus["date"] = pd.to_datetime(focus["date"])

    fig, ax1 = plt.subplots(figsize=(14, 4.5))
    ax2 = ax1.twinx()

    ax1.plot(focus["date"], focus["day_total"], color="#d94801", marker="o", label="Day total proximity")
    ax2.plot(focus["date"], focus["sync_prop"], color="#225ea8", marker="s", label="Sync proportion")

    flagged = focus[focus["management_anomaly"]]
    ax1.scatter(flagged["date"], flagged["day_total"], color="red", s=45, zorder=5, label="Flagged anomaly day")

    target_date = pd.Timestamp("2023-03-02")
    ax1.axvline(target_date, linestyle="--", color="black", alpha=0.8, linewidth=1.2)
    ax1.text(target_date, ax1.get_ylim()[1] * 0.95, "2023-03-02", ha="left", va="top", fontsize=9)

    ax1.set_title("2023 High_shade: Daily Total vs Synchronization")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Daily total proximity")
    ax2.set_ylabel("Synchronization proportion")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_high_shade_2023_focus.png")
    plt.close(fig)


def plot_mating_overview(final_df: pd.DataFrame, table_dict: Dict[str, pd.DataFrame]) -> None:
    """Plot mating event count, time-zone preference, ram/ewe patterns."""
    result = final_df.copy()
    mating = result[result["Mating"] == "Y"].copy()

    # A) Mating count by year and treatment.
    count_df = (
        mating.groupby(["year", "treatment"], as_index=False)
        .size()
        .rename(columns={"size": "mating_events"})
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=count_df, x="treatment", y="mating_events", hue="year", ax=ax)
    ax.set_title("Inferred Mating Events by Year and Treatment")
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Event count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_mating_events_by_group.png")
    plt.close(fig)

    # B) Time-zone preference.
    tz = table_dict["tz_summary"].copy()
    if not tz.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True, constrained_layout=True)
        for i, year in enumerate(YEARS):
            sub = tz[tz["year"] == year]
            if sub.empty:
                axes[i].set_axis_off()
                continue
            pivot = sub.pivot(index="treatment", columns="time_zone", values="mating_events").fillna(0).reindex(PLOTS)
            pivot.plot(kind="bar", stacked=True, ax=axes[i], colormap="tab20")
            axes[i].set_title(f"{year} Time-Zone Preference")
            axes[i].set_xlabel("Treatment")
            axes[i].set_ylabel("Event count")
            axes[i].legend(loc="upper right", fontsize=8)
        fig.savefig(FIG_DIR / "04_time_zone_preference.png")
        plt.close(fig)

    # C) Ram daily partner distribution.
    ram_day = table_dict["ram_day"].copy()
    if not ram_day.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.boxplot(data=ram_day, x="treatment", y="mating_ewe_count", hue="year", ax=ax)
        ax.set_title("Ram Daily Mating Partner Count Distribution")
        ax.set_xlabel("Treatment")
        ax.set_ylabel("Ewe partners per ram-day")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "05_ram_daily_partner_distribution.png")
        plt.close(fig)

    # D) Ewe mating count distribution.
    ewe = table_dict["ewe_summary"].copy()
    if not ewe.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(data=ewe, x="mating_count", hue="year", bins=12, multiple="dodge", ax=ax)
        ax.set_title("Ewe Total Mating Count Distribution")
        ax.set_xlabel("Inferred mating count per ewe")
        ax.set_ylabel("Ewe count")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "06_ewe_mating_count_distribution.png")
        plt.close(fig)


def _format_flagged_days_for_report(flagged_days: pd.DataFrame) -> str:
    """Build a compact textual summary for anomaly days."""
    if flagged_days.empty:
        return "None."

    parts: List[str] = []
    for (year, treatment), g in flagged_days.groupby(["year", "treatment"]):
        dates = ", ".join(g["date"].tolist())
        parts.append(f"- {year} {treatment}: {dates}")
    return "\n".join(parts)


def _value_from_table(df: pd.DataFrame, filters: Dict[str, object], column: str, default: float = 0.0) -> float:
    """Safely read a scalar value from a table."""
    x = df.copy()
    for k, v in filters.items():
        x = x[x[k] == v]
    if x.empty:
        return default
    return float(x.iloc[0][column])


def write_reports(
    final_df: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    prefilter_audit: pd.DataFrame,
    table_dict: Dict[str, pd.DataFrame],
    split_index: pd.DataFrame,
) -> None:
    """Write Chinese and English Markdown reports."""
    overview = table_dict["overview"]
    flagged_days = table_dict["flagged_days"]
    focus = table_dict["focus_table"]
    mating = final_df[final_df["Mating"] == "Y"].copy()
    total_pairs = int(len(final_df))
    total_events = int(len(mating))
    low_signal_pairs = int(pair_metrics["low_signal_pair"].sum())
    removed_by_cap = int(pair_metrics["candidate_removed_by_cap"].sum())
    prefilter_total = int(len(prefilter_audit))
    prefilter_removed_step2 = int((~prefilter_audit["pass_step2_non_zero_days"]).sum())
    prefilter_removed_step4 = int(
        (
            prefilter_audit["pass_step2_non_zero_days"]
            & (~prefilter_audit["pass_step4_robust_z"])
        ).sum()
    )
    prefilter_kept = int(prefilter_audit["pair_prefilter_keep"].sum())

    hs_0302_sync = _value_from_table(
        anomaly_df, {"year": 2023, "treatment": "High_shade", "date": "2023-03-02"}, "sync_prop"
    )
    hs_0302_rz = _value_from_table(
        anomaly_df, {"year": 2023, "treatment": "High_shade", "date": "2023-03-02"}, "rz_sync"
    )
    hs_0302_flag = bool(
        _value_from_table(
            anomaly_df,
            {"year": 2023, "treatment": "High_shade", "date": "2023-03-02"},
            "management_anomaly",
            default=0.0,
        )
    )
    hs_0215_sync = _value_from_table(
        anomaly_df, {"year": 2023, "treatment": "High_shade", "date": "2023-02-15"}, "sync_prop"
    )
    hs_0215_rz = _value_from_table(
        anomaly_df, {"year": 2023, "treatment": "High_shade", "date": "2023-02-15"}, "rz_sync"
    )
    hs_0215_flag = bool(
        _value_from_table(
            anomaly_df,
            {"year": 2023, "treatment": "High_shade", "date": "2023-02-15"},
            "management_anomaly",
            default=0.0,
        )
    )

    hs_0302_cn = (
        "该日被识别为管理异常日，并在交配判定中排除。"
        if hs_0302_flag
        else "该日未达到管理异常阈值，未作为管理异常日排除。"
    )
    hs_0215_cn = (
        "该日被识别为管理异常日。"
        if hs_0215_flag
        else "该日未达到管理异常阈值。"
    )
    hs_0302_en = (
        "This date is flagged as a management anomaly and excluded from mating inference."
        if hs_0302_flag
        else "This date does not reach the management-anomaly threshold and is not excluded as anomaly."
    )
    hs_0215_en = (
        "This date is flagged as a management anomaly."
        if hs_0215_flag
        else "This date does not reach the management-anomaly threshold."
    )

    event_by_group = (
        split_index[["year", "treatment", "mating_Y"]]
        .sort_values(["year", "treatment"])
        .to_string(index=False)
    )
    flagged_text = _format_flagged_days_for_report(flagged_days)

    # Chinese report.
    cn = f"""# UWA 公羊母羊接近数据探索性分析报告（中文）

## 1. 分析目标
- 基于接近传感器数据，识别可能发生交配的公羊-母羊 pair。
- 识别牧场管理活动（如投喂）导致的群体性异常，避免误判为交配。
- 输出按年份和 paddock 分组的最终结果文件（2 年 x 3 paddock = 6 个文件）。

## 2. 数据与处理范围
- 输入数据：`data/uwa_proximity_data/HS_proximity_2023_day_UWA.csv`、`data/uwa_proximity_data/HS_proximity_2024_day_UWA.csv` 及对应 `4h` 文件。
- 仅纳入完整天：剔除每年首尾不完整日期（2月14日、3月7日），2023 年保留 20 天，2024 年保留 21 天。
- treatment 标准化为：`High_shade`、`Medium_shade`、`No_shade`。

## 3. 异常处理逻辑
### 3.1 pair级预过滤（先执行）
- Step 1：统计每个 pair 的非零实验天数；
- Step 2：过滤 `非零天数 < {PAIR_MIN_NONZERO_DAYS}` 的 pair；
- Step 3：对剩余 pair，取非零天数接近次数，去掉 {PAIR_TRIM_N} 个最大值和 {PAIR_TRIM_N} 个最小值，计算均值；
- Step 4：按 `year + treatment` 对 Step 3 的均值做 Robust z-score，过滤 `|z| > {PAIR_RZ_THRESHOLD:.0f}` 的 pair。
- 过滤结果：初始 **{prefilter_total}**，Step2 去除 **{prefilter_removed_step2}**，Step4 去除 **{prefilter_removed_step4}**，保留 **{prefilter_kept}**。

### 3.2 低信号 pair（数据缺失/弱接触）
- 判定条件：`zero_day_ratio > 0.5` 且 `period_total < 该年该treatment内20%分位数`。
- 结果：共标记低信号 pair **{low_signal_pairs}** / **{total_pairs}**。
- 这些 pair 直接记为 `Mating = N`，避免把随机噪声当作交配。

### 3.3 牧场管理异常日（群体同步冲高）
- 对每个 `year + treatment + day` 计算：
  - `day_total`：该天所有 pair 接近次数总和；
  - `sync_prop`：该天“超过各自90分位”的 pair 占比（群体同步程度）。
- 使用 robust z-score，若 `rz_total >= 3` 或 `rz_sync >= 3` 则标记为管理异常日。
- 标记结果：
{flagged_text}

### 3.4 对你提到的 2023-03-02 High Shade 的验证
- `sync_prop = {hs_0302_sync:.3f}`，`rz_sync = {hs_0302_rz:.3f}`。
- {hs_0302_cn}
- 对比 2023-02-15（同 paddock）`sync_prop = {hs_0215_sync:.3f}`，`rz_sync = {hs_0215_rz:.3f}`。{hs_0215_cn}

## 4. 交配判定规则
- 在剔除管理异常日后，对每个 pair 选最强候选日，需同时满足：
  - `best_day_count >= {MATING_ABS_COUNT_THRESHOLD:.0f}`
  - `best_day_z >= {MATING_SCORE_THRESHOLD:.1f}`
  - `best_day_fc >= {MATING_FC_THRESHOLD:.1f}`
  - `best_day_dominance >= {MATING_DOMINANCE_THRESHOLD:.1f}`
- 生物学约束过滤：
  - 同一 `ram + day` 最多保留 {MAX_EWE_PER_RAM_PER_DAY} 个母羊候选；
  - 同一 `ewe + day` 最多保留 {MAX_RAM_PER_EWE_PER_DAY} 个公羊候选；
- 被约束删除的候选数：**{removed_by_cap}**。

## 5. 结果概览
- 最终推断交配事件：**{total_events}**（pair 级，`Mating=Y`）。
- 各组事件数：
```
{event_by_group}
```

## 6. 公羊与母羊维度发现
- 公羊维度：见 `data/uwa_proximity_eda/tables/ram_day.csv`，以及图 `05_ram_daily_partner_distribution.png`。
- 母羊维度：见 `data/uwa_proximity_eda/tables/ewe_summary.csv`，以及图 `06_ewe_mating_count_distribution.png`。
- 时间段偏好：见 `data/uwa_proximity_eda/tables/tz_summary.csv` 和图 `04_time_zone_preference.png`。

## 7. 可视化与证据文件
- 热图：`data/uwa_proximity_eda/figures/01_day_heatmaps_2023.png`、`data/uwa_proximity_eda/figures/01_day_heatmaps_2024.png`
- 重点异常图：`data/uwa_proximity_eda/figures/02_high_shade_2023_focus.png`
- 交配概览图：`data/uwa_proximity_eda/figures/03_mating_events_by_group.png`
- 时间段偏好图：`data/uwa_proximity_eda/figures/04_time_zone_preference.png`
- 公羊日配对分布：`data/uwa_proximity_eda/figures/05_ram_daily_partner_distribution.png`
- 母羊总配对分布：`data/uwa_proximity_eda/figures/06_ewe_mating_count_distribution.png`

## 8. 最终结果文件（6个）
- 见 `data/uwa_proximity_res/`：
  - `uwa_proximity_mating_2023_High_shade.csv`
  - `uwa_proximity_mating_2023_Medium_shade.csv`
  - `uwa_proximity_mating_2023_No_shade.csv`
  - `uwa_proximity_mating_2024_High_shade.csv`
  - `uwa_proximity_mating_2024_Medium_shade.csv`
  - `uwa_proximity_mating_2024_No_shade.csv`
"""

    # English report.
    en = f"""# UWA Ram-Ewe Proximity EDA and Anomaly-Aware Mating Inference (English)

## 1. Objectives
- Detect likely mating events from ram-ewe proximity signals.
- Separate biologically plausible mating spikes from flock-management-driven synchronization.
- Export final year-by-paddock outputs (2 years x 3 paddocks = 6 files).

## 2. Data Scope
- Inputs: day-level and 4-hour-level files in `data/uwa_proximity_data/`.
- Full-day window only: first and last partial days were excluded (Feb 14 and Mar 7).
- Treatment normalization: `High_shade`, `Medium_shade`, `No_shade`.

## 3. Anomaly Logic
### 3.1 Pair pre-filtering (first stage)
- Step 1: count non-zero experiment days per pair.
- Step 2: remove pairs with `non_zero_days < {PAIR_MIN_NONZERO_DAYS}`.
- Step 3: on remaining pairs, compute non-zero trimmed mean
  by removing {PAIR_TRIM_N} largest and {PAIR_TRIM_N} smallest non-zero values.
- Step 4: apply robust z-score on Step 3 metric by `year + treatment`,
  remove pairs with `|z| > {PAIR_RZ_THRESHOLD:.0f}`.
- Filtering result: start **{prefilter_total}**, removed by Step 2 **{prefilter_removed_step2}**, removed by Step 4 **{prefilter_removed_step4}**, kept **{prefilter_kept}**.

### 3.2 Low-signal pair handling
- Rule: `zero_day_ratio > 0.5` and `period_total < 20th percentile` within each year-treatment group.
- Result: **{low_signal_pairs}** low-signal pairs out of **{total_pairs}**, forced to `Mating = N`.

### 3.3 Management anomaly day mining
- For each `year + treatment + day`, two indicators were used:
  - `day_total`: total proximity counts across all pairs.
  - `sync_prop`: share of pairs above their own q90 on that day.
- Day is flagged if `rz_total >= 3` or `rz_sync >= 3`.
- Flagged days:
{flagged_text}

### 3.4 Validation of the user-observed day (2023-03-02, High Shade)
- `sync_prop = {hs_0302_sync:.3f}`, `rz_sync = {hs_0302_rz:.3f}`.
- {hs_0302_en}
- Reference day 2023-02-15: `sync_prop = {hs_0215_sync:.3f}`, `rz_sync = {hs_0215_rz:.3f}`. {hs_0215_en}

## 4. Mating Inference Rule
- After removing management-anomaly days, each pair's strongest candidate day is tested:
  - `best_day_count >= {MATING_ABS_COUNT_THRESHOLD:.0f}`
  - `best_day_z >= {MATING_SCORE_THRESHOLD:.1f}`
  - `best_day_fc >= {MATING_FC_THRESHOLD:.1f}`
  - `best_day_dominance >= {MATING_DOMINANCE_THRESHOLD:.1f}`
- Biological plausibility caps:
  - max {MAX_EWE_PER_RAM_PER_DAY} ewes per `ram + day`
  - max {MAX_RAM_PER_EWE_PER_DAY} rams per `ewe + day`
- Candidates removed by caps: **{removed_by_cap}**.

## 5. Key Results
- Final inferred mating events (`Mating=Y`): **{total_events}**.
- Event counts by group:
```
{event_by_group}
```

## 6. Ram/Ewe/Time-Zone Insights
- Ram daily partner behavior: `data/uwa_proximity_eda/tables/ram_day.csv` and Figure `05_ram_daily_partner_distribution.png`.
- Ewe mating frequency and partner diversity: `data/uwa_proximity_eda/tables/ewe_summary.csv` and Figure `06_ewe_mating_count_distribution.png`.
- Time-zone preference: `data/uwa_proximity_eda/tables/tz_summary.csv` and Figure `04_time_zone_preference.png`.

## 7. Evidence Outputs
- Heatmaps: `data/uwa_proximity_eda/figures/01_day_heatmaps_2023.png`, `data/uwa_proximity_eda/figures/01_day_heatmaps_2024.png`
- Focus anomaly chart: `data/uwa_proximity_eda/figures/02_high_shade_2023_focus.png`
- Mating overview chart: `data/uwa_proximity_eda/figures/03_mating_events_by_group.png`
- Time-zone preference chart: `data/uwa_proximity_eda/figures/04_time_zone_preference.png`
- Ram partner distribution chart: `data/uwa_proximity_eda/figures/05_ram_daily_partner_distribution.png`
- Ewe mating-count distribution chart: `data/uwa_proximity_eda/figures/06_ewe_mating_count_distribution.png`

## 8. Final Files (6)
- In `data/uwa_proximity_res/`:
  - `uwa_proximity_mating_2023_High_shade.csv`
  - `uwa_proximity_mating_2023_Medium_shade.csv`
  - `uwa_proximity_mating_2023_No_shade.csv`
  - `uwa_proximity_mating_2024_High_shade.csv`
  - `uwa_proximity_mating_2024_Medium_shade.csv`
  - `uwa_proximity_mating_2024_No_shade.csv`
"""

    (REPORT_DIR / "uwa_proximity_eda_cn.md").write_text(cn, encoding="utf-8")
    (REPORT_DIR / "uwa_proximity_eda_en.md").write_text(en, encoding="utf-8")

    # Also save the focus table for direct inspection.
    focus.to_csv(TABLE_DIR / "focus_high_shade_2023.csv", index=False)


def run_pipeline() -> None:
    """Execute end-to-end EDA + inference pipeline."""
    ensure_dirs([EDA_DIR, RES_DIR, FIG_DIR, TABLE_DIR, REPORT_DIR])
    setup_plot_style()

    raw_bundles = {year: load_year_data(year) for year in YEARS}
    bundles: Dict[int, YearData] = {}

    anomaly_parts = []
    pair_parts = []
    prefilter_parts = []
    for year in YEARS:
        filtered_bundle, prefilter_df = prefilter_pairs_before_anomaly(raw_bundles[year])
        bundles[year] = filtered_bundle
        anomaly_df = detect_management_anomalies(filtered_bundle)
        pair_df = infer_pair_candidates(filtered_bundle, anomaly_df)
        anomaly_parts.append(anomaly_df)
        pair_parts.append(pair_df)
        prefilter_parts.append(prefilter_df)

    anomaly_all = pd.concat(anomaly_parts, ignore_index=True)
    pair_all = pd.concat(pair_parts, ignore_index=True)
    prefilter_all = pd.concat(prefilter_parts, ignore_index=True)
    pair_all = apply_biological_caps(pair_all)

    final_df = build_final_results(bundles, pair_all)
    split_index = save_split_result_files(final_df)

    tables = build_summary_tables(final_df, pair_all, anomaly_all, prefilter_all)
    save_tables(final_df, pair_all, anomaly_all, prefilter_all, tables, split_index)

    plot_day_heatmaps(anomaly_all)
    plot_management_focus(anomaly_all)
    plot_mating_overview(final_df, tables)

    write_reports(final_df, pair_all, anomaly_all, prefilter_all, tables, split_index)

    # Console summary for quick verification.
    total_pairs = len(final_df)
    total_mating = int((final_df["Mating"] == "Y").sum())
    prefilter_kept = int(prefilter_all["pair_prefilter_keep"].sum())
    prefilter_removed = int((~prefilter_all["pair_prefilter_keep"]).sum())
    print(f"Total pairs: {total_pairs}")
    print(f"Pre-filter removed pairs: {prefilter_removed}")
    print(f"Pre-filter kept pairs: {prefilter_kept}")
    print(f"Total inferred mating events (Y): {total_mating}")
    print(f"Low-signal pairs: {int(pair_all['low_signal_pair'].sum())}")
    print(f"Management anomaly days: {int(anomaly_all['management_anomaly'].sum())}")
    print("Saved final result files:")
    for _, row in split_index.iterrows():
        print(f"- {row['file']} | rows={row['rows']} | mating_Y={row['mating_Y']}")
    print("Saved reports:")
    print(f"- {REPORT_DIR / 'uwa_proximity_eda_cn.md'}")
    print(f"- {REPORT_DIR / 'uwa_proximity_eda_en.md'}")


if __name__ == "__main__":
    run_pipeline()
