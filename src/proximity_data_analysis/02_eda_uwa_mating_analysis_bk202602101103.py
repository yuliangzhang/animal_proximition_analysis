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


# ----------------------------
# Paths and constants
# ----------------------------
DATA_DIR = Path("data/uwa_proximity_data")
EDA_DIR = Path("data/uwa_proximity_eda")
RES_DIR = Path("data/uwa_proximity_res")
TABLE_DIR = EDA_DIR / "tables"

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


def build_parameter_table() -> pd.DataFrame:
    """Save analysis parameters for reproducible reporting."""
    params = [
        ("PAIR_MIN_NONZERO_DAYS", PAIR_MIN_NONZERO_DAYS, "Pair pre-filter Step 2 threshold"),
        ("PAIR_TRIM_N", PAIR_TRIM_N, "Trim count from each tail in non-zero mean"),
        ("PAIR_RZ_THRESHOLD", PAIR_RZ_THRESHOLD, "Pair pre-filter Step 4 robust z-score threshold"),
        ("LOW_SIGNAL_ZERO_RATIO_THRESHOLD", LOW_SIGNAL_ZERO_RATIO_THRESHOLD, "Low-signal zero-day ratio"),
        ("LOW_SIGNAL_TOTAL_QUANTILE", LOW_SIGNAL_TOTAL_QUANTILE, "Low-signal period total quantile"),
        ("MANAGEMENT_RZ_THRESHOLD", MANAGEMENT_RZ_THRESHOLD, "Management anomaly robust z threshold"),
        ("MATING_SCORE_THRESHOLD", MATING_SCORE_THRESHOLD, "Candidate day robust score threshold"),
        ("MATING_FC_THRESHOLD", MATING_FC_THRESHOLD, "Candidate day fold-change threshold"),
        ("MATING_DOMINANCE_THRESHOLD", MATING_DOMINANCE_THRESHOLD, "Candidate day dominance threshold"),
        ("MATING_ABS_COUNT_THRESHOLD", MATING_ABS_COUNT_THRESHOLD, "Candidate day absolute count threshold"),
        ("MAX_EWE_PER_RAM_PER_DAY", MAX_EWE_PER_RAM_PER_DAY, "Biological cap for ram per day"),
        ("MAX_RAM_PER_EWE_PER_DAY", MAX_RAM_PER_EWE_PER_DAY, "Biological cap for ewe per day"),
    ]
    return pd.DataFrame(params, columns=["parameter", "value", "description"])


def run_pipeline() -> None:
    """Execute end-to-end EDA + inference pipeline."""
    ensure_dirs([EDA_DIR, RES_DIR, TABLE_DIR])

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
    build_parameter_table().to_csv(TABLE_DIR / "analysis_parameters.csv", index=False)

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
    print("Saved analysis tables:")
    print(f"- {TABLE_DIR}")


if __name__ == "__main__":
    run_pipeline()
