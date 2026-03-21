from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("data/muresk_farm_proximity_data/gps_proximition_data")
REPORT_DIR = BASE_DIR / "reports"
ANALYSIS_DIR = BASE_DIR / "analysis"

GLOBAL_SUMMARY_CSV = REPORT_DIR / "gps_proximity_relation_report_summary.csv"
SELECTED_SUMMARY_CSV = REPORT_DIR / "selected_gps_proximity_relation_summary.csv"
AVAILABILITY_CSV = ANALYSIS_DIR / "task11_selected_pair_availability.csv"

OUTPUT_REPORT = REPORT_DIR / "can_gps_replace_proximity_loggers_report.md"

SELECTED_PAIRS = ["N172_N171", "N172_N182", "N172_N189"]
TARGET_WINDOW_MIN = 240
TARGET_DISTANCE_M = 15
GRID_WINDOWS = [30, 60, 90, 120, 150, 180, 210, 240]
GRID_THRESHOLDS = [10, 15, 20, 25, 30, 35, 40, 45, 50]
LAG_SEARCH_MIN = 1440
LAG_STEP_MIN = 5
ROBUST_DISTANCE_VALID_RATIO = 0.70


def fmt_num(v: object, digits: int = 3) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def fmt_pct(v: object, digits: int = 1) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    try:
        return f"{float(v) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(v)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = [header, sep]
    for row in df.itertuples(index=False):
        vals = ["" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in [GLOBAL_SUMMARY_CSV, SELECTED_SUMMARY_CSV, AVAILABILITY_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    global_df = pd.read_csv(GLOBAL_SUMMARY_CSV)
    selected_df = pd.read_csv(SELECTED_SUMMARY_CSV)
    availability_df = pd.read_csv(AVAILABILITY_CSV)

    global_df = global_df[global_df["pair_id"].isin(SELECTED_PAIRS)].copy()
    selected_df = selected_df[selected_df["pair_id"].isin(SELECTED_PAIRS)].copy()
    availability_df = availability_df[availability_df["pair_id"].isin(SELECTED_PAIRS)].copy()

    return global_df, selected_df, availability_df


def build_global_grid_table(global_df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "pair_id",
        "best_window_min",
        "best_distance_threshold_m",
        "best_lag_minutes",
        "best_pearson_corr",
        "lag0_pearson_corr",
        "corr_improvement",
        "plausible_window_min",
        "plausible_threshold_m",
        "plausible_lag_minutes",
        "plausible_pearson_corr",
    ]
    t = global_df[keep].copy()
    t = t.sort_values("pair_id", kind="stable")
    t["best_pearson_corr"] = t["best_pearson_corr"].map(lambda x: fmt_num(x, 3))
    t["lag0_pearson_corr"] = t["lag0_pearson_corr"].map(lambda x: fmt_num(x, 3))
    t["corr_improvement"] = t["corr_improvement"].map(lambda x: fmt_num(x, 3))
    t["plausible_pearson_corr"] = t["plausible_pearson_corr"].map(lambda x: fmt_num(x, 3))
    return t


def build_selected_eval_table(selected_df: pd.DataFrame) -> pd.DataFrame:
    use = selected_df[selected_df["analysis_set"].isin(["full", "quality_no_midday"])].copy()
    use = use.sort_values(["pair_id", "analysis_set"], kind="stable")
    use["analysis_set"] = use["analysis_set"].map(
        {"full": "full", "quality_no_midday": "robust_no_midday"}
    )
    keep = [
        "pair_id",
        "beacon_collar_design",
        "analysis_set",
        "lag0_corr",
        "best_corr",
        "corr_gain",
        "best_lag_min",
        "lag0_points",
        "best_points",
        "bucket_kept_ratio",
    ]
    t = use[keep].copy()
    for c in ["lag0_corr", "best_corr", "corr_gain"]:
        t[c] = t[c].map(lambda x: fmt_num(x, 3))
    t["bucket_kept_ratio"] = t["bucket_kept_ratio"].map(lambda x: fmt_pct(x, 1))
    return t


def build_availability_table(availability_df: pd.DataFrame) -> pd.DataFrame:
    t = availability_df.sort_values("pair_id", kind="stable").copy()
    keep = [
        "pair_id",
        "beacon_collar_design",
        "avg_distance_valid_ratio_all",
        "avg_distance_valid_ratio_12_16",
        "avg_distance_valid_ratio_other",
    ]
    t = t[keep]
    for c in ["avg_distance_valid_ratio_all", "avg_distance_valid_ratio_12_16", "avg_distance_valid_ratio_other"]:
        t[c] = t[c].map(lambda x: fmt_pct(x, 1))
    return t


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    global_df, selected_df, availability_df = load_inputs()

    global_table = build_global_grid_table(global_df)
    selected_table = build_selected_eval_table(selected_df)
    availability_table = build_availability_table(availability_df)

    robust = selected_df[selected_df["analysis_set"] == "quality_no_midday"].copy()
    if robust.empty:
        robust_mean = np.nan
        robust_large_lag_ratio = np.nan
    else:
        robust_mean = float(robust["best_corr"].mean())
        robust_large_lag_ratio = float((robust["best_lag_min"].abs() > 360).mean())

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    lines.append("# Can GPS Devices Replace Proximity Loggers for Detecting Mating Behavior?")
    lines.append("")
    lines.append(f"- Generated at: `{generated_at}`")
    lines.append("- Data source: `Muresk Farm Experimental Data provided by Luoyang and Peter`")
    lines.append("- Focus pairs: `N172_N171`, `N172_N182`, `N172_N189`")
    lines.append("")

    lines.append("## 1. Objective")
    lines.append("")
    lines.append(
        "This report evaluates whether GPS-based distance signals can replace expensive proximity loggers for "
        "mating-behavior detection. The core question is whether high proximity activity and GPS short-distance "
        "activity are strongly aligned after accounting for device clock misalignment."
    )
    lines.append("")

    lines.append("## 2. Data and Processing Pipeline")
    lines.append("")
    lines.append("- Proximity logger sampling interval: `1 minute`.")
    lines.append("- GPS sampling interval: `5 minutes`.")
    lines.append(
        "- Pair-level analysis data were built from receiver-beacon proximity counts and synchronized pairwise GPS distances."
    )
    lines.append(
        "- Raw timestamps were aggregated to common buckets before correlation analysis to reduce short-term noise."
    )
    lines.append("")
    lines.append("### Metric Definitions")
    lines.append("")
    lines.append("- `proximity_rate = proximity_sum / proximity_valid_minutes`")
    lines.append("- `gps_close_ratio = distance_in_range_count / distance_valid_count`")
    lines.append(
        f"- In this final report, `distance_in_range` is defined as `distance <= {TARGET_DISTANCE_M}m`."
    )
    lines.append(
        f"- Quality control for robust analysis: keep buckets with `distance_valid_ratio >= {int(ROBUST_DISTANCE_VALID_RATIO * 100)}%`; "
        "for single-collar pairs, remove the high-loss period `12:00-16:00`."
    )
    lines.append("")

    lines.append("## 3. Grid Search Design (Task 9 Scope)")
    lines.append("")
    lines.append(
        f"- Time window grid: `{', '.join(str(x) for x in GRID_WINDOWS)} min`."
    )
    lines.append(
        f"- Distance threshold grid: `{', '.join(str(x) for x in GRID_THRESHOLDS)} m`."
    )
    lines.append(
        f"- Lag scan for alignment: `±{LAG_SEARCH_MIN} min`, step `{LAG_STEP_MIN} min` (GPS fixed, proximity shifted)."
    )
    lines.append(
        "- Selection rationale for final operating setup: choose a coarse window (`240 min`) for stability and "
        "choose `15 m` as a practical close-contact threshold, considering typical GPS error near ~10 m."
    )
    lines.append("")

    lines.append("## 4. Key Results for Selected Pairs")
    lines.append("")
    lines.append("### 4.1 From Global Grid Summary (Best vs Plausible-Lag)")
    lines.append("")
    lines.append(md_table(global_table))
    lines.append("")
    lines.append(
        "Interpretation note: `best_lag` often approaches ~24h, which can be caused by true clock/date offset but can "
        "also be caused by daily behavior periodicity. Therefore, lag-adjusted high correlation alone is not enough "
        "for replacement claims."
    )
    lines.append("")

    lines.append(f"### 4.2 Fixed Setup Evaluation (`window={TARGET_WINDOW_MIN}min`, `distance<={TARGET_DISTANCE_M}m`)")
    lines.append("")
    lines.append(md_table(selected_table))
    lines.append("")
    lines.append("### 4.3 GPS Availability Risk")
    lines.append("")
    lines.append(md_table(availability_table))
    lines.append("")
    lines.append(
        "Single-collar pairs (`N172_N182`, `N172_N189`) show clear midday (`12:00-16:00`) validity drops, which can "
        "create false disagreement: high proximity but low GPS close ratio due to missing GPS."
    )
    lines.append("")

    lines.append("## 5. Figures")
    lines.append("")
    lines.append("### 5.1 Window-240min Trend Figures (from grid-search outputs)")
    lines.append("")
    for pair_id in SELECTED_PAIRS:
        lines.append(f"#### {pair_id} (`<=15m`)")
        lines.append("")
        lines.append(f"![{pair_id} window240](../figures/window_240min/{pair_id}_le_15m.png)")
        lines.append("")

    lines.append("### 5.2 Robust Alignment Diagnostics")
    lines.append("")
    lines.append("![GPS availability](figures_selected/availability_by_pair.png)")
    lines.append("")
    lines.append("![Correlation full vs robust](figures_selected/correlation_full_vs_robust.png)")
    lines.append("")
    for pair_id in SELECTED_PAIRS:
        lines.append(f"#### {pair_id} (raw vs robust + lag-adjusted)")
        lines.append("")
        lines.append(f"![{pair_id} robust](figures_selected/{pair_id}_raw_vs_robust.png)")
        lines.append("")

    lines.append("## 6. Decision-Oriented Conclusion")
    lines.append("")
    lines.append(
        f"- Robust mean best correlation across selected pairs: `{fmt_num(robust_mean, 3)}`."
    )
    lines.append(
        f"- Share of robust results requiring very large lag (`|lag| > 360 min`): `{fmt_pct(robust_large_lag_ratio, 1)}`."
    )
    lines.append(
        "- Current evidence supports that GPS can track broad contact trends, but direct one-to-one replacement is "
        "not yet defensible without time-synchronization calibration and strict missing-data controls."
    )
    lines.append(
        "- Practical recommendation: use GPS as the main scalable signal for screening at coarse windows, while "
        "keeping proximity logger as reference for calibration and event-level validation."
    )
    lines.append("")

    lines.append("## 7. Risks, Limits, and What Was Corrected")
    lines.append("")
    lines.append(
        "- Important correction: a very high lag-adjusted correlation can be inflated by daily periodicity; "
        "it is treated as supportive evidence, not standalone proof of replacement feasibility."
    )
    lines.append(
        "- Single-collar missingness (especially `12:00-16:00`) materially affects reliability and was explicitly handled."
    )
    lines.append(
        "- Final recommendation is based on both alignment quality and data validity, not correlation alone."
    )
    lines.append("")

    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
