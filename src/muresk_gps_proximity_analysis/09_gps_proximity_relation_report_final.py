from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path("data/muresk_farm_proximity_data/gps_proximition_data")
REPORT_DIR = BASE_DIR / "reports"
ANALYSIS_DIR = BASE_DIR / "analysis"

PAIR_SUMMARY_FILE = BASE_DIR / "pair_distance_summary.csv"

SELECTED_PAIRS = ["N172_N171", "N172_N182", "N172_N189"]
AGGREGATE_WINDOW_MIN = 240
DISTANCE_THRESHOLD_M = 15

LAG_MAX_MIN = 1440  # allow severe clock misalignment up to 24h
LAG_STEP_MIN = 5
BASE_SAMPLE_MIN = 5.0

# Missingness control for robustness analysis
MIN_DISTANCE_VALID_RATIO_PER_BUCKET = 0.70
SUSPECT_HOURS = {12, 13, 14, 15}

REPORT_FILE = REPORT_DIR / "selected_gps_proximity_relation_report.md"
SUMMARY_CSV = REPORT_DIR / "selected_gps_proximity_relation_summary.csv"
FIG_DIR = REPORT_DIR / "figures_selected"


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No data._"
    view = df.copy().head(max_rows)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = [header, sep]
    for row in view.itertuples(index=False):
        rows.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    if len(df) > max_rows:
        rows.append(f"\n_Only first {max_rows} rows shown (total {len(df)})._")
    return "\n".join(rows)


def load_selected_pair_meta() -> pd.DataFrame:
    if not PAIR_SUMMARY_FILE.exists():
        raise FileNotFoundError(f"Missing pair summary: {PAIR_SUMMARY_FILE}")
    pair_summary = pd.read_csv(PAIR_SUMMARY_FILE)
    pair_summary["pair_id"] = pair_summary["receiver_gps_id"] + "_" + pair_summary["beacon_gps_id"]
    selected = pair_summary[pair_summary["pair_id"].isin(SELECTED_PAIRS)].copy()
    if len(selected) != len(SELECTED_PAIRS):
        missing = sorted(set(SELECTED_PAIRS) - set(selected["pair_id"]))
        raise ValueError(f"Selected pair files missing in summary: {missing}")
    return selected


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
        raise ValueError(f"No valid records in pair file: {path}")
    return df


def aggregate_pair(df: pd.DataFrame, lag_min: int) -> pd.DataFrame:
    index = pd.DatetimeIndex(df["time_slot"])
    series_p = pd.Series(df["proximity_count"].to_numpy(), index=index)
    series_p_valid_min = pd.Series(
        np.where(series_p.notna(), BASE_SAMPLE_MIN, 0.0),
        index=index,
        dtype="float64",
    )
    series_d = pd.Series(df["distance_m"].to_numpy(), index=index)

    if lag_min != 0:
        shift = pd.Timedelta(minutes=lag_min)
        series_p.index = series_p.index + shift
        series_p_valid_min.index = series_p_valid_min.index + shift

    freq = f"{AGGREGATE_WINDOW_MIN}min"
    proximity_sum = series_p.resample(freq).sum(min_count=1)
    proximity_valid_min = series_p_valid_min.resample(freq).sum(min_count=1)
    proximity_rate = (proximity_sum / proximity_valid_min).where(proximity_valid_min > 0)

    distance_valid_count = series_d.notna().astype("int64").resample(freq).sum(min_count=1)
    distance_in_range_count = ((series_d <= DISTANCE_THRESHOLD_M) & series_d.notna()).astype("int64").resample(freq).sum(min_count=1)
    distance_in_range_ratio = (distance_in_range_count / distance_valid_count).where(distance_valid_count > 0)

    expected_points = int(AGGREGATE_WINDOW_MIN / BASE_SAMPLE_MIN)
    distance_valid_ratio = distance_valid_count / expected_points

    out = pd.DataFrame(
        {
            "time_slot": proximity_sum.index,
            "proximity_sum": proximity_sum.values,
            "proximity_valid_minutes": proximity_valid_min.values,
            "proximity_rate": proximity_rate.values,
            "distance_valid_count": distance_valid_count.reindex(proximity_sum.index).values,
            "distance_in_range_count": distance_in_range_count.reindex(proximity_sum.index).values,
            "distance_in_range_ratio": distance_in_range_ratio.reindex(proximity_sum.index).values,
            "distance_valid_ratio": distance_valid_ratio.reindex(proximity_sum.index).values,
        }
    )
    return out


def pearson_corr(a: pd.Series, b: pd.Series) -> tuple[float | None, int]:
    t = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(t) < 8 or t["a"].nunique() < 2 or t["b"].nunique() < 2:
        return None, int(len(t))
    val = t["a"].corr(t["b"], method="pearson")
    return (float(val) if pd.notna(val) else None), int(len(t))


def high_low_delta(a: pd.Series, b: pd.Series) -> float | None:
    t = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(t) < 12:
        return None
    q75 = t["a"].quantile(0.75)
    q25 = t["a"].quantile(0.25)
    hi = t.loc[t["a"] >= q75, "b"]
    lo = t.loc[t["a"] <= q25, "b"]
    if len(hi) < 3 or len(lo) < 3:
        return None
    return float(hi.median() - lo.median())


def scan_best_lag(df_raw: pd.DataFrame, mask: pd.Series | None) -> dict:
    lags = list(range(-LAG_MAX_MIN, LAG_MAX_MIN + 1, LAG_STEP_MIN))
    best = None
    lag0 = None

    for lag in lags:
        agg = aggregate_pair(df_raw, lag)
        if mask is not None:
            agg = agg[mask.reindex(agg.index, fill_value=False).to_numpy()]
        corr, n = pearson_corr(agg["proximity_rate"], agg["distance_in_range_ratio"])
        if lag == 0:
            lag0 = {"lag_min": 0, "corr": corr, "n": n}
        if corr is None:
            continue
        if (best is None) or (corr > best["corr"]):
            best = {"lag_min": lag, "corr": corr, "n": n, "agg": agg}

    if lag0 is None:
        lag0 = {"lag_min": 0, "corr": None, "n": 0}
    if best is None:
        best = {"lag_min": None, "corr": None, "n": 0, "agg": None}
    return {"lag0": lag0, "best": best}


def build_masks(base_agg_lag0: pd.DataFrame, is_single: bool) -> dict[str, pd.Series]:
    idx = pd.RangeIndex(len(base_agg_lag0))
    full = pd.Series(True, index=idx)
    quality = base_agg_lag0["distance_valid_ratio"] >= MIN_DISTANCE_VALID_RATIO_PER_BUCKET
    if is_single:
        no_midday = ~base_agg_lag0["time_slot"].dt.hour.isin(SUSPECT_HOURS)
    else:
        no_midday = full.copy()
    quality_no_midday = quality & no_midday
    return {
        "full": full,
        "quality_only": quality,
        "quality_no_midday": quality_no_midday,
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    selected_meta = load_selected_pair_meta().sort_values("pair_id", kind="stable")

    rows = []
    availability_rows = []
    pair_plot_ctx: dict[str, dict] = {}

    for row in selected_meta.itertuples(index=False):
        pair_id = row.pair_id
        design = str(row.beacon_collar_design).lower()
        is_single = design == "single"
        df_raw = load_pair_raw(pair_id)

        base_agg_lag0 = aggregate_pair(df_raw, 0).reset_index(drop=True)
        masks = build_masks(base_agg_lag0, is_single=is_single)

        availability_rows.append(
            {
                "pair_id": pair_id,
                "beacon_collar_design": design,
                "avg_distance_valid_ratio_all": float(base_agg_lag0["distance_valid_ratio"].mean()),
                "avg_distance_valid_ratio_12_16": float(
                    base_agg_lag0.loc[base_agg_lag0["time_slot"].dt.hour.isin(SUSPECT_HOURS), "distance_valid_ratio"].mean()
                ),
                "avg_distance_valid_ratio_other": float(
                    base_agg_lag0.loc[~base_agg_lag0["time_slot"].dt.hour.isin(SUSPECT_HOURS), "distance_valid_ratio"].mean()
                ),
            }
        )

        for analysis_set, mask in masks.items():
            scan = scan_best_lag(df_raw=df_raw, mask=mask)
            best_agg = scan["best"]["agg"]
            lag0_agg = aggregate_pair(df_raw, 0)
            if mask is not None:
                lag0_agg = lag0_agg[mask.reindex(lag0_agg.index, fill_value=False).to_numpy()]
                if best_agg is not None:
                    best_agg = best_agg[mask.reindex(best_agg.index, fill_value=False).to_numpy()]

            delta = None
            if best_agg is not None:
                delta = high_low_delta(best_agg["proximity_rate"], best_agg["distance_in_range_ratio"])

            rows.append(
                {
                    "pair_id": pair_id,
                    "beacon_collar_design": design,
                    "analysis_set": analysis_set,
                    "window_min": AGGREGATE_WINDOW_MIN,
                    "distance_threshold_m": DISTANCE_THRESHOLD_M,
                    "lag0_corr": scan["lag0"]["corr"],
                    "lag0_points": scan["lag0"]["n"],
                    "best_lag_min": scan["best"]["lag_min"],
                    "best_corr": scan["best"]["corr"],
                    "best_points": scan["best"]["n"],
                    "corr_gain": (
                        (scan["best"]["corr"] - scan["lag0"]["corr"])
                        if (scan["best"]["corr"] is not None and scan["lag0"]["corr"] is not None)
                        else None
                    ),
                    "high_low_delta_gps_ratio": delta,
                    "bucket_kept_ratio": float(mask.mean()),
                }
            )

            if analysis_set == "full":
                pair_plot_ctx.setdefault(pair_id, {})
                pair_plot_ctx[pair_id]["full_best_lag"] = scan["best"]["lag_min"]
            if analysis_set == "quality_no_midday":
                pair_plot_ctx.setdefault(pair_id, {})
                pair_plot_ctx[pair_id]["robust_best_lag"] = scan["best"]["lag_min"]
                pair_plot_ctx[pair_id]["robust_mask"] = mask.copy()

        pair_plot_ctx.setdefault(pair_id, {})
        pair_plot_ctx[pair_id]["df_raw"] = df_raw.copy()
        pair_plot_ctx[pair_id]["design"] = design

    result_df = pd.DataFrame(rows).sort_values(["pair_id", "analysis_set"], kind="stable")
    availability_df = pd.DataFrame(availability_rows).sort_values("pair_id", kind="stable")

    result_df.to_csv(ANALYSIS_DIR / "task11_selected_pair_shift_eval.csv", index=False, encoding="utf-8")
    availability_df.to_csv(ANALYSIS_DIR / "task11_selected_pair_availability.csv", index=False, encoding="utf-8")
    result_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")

    full_df = result_df[result_df["analysis_set"] == "full"].copy()
    robust_df = result_df[result_df["analysis_set"] == "quality_no_midday"].copy()

    # -------- Figure 1: availability comparison --------
    fig_avail, ax_avail = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(availability_df))
    width = 0.25
    ax_avail.bar(x - width, availability_df["avg_distance_valid_ratio_all"], width, label="All day")
    ax_avail.bar(x, availability_df["avg_distance_valid_ratio_12_16"], width, label="12:00-16:00")
    ax_avail.bar(x + width, availability_df["avg_distance_valid_ratio_other"], width, label="Other hours")
    ax_avail.set_xticks(x)
    ax_avail.set_xticklabels(availability_df["pair_id"], rotation=20)
    ax_avail.set_ylim(0, 1.05)
    ax_avail.set_ylabel("GPS valid ratio")
    ax_avail.set_title("GPS Availability by Pair and Time Segment")
    ax_avail.grid(axis="y", alpha=0.25)
    ax_avail.legend()
    fig_avail.tight_layout()
    avail_fig_path = FIG_DIR / "availability_by_pair.png"
    fig_avail.savefig(avail_fig_path, dpi=180)
    plt.close(fig_avail)

    # -------- Figure 2: correlation comparison --------
    cmp_df = (
        full_df[["pair_id", "best_corr"]]
        .rename(columns={"best_corr": "full_best_corr"})
        .merge(
            robust_df[["pair_id", "best_corr"]].rename(columns={"best_corr": "robust_best_corr"}),
            on="pair_id",
            how="left",
        )
        .sort_values("pair_id", kind="stable")
    )
    fig_corr, ax_corr = plt.subplots(figsize=(8.5, 5))
    x2 = np.arange(len(cmp_df))
    w2 = 0.35
    ax_corr.bar(x2 - w2 / 2, cmp_df["full_best_corr"], w2, label="Full")
    ax_corr.bar(x2 + w2 / 2, cmp_df["robust_best_corr"], w2, label="Robust filtered")
    ax_corr.set_xticks(x2)
    ax_corr.set_xticklabels(cmp_df["pair_id"], rotation=20)
    ax_corr.set_ylim(0, 1.0)
    ax_corr.set_ylabel("Best Pearson correlation")
    ax_corr.set_title("Correlation Improvement After Robust Filtering")
    ax_corr.grid(axis="y", alpha=0.25)
    ax_corr.legend()
    fig_corr.tight_layout()
    corr_fig_path = FIG_DIR / "correlation_full_vs_robust.png"
    fig_corr.savefig(corr_fig_path, dpi=180)
    plt.close(fig_corr)

    # -------- Figure 3+: per-pair trend plots --------
    pair_fig_paths: dict[str, Path] = {}
    for pair_id in SELECTED_PAIRS:
        ctx = pair_plot_ctx.get(pair_id, {})
        if not ctx:
            continue
        df_raw = ctx.get("df_raw")
        if df_raw is None:
            continue
        full_lag = int(ctx.get("full_best_lag") or 0)
        robust_lag = int(ctx.get("robust_best_lag") or 0)
        robust_mask = ctx.get("robust_mask")
        if robust_mask is None:
            robust_mask = pd.Series(True, index=pd.RangeIndex(len(aggregate_pair(df_raw, 0))))

        agg_raw = aggregate_pair(df_raw, 0).reset_index(drop=True)
        agg_full = aggregate_pair(df_raw, full_lag).reset_index(drop=True)
        agg_rob = aggregate_pair(df_raw, robust_lag).reset_index(drop=True)

        if len(robust_mask) != len(agg_rob):
            robust_mask = pd.Series(True, index=pd.RangeIndex(len(agg_rob)))

        fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.8), sharex=True)
        ax_top, ax_bot = axes

        # Top: raw
        t0 = agg_raw.dropna(subset=["proximity_rate", "distance_in_range_ratio"])
        ax_top.plot(t0["time_slot"], t0["proximity_rate"], color="#b45309", linewidth=1.5, label="Proximity rate (raw)")
        ax_top_b = ax_top.twinx()
        ax_top_b.plot(
            t0["time_slot"],
            t0["distance_in_range_ratio"],
            color="#1d4ed8",
            linewidth=1.5,
            label=f"GPS close ratio <= {DISTANCE_THRESHOLD_M:.0f}m",
        )
        ax_top.set_ylim(0, 1.0)
        ax_top_b.set_ylim(0, 1.0)
        ax_top.set_ylabel("Proximity rate")
        ax_top_b.set_ylabel("GPS close ratio")
        ax_top.grid(alpha=0.25)
        top_lines = ax_top.get_lines() + ax_top_b.get_lines()
        ax_top.legend(top_lines, [l.get_label() for l in top_lines], loc="upper right")

        # Bottom: robust-filtered + lag adjusted
        keep = robust_mask.to_numpy(dtype=bool)
        t1 = agg_rob.loc[keep].dropna(subset=["proximity_rate", "distance_in_range_ratio"])
        ax_bot.plot(t1["time_slot"], t1["proximity_rate"], color="#b45309", linewidth=1.5, label="Proximity rate (robust+lag)")
        ax_bot_b = ax_bot.twinx()
        ax_bot_b.plot(
            t1["time_slot"],
            t1["distance_in_range_ratio"],
            color="#1d4ed8",
            linewidth=1.5,
            label=f"GPS close ratio <= {DISTANCE_THRESHOLD_M:.0f}m",
        )
        ax_bot.set_ylim(0, 1.0)
        ax_bot_b.set_ylim(0, 1.0)
        ax_bot.set_ylabel("Proximity rate")
        ax_bot_b.set_ylabel("GPS close ratio")
        ax_bot.set_xlabel("Time")
        ax_bot.grid(alpha=0.25)
        bot_lines = ax_bot.get_lines() + ax_bot_b.get_lines()
        ax_bot.legend(bot_lines, [l.get_label() for l in bot_lines], loc="upper right")

        fig.suptitle(
            f"{pair_id} ({ctx.get('design','')}) | "
            f"Top: raw (lag=0), Bottom: robust filter + lag={robust_lag:+d}min",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        out_path = FIG_DIR / f"{pair_id}_raw_vs_robust.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        pair_fig_paths[pair_id] = out_path

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# Selected Pairs Final Report: Proximity Logger vs GPS")
    lines.append("")
    lines.append(f"- Generated at: `{now}`")
    lines.append(f"- Selected pairs: `{', '.join(SELECTED_PAIRS)}`")
    lines.append(f"- Fixed configuration: `window={AGGREGATE_WINDOW_MIN}min`, `distance<={DISTANCE_THRESHOLD_M}m`")
    lines.append(f"- Lag search range: `±{LAG_MAX_MIN}min`, step `{LAG_STEP_MIN}min`")
    lines.append("")

    lines.append("## 1. Why these pairs")
    lines.append("")
    lines.append(
        "This report focuses on `N172_N171`, `N172_N182`, and `N172_N189` as the manually validated "
        "highest-quality proximity logger pairs."
    )
    lines.append(
        "`N172_N171` uses `double` collar design; `N172_N182` and `N172_N189` use `single` design and are "
        "expected to suffer more GPS missingness."
    )
    lines.append("")

    lines.append("## 2. GPS Availability Risk Check")
    lines.append("")
    lines.append(
        f"We track per-bucket GPS validity ratio and define robust buckets as ratio >= `{MIN_DISTANCE_VALID_RATIO_PER_BUCKET:.0%}`."
    )
    lines.append("For single-design pairs, we also inspect the known risk window `12:00-16:00`.")
    lines.append("")
    lines.append(md_table(availability_df, max_rows=10))
    lines.append("")
    lines.append("![GPS availability by pair](figures_selected/availability_by_pair.png)")
    lines.append("")

    lines.append("## 3. Alignment Results (Full Data)")
    lines.append("")
    lines.append(md_table(full_df, max_rows=20))
    lines.append("")

    lines.append("## 4. Alignment Results (Robust: quality buckets + single no-midday)")
    lines.append("")
    lines.append(md_table(robust_df, max_rows=20))
    lines.append("")
    lines.append("![Correlation comparison](figures_selected/correlation_full_vs_robust.png)")
    lines.append("")
    lines.append("### Pair Trend Figures")
    lines.append("")
    for pair_id in SELECTED_PAIRS:
        if pair_id in pair_fig_paths:
            lines.append(f"#### {pair_id}")
            lines.append("")
            lines.append(f"![{pair_id} trend](figures_selected/{pair_fig_paths[pair_id].name})")
            lines.append("")
    lines.append("")

    lines.append("## 5. Interpretation for GPS Replacement")
    lines.append("")
    # build concise judgments
    judgments = []
    for pair in SELECTED_PAIRS:
        f = full_df[full_df["pair_id"] == pair]
        r = robust_df[robust_df["pair_id"] == pair]
        if f.empty or r.empty:
            continue
        f_best = f["best_corr"].iloc[0]
        r_best = r["best_corr"].iloc[0]
        f_lag = f["best_lag_min"].iloc[0]
        r_lag = r["best_lag_min"].iloc[0]
        judgments.append((pair, f_best, r_best, f_lag, r_lag))

    for pair, f_best, r_best, f_lag, r_lag in judgments:
        lines.append(
            f"- `{pair}`: full best corr `{f_best:.3f}` at lag `{f_lag:+.0f}min`; "
            f"robust best corr `{r_best:.3f}` at lag `{r_lag:+.0f}min`."
        )

    robust_mean = robust_df["best_corr"].mean()
    full_mean = full_df["best_corr"].mean()
    robust_large_lag_ratio = (
        float((robust_df["best_lag_min"].abs() > 360).mean()) if not robust_df.empty else np.nan
    )
    lines.append("")
    lines.append(
        f"Average best correlation across selected pairs: full `{full_mean:.3f}`, robust `{robust_mean:.3f}`."
    )
    if not np.isnan(robust_large_lag_ratio):
        lines.append(
            f"Large-lag dependency in robust set (|lag| > 360min): `{robust_large_lag_ratio:.1%}` of pairs."
        )
    if robust_mean >= 0.40 and (np.isnan(robust_large_lag_ratio) or robust_large_lag_ratio <= 0.34):
        lines.append(
            "Under low GPS-loss conditions, GPS can be used as a practical proxy for proximity logger at the "
            "current coarse behavioral scale, without relying on extreme time shifts."
        )
    elif robust_mean >= 0.40 and robust_large_lag_ratio > 0.34:
        lines.append(
            "Correlation can become high only after large time shifts for many pairs; this suggests potential "
            "clock-drift and/or daily-cycle aliasing. GPS can support trend analysis, but direct replacement "
            "needs calibrated time synchronization."
        )
    else:
        lines.append(
            "Even after robust filtering, consistency is moderate; GPS should be used as a complementary signal "
            "rather than a full replacement."
        )
    lines.append("")

    if robust_mean >= 0.40 and (np.isnan(robust_large_lag_ratio) or robust_large_lag_ratio <= 0.34):
        final_recommendation_line = (
            "- In low-loss windows, GPS is viable as a replacement for trend-level proximity assessment; "
            "for fine-grained event auditing, retain proximity logger as reference."
        )
    elif robust_mean >= 0.40 and robust_large_lag_ratio > 0.34:
        final_recommendation_line = (
            "- With current severe lag dependency, GPS should be treated as a complementary signal unless time "
            "synchronization is calibrated (or a defensible lag-correction protocol is fixed in advance)."
        )
    else:
        final_recommendation_line = (
            "- GPS currently should remain complementary; keep proximity logger as the primary reference for "
            "contact validation."
        )

    lines.append("## 6. Potential Issues Identified")
    lines.append("")
    lines.append(
        "- Best lag may still be large for some pairs, which can be driven by behavioral periodicity, not only clock drift."
    )
    lines.append(
        "- Single collar design shows stronger midday (`12:00-16:00`) data-quality drop; this can create false disagreement "
        "(`high proximity`, `low GPS close ratio`)."
    )
    lines.append(
        "- Event imbalance exists: many windows have low/zero proximity; correlation can be unstable for rare-contact pairs."
    )
    lines.append(
        "- Interpolation (<=15min) improves continuity but may smooth peaks and affect strict event-level alignment."
    )
    lines.append("")

    lines.append("## 7. Final Recommendation (for this selected high-quality subset)")
    lines.append("")
    lines.append(
        f"- Keep analysis at coarse aggregation (`{AGGREGATE_WINDOW_MIN}min`) with near-distance threshold `{DISTANCE_THRESHOLD_M}m`."
    )
    lines.append(
        f"- Apply bucket quality control (`distance_valid_ratio >= {MIN_DISTANCE_VALID_RATIO_PER_BUCKET:.0%}`); for `single` design, "
        "treat `12:00-16:00` as high-risk."
    )
    lines.append(final_recommendation_line)
    lines.append("")

    lines.append("## 8. Output Index")
    lines.append("")
    lines.append(f"- Final report summary CSV: `{SUMMARY_CSV}`")
    lines.append(f"- Pair-level evaluation table: `{ANALYSIS_DIR / 'task11_selected_pair_shift_eval.csv'}`")
    lines.append(f"- Availability diagnostics: `{ANALYSIS_DIR / 'task11_selected_pair_availability.csv'}`")
    lines.append("")

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report: {REPORT_FILE}")
    print(f"Saved summary csv: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
