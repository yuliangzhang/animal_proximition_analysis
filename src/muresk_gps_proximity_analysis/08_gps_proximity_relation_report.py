from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("data/muresk_farm_proximity_data/gps_proximition_data")
REPORT_DIR = BASE_DIR / "reports"

PAIR_SUMMARY_FILE = BASE_DIR / "pair_distance_summary.csv"
GRID_SUMMARY_FILE = BASE_DIR / "gps_proximity_grid_lag_summary.csv"
GRID_SCAN_FILE = BASE_DIR / "gps_proximity_grid_lag_scan.csv"
BEST_BY_PAIR_FILE = BASE_DIR / "analysis" / "gps_proximity_grid_best_by_pair.csv"

REPORT_FILE = REPORT_DIR / "gps_proximity_relation_report.md"
REPORT_SUMMARY_CSV = REPORT_DIR / "gps_proximity_relation_report_summary.csv"


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No data._"

    work = df.copy().head(max_rows)
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    header = "| " + " | ".join(work.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    lines = [header, sep]
    for row in work.itertuples(index=False):
        vals = ["" if pd.isna(v) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Only first {max_rows} rows shown (total {len(df)})._")
    return "\n".join(lines)


def evaluate_feasibility(best_by_pair: pd.DataFrame, out_of_range_ratio: float) -> str:
    if best_by_pair.empty:
        return "数据不足，无法评估。"

    max_corr = best_by_pair["best_pearson_corr"].max()
    mean_corr = best_by_pair["best_pearson_corr"].mean()

    if out_of_range_ratio >= 0.5:
        return (
            "最优结果大多依赖超大 lag（>±6h），更可能是日周期而非时钟漂移，"
            "当前不建议直接下结论为可替代。"
        )

    if max_corr >= 0.70 and mean_corr >= 0.50:
        return "在当前数据下，GPS 对 proximity logger 具有较强替代潜力。"
    if max_corr >= 0.50 and mean_corr >= 0.30:
        return "在部分配对上有替代潜力，但整体稳定性不足，建议分场景使用。"
    if max_corr >= 0.30:
        return "存在中等一致性，但不足以直接替代，建议作为辅助信号。"
    return "整体一致性偏弱，目前不建议用 GPS 直接替代 proximity logger。"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not GRID_SUMMARY_FILE.exists():
        raise FileNotFoundError(f"Grid summary file not found: {GRID_SUMMARY_FILE}")
    if not GRID_SCAN_FILE.exists():
        raise FileNotFoundError(f"Grid scan file not found: {GRID_SCAN_FILE}")
    if not PAIR_SUMMARY_FILE.exists():
        raise FileNotFoundError(f"Pair summary file not found: {PAIR_SUMMARY_FILE}")

    pair_summary = pd.read_csv(PAIR_SUMMARY_FILE)
    grid_summary = pd.read_csv(GRID_SUMMARY_FILE)
    grid_scan = pd.read_csv(GRID_SCAN_FILE)

    if BEST_BY_PAIR_FILE.exists():
        best_by_pair = pd.read_csv(BEST_BY_PAIR_FILE)
    else:
        best_by_pair = (
            grid_summary.dropna(subset=["best_pearson_corr"])
            .sort_values(["pair_id", "best_pearson_corr"], ascending=[True, False], kind="stable")
            .groupby("pair_id", as_index=False)
            .head(1)
        )

    # Plausible lag window (within +-6 hours)
    plausible_scan = grid_scan[
        grid_scan["lag_minutes"].abs() <= 360
    ].dropna(subset=["pearson_corr"])
    plausible_best = (
        plausible_scan.sort_values(["pair_id", "pearson_corr"], ascending=[True, False], kind="stable")
        .groupby("pair_id", as_index=False)
        .head(1)
        .rename(
            columns={
                "lag_minutes": "plausible_lag_minutes",
                "pearson_corr": "plausible_pearson_corr",
                "spearman_corr": "plausible_spearman_corr",
                "overlap_points": "plausible_overlap_points",
                "aggregate_window_min": "plausible_window_min",
                "distance_threshold_m": "plausible_threshold_m",
            }
        )
    )

    # Global best configs
    global_best = (
        grid_summary.dropna(subset=["best_pearson_corr"])
        .sort_values("best_pearson_corr", ascending=False, kind="stable")
        .head(15)
        .reset_index(drop=True)
    )

    # Improvement stats
    improve_df = grid_summary.copy()
    improve_df["corr_improvement"] = improve_df["corr_improvement"].astype("float64")
    improved_ratio = float((improve_df["corr_improvement"] > 0).mean()) if len(improve_df) else np.nan
    median_improvement = float(improve_df["corr_improvement"].median()) if len(improve_df) else np.nan

    # Lag stability by pair
    lag_stats = (
        grid_summary.dropna(subset=["best_lag_minutes"])
        .groupby("pair_id")["best_lag_minutes"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "grid_count",
                "mean": "lag_mean_min",
                "median": "lag_median_min",
                "std": "lag_std_min",
                "min": "lag_min_min",
                "max": "lag_max_min",
            }
        )
    )

    out_of_range_ratio = float((best_by_pair["best_lag_minutes"].abs() > 360).mean()) if len(best_by_pair) else np.nan
    feasibility_text = evaluate_feasibility(best_by_pair, out_of_range_ratio=out_of_range_ratio)

    # Save concise csv summary for downstream use
    summary_rows = []
    best_by_pair_merged = best_by_pair.merge(
        plausible_best[
            [
                "pair_id",
                "plausible_window_min",
                "plausible_threshold_m",
                "plausible_lag_minutes",
                "plausible_pearson_corr",
                "plausible_spearman_corr",
                "plausible_overlap_points",
            ]
        ],
        on="pair_id",
        how="left",
    )

    for row in best_by_pair_merged.itertuples(index=False):
        summary_rows.append(
            {
                "pair_id": row.pair_id,
                "best_window_min": row.aggregate_window_min,
                "best_distance_threshold_m": row.distance_threshold_m,
                "best_lag_minutes": row.best_lag_minutes,
                "best_pearson_corr": row.best_pearson_corr,
                "lag0_pearson_corr": row.lag0_pearson_corr,
                "corr_improvement": row.corr_improvement,
                "plausible_window_min": row.plausible_window_min,
                "plausible_threshold_m": row.plausible_threshold_m,
                "plausible_lag_minutes": row.plausible_lag_minutes,
                "plausible_pearson_corr": row.plausible_pearson_corr,
            }
        )
    pd.DataFrame(summary_rows).to_csv(REPORT_SUMMARY_CSV, index=False, encoding="utf-8")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# Proximity Logger 与 GPS 距离相关性探索报告")
    lines.append("")
    lines.append(f"- 生成时间：`{now}`")
    lines.append(f"- 输入结果：`{GRID_SUMMARY_FILE}`")
    lines.append(f"- 配对概要：`{PAIR_SUMMARY_FILE}`")
    lines.append("")

    lines.append("## 1. 数据与方法概览")
    lines.append("")
    lines.append("- Grid 维度：聚合窗口 `30~240min`（步长30） × 距离阈值 `10~50m`（步长5）")
    lines.append("- 对齐方法：固定 GPS close ratio，按 `5min` 步长平移 proximity，搜索最佳 `lag`")
    lines.append("- 目标：使 proximity 接近率与 GPS 近距离占比的一致性（Pearson）最大")
    lines.append("")

    lines.append("## 2. 输入配对数据质量")
    lines.append("")
    quality_cols = [
        "receiver_gps_id",
        "beacon_gps_id",
        "distance_non_missing_slots",
        "proximity_non_missing_slots",
        "proximity_positive_slots",
        "median_distance_m",
    ]
    existing_quality_cols = [c for c in quality_cols if c in pair_summary.columns]
    lines.append(md_table(pair_summary[existing_quality_cols], max_rows=20))
    lines.append("")

    lines.append("## 3. 每对羊最优配置（按 best_pearson_corr）")
    lines.append("")
    best_cols = [
        "pair_id",
        "aggregate_window_min",
        "distance_threshold_m",
        "best_lag_minutes",
        "best_pearson_corr",
        "lag0_pearson_corr",
        "corr_improvement",
    ]
    lines.append(md_table(best_by_pair[best_cols], max_rows=20))
    lines.append("")
    lines.append("### 3.1 可解释 lag（±6h）约束下的最优配置")
    lines.append("")
    plausible_cols = [
        "pair_id",
        "plausible_window_min",
        "plausible_threshold_m",
        "plausible_lag_minutes",
        "plausible_pearson_corr",
        "plausible_spearman_corr",
        "plausible_overlap_points",
    ]
    lines.append(md_table(plausible_best[plausible_cols], max_rows=20))
    lines.append("")

    lines.append("## 4. 全局 Top 配置")
    lines.append("")
    top_cols = [
        "pair_id",
        "aggregate_window_min",
        "distance_threshold_m",
        "best_lag_minutes",
        "best_pearson_corr",
        "corr_improvement",
    ]
    lines.append(md_table(global_best[top_cols], max_rows=15))
    lines.append("")

    lines.append("## 5. Lag 稳定性")
    lines.append("")
    lines.append(md_table(lag_stats, max_rows=20))
    lines.append("")

    lines.append("## 6. 主动分析结论")
    lines.append("")
    lines.append(f"- `lag` 调整后相关性提升占比：`{improved_ratio:.2%}`")
    lines.append(f"- 相关性提升中位数：`{median_improvement:.4f}`")
    lines.append(f"- 每对最优 lag 超出 ±6h 的占比：`{out_of_range_ratio:.2%}`")
    lines.append(f"- 可行性判断：{feasibility_text}")
    lines.append("- 注意：若最优 lag 频繁接近 `±24h`，可能受到昼夜行为周期影响，而非纯时钟漂移。")
    lines.append("- 建议：优先关注在多个阈值/窗口下 lag 稳定且相关性稳定提升的配对与参数组合。")
    lines.append("")

    lines.append("## 7. 产物索引")
    lines.append("")
    lines.append(f"- Grid 汇总：`{GRID_SUMMARY_FILE}`")
    lines.append(f"- Best by pair：`{BEST_BY_PAIR_FILE}`")
    lines.append(f"- 图像根目录：`{BASE_DIR / 'figures'}`")
    lines.append(f"- 报告摘要 CSV：`{REPORT_SUMMARY_CSV}`")
    lines.append("")

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report: {REPORT_FILE}")
    print(f"Saved report summary csv: {REPORT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
