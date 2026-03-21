import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, Optional, List

import pandas as pd
from datetime import timedelta


# Paths
RAW_EXCEL_PATH = "data/raw_data/HS_proximity_2024_5min_UWA.xlsx"
# For VID lookup by EID
VID_MASTERFILE_CSV = "data/masterfile/masterfile_2024.csv"
# For GPS lookup by collar IDs
GPS_MASTER_XLSX = "data/raw_data/2024-GPS-Summer-Recovery-MASTER.xlsx"
OUTPUT_DIR = "data/proximition_split"


def _sanitize_eid(eid: str) -> str:
    """
    Normalize EID string by replacing spaces with '-' and stripping.
    """
    if pd.isna(eid):
        return ""
    return str(eid).strip().replace(" ", "-")


def _first_non_null(*values) -> Optional[str]:
    for v in values:
        if pd.notna(v) and str(v).strip() not in ("", "nan", "None"):
            return str(v).strip()
    return None


def _norm_collar(val: str) -> str:
    """Normalize collar IDs for matching: remove spaces and lowercase."""
    if pd.isna(val):
        return ""
    return str(val).strip().replace(" ", "").lower()


def _norm_gps_code(val: str) -> str:
    if pd.isna(val):
        return ""
    # Remove spaces only per requirement; keep letter case
    return str(val).strip().replace(" ", "")


def load_collar_to_gps_mapping(gps_master_xlsx: str) -> Dict[str, str]:
    """Build a mapping: normalized collar ID -> GPS code (spaces removed)."""
    xl = pd.ExcelFile(gps_master_xlsx)
    if "Sheet1" in xl.sheet_names:
        df = pd.read_excel(gps_master_xlsx, sheet_name="Sheet1")
    else:
        df = pd.read_excel(gps_master_xlsx)

    mapping: Dict[str, str] = {}
    candidates = [("Collar EID", "GPS #"), ("Collar EID.1", "GPS #.1")]
    for kcol, vcol in candidates:
        if kcol in df.columns and vcol in df.columns:
            sub = df[[kcol, vcol]].dropna(how="all")
            for _, row in sub.iterrows():
                key = _norm_collar(row.get(kcol))
                gps = _norm_gps_code(row.get(vcol))
                if key and gps:
                    mapping[key] = gps
    return mapping


def load_eid_to_vid(masterfile_csv: str) -> Dict[str, str]:
    """Build a mapping: normalized EID (with '-') -> VID (no spaces, uppercase)."""
    mf = pd.read_csv(masterfile_csv)
    if "eid" not in mf.columns or "vid" not in mf.columns:
        raise ValueError("masterfile must contain 'eid' and 'vid'")
    mf["eid"] = mf["eid"].astype(str).apply(_sanitize_eid)
    mf["vid"] = mf["vid"].astype(str).str.replace(" ", "", regex=False).str.upper()
    mapping: Dict[str, str] = {}
    for _, row in mf.iterrows():
        eid = row.get("eid")
        vid = row.get("vid")
        if eid:
            mapping[eid] = vid or ""
    return mapping


def _extract_time_columns(df: pd.DataFrame) -> List[str]:
    """
    Determine which columns are timestamps. Exclude the meta columns.
    """
    exclude = {"Ram_EID", "Ram_coller", "Ewe_EID", "Ewe_coller", "Plot"}
    time_cols = [c for c in df.columns if c not in exclude]
    return time_cols


def _row_to_long_df(row: pd.Series, time_cols: List[str]) -> pd.DataFrame:
    """
    Convert a wide time-series row into a long two-column DataFrame: [timestamp, value].
    Non-numeric or null values are set to 0.
    """
    # Build dict of timestamp->value for selected columns
    data = []
    for col in time_cols:
        ts = col
        # If column header is a datetime, format it; else use the string as-is
        if hasattr(col, "strftime"):
            ts_str = col.strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = str(col)

        val = row.get(col)
        # Convert to numeric, coerce errors to NaN then fill with 0
        num = pd.to_numeric(val, errors="coerce")
        if pd.isna(num):
            num = 0
        data.append((ts_str, num))

    out = pd.DataFrame(data, columns=["timestamp", "value"])
    return out


def _filter_windows_around_positive(df: pd.DataFrame, window_minutes: int = 30) -> pd.DataFrame:
    """
    Keep only rows where `value` > 0 and the +/- window_minutes around those times,
    but only retain each window if the sum of values within that window > 3.
    If multiple positive points' windows overlap, merge these windows.
    Finally, return rows within merged windows sorted by time.
    """
    if df.empty:
        return df

    # Parse timestamps
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.copy()
    df["ts"] = ts
    df = df.dropna(subset=["ts"])  # drop rows with invalid timestamps

    # Identify positive points
    positives = df.loc[df["value"] > 0, "ts"]
    if positives.empty:
        # No positive values; return empty result preserving columns
        return df.iloc[0:0][["timestamp", "value"]]

    # Build intervals around each positive time
    half = timedelta(minutes=window_minutes)
    intervals = sorted([(t - half, t + half) for t in positives])
    
    # Keep only intervals whose in-window sum(value) > 3
    kept: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in intervals:
        win_sum = df.loc[(df["ts"] >= start) & (df["ts"] <= end), "value"].sum()
        if win_sum > 3:
            kept.append((start, end))
    if not kept:
        return df.iloc[0:0][["timestamp", "value"]]

    # Merge overlapping kept intervals
    kept.sort(key=lambda x: x[0])
    merged: List[List[pd.Timestamp]] = []
    for start, end in kept:
        if not merged:
            merged.append([start, end])
        else:
            last_start, last_end = merged[-1]
            if start <= last_end:
                if end > last_end:
                    merged[-1][1] = end
            else:
                merged.append([start, end])

    # Filter rows inside any merged interval
    mask = pd.Series(False, index=df.index)
    for start, end in merged:
        mask |= (df["ts"] >= start) & (df["ts"] <= end)

    out = df.loc[mask, ["ts", "value"]].sort_values("ts")
    # Format timestamps as strings again
    out["timestamp"] = out["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out = out[["timestamp", "value"]]
    return out


def _build_filename(vid_letter: str, ram_gps: str, ewe_gps: str, ram_eid: str, ewe_eid: str) -> str:
    # underscore separated as requested: initial + ram_gps + ewe_gps + ram_eid + ewe_eid
    base = f"{vid_letter}_{ram_gps}_{ewe_gps}_{ram_eid}_{ewe_eid}"
    return f"{base}.csv"


def process_single_row(
    row: pd.Series,
    time_cols: List[str],
    collar_to_gps: Dict[str, str],
    eid_to_vid: Dict[str, str],
    out_dir: str,
    row_idx: int,
) -> Tuple[Optional[str], str]:
    """
    Process one row of the proximity sheet and write its two-column CSV if mapping is available.
    Returns the output path if written, else None.
    """
    ram_eid = _sanitize_eid(row.get("Ram_EID"))
    ewe_eid = _sanitize_eid(row.get("Ewe_EID"))

    # Find GPS codes via collar IDs (match after removing spaces and lowercasing)
    ram_collar = _norm_collar(row.get("Ram_coller"))
    ewe_collar = _norm_collar(row.get("Ewe_coller"))
    ram_gps = collar_to_gps.get(ram_collar)
    ewe_gps = collar_to_gps.get(ewe_collar)
    if not ram_gps or not ewe_gps:
        return None, "missing_gps_by_collar"

    # Lookup vids for initial
    ram_vid = eid_to_vid.get(ram_eid, "")
    ewe_vid = eid_to_vid.get(ewe_eid, "")
    # Filename initial MUST be Ewe's VID initial only
    chosen_vid = (ewe_vid or "").strip()
    vid_letter = chosen_vid[:1].upper() if chosen_vid else "X"

    long_df = _row_to_long_df(row, time_cols)
    # Apply filtering: keep only value>0 and +/- 30 minutes, merge overlaps
    long_df = _filter_windows_around_positive(long_df, window_minutes=30)

    # Ensure output dir exists
    os.makedirs(out_dir, exist_ok=True)

    # Construct filename per spec
    filename = _build_filename(vid_letter, str(ram_gps), str(ewe_gps), str(ram_eid), str(ewe_eid))
    out_path = os.path.join(out_dir, filename)
    long_df.to_csv(out_path, index=False)
    return out_path, "ok"


def process_proximity(
    excel_path: str = RAW_EXCEL_PATH,
    vid_masterfile_csv: str = VID_MASTERFILE_CSV,
    gps_master_xlsx: str = GPS_MASTER_XLSX,
    out_dir: str = OUTPUT_DIR,
    limit: Optional[int] = 10,
    max_workers: int = 8,
) -> None:
    """
    Main processing entry:
    - Load masterfile mapping
    - Load proximity Excel
    - Process up to `limit` rows (None for all) in parallel
    - Write per-row two-column CSVs to `out_dir`
    """
    print(f"Loading VID mapping from: {vid_masterfile_csv}")
    eid_to_vid = load_eid_to_vid(vid_masterfile_csv)
    print(f"VIDs loaded for EIDs: {len(eid_to_vid)}")

    print(f"Loading Collar->GPS mapping from: {gps_master_xlsx}")
    collar_to_gps = load_collar_to_gps_mapping(gps_master_xlsx)
    print(f"Collar IDs mapped to GPS: {len(collar_to_gps)}")

    print(f"Loading proximity Excel: {excel_path}")
    df = pd.read_excel(excel_path)
    time_cols = _extract_time_columns(df)
    print(f"Detected {len(time_cols)} timestamp columns")

    # Apply limit
    if limit is not None:
        df = df.iloc[:limit].copy()
        print(f"Processing first {len(df)} rows for test run...")
    else:
        print(f"Processing all {len(df)} rows...")

    written = 0
    skipped_missing = 0
    skipped_mismatch = 0
    paths: List[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_idx = {
            ex.submit(process_single_row, df.iloc[i], time_cols, collar_to_gps, eid_to_vid, out_dir, i): i
            for i in range(len(df))
        }
        for fut in as_completed(fut_to_idx):
            out_path, reason = fut.result()
            if out_path:
                written += 1
                paths.append(out_path)
            else:
                if reason == "treatment_mismatch":
                    skipped_mismatch += 1
                else:
                    skipped_missing += 1

    print(
        f"Done. Written files: {written}, "
        f"Skipped missing mapping: {skipped_missing}, "
        f"Skipped treatment mismatch: {skipped_mismatch}"
    )
    if written:
        print(f"Sample output: {paths[:3]}")


if __name__ == "__main__":
    # Default run: process 10 rows for verification
    process_proximity(limit=10)
