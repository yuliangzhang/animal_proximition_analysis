import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Set, Tuple

import math
import pandas as pd


GPS_DATA_DIR = "data/gps_data"
PROXIMITION_DIR = "data/proximition_split"
OUTPUT_DIR = "data/proximition_distance_split"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in meters between two WGS84 points."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def parse_proximition_filename(fname: str) -> Optional[Tuple[str, str]]:
    """
    Extract GPS codes from filename like:
    P_GPS1043_GPS0006_951-..._940-....csv -> (GPS1043, GPS0006)
    """
    base = os.path.basename(fname)
    if base.lower().endswith(".csv"):
        base = base[:-4]
    parts = base.split("_")
    if len(parts) < 3:
        return None
    # parts[0] is initial, [1] is ram_gps, [2] is ewe_gps
    return parts[1], parts[2]


def collect_needed_codes(files: Iterable[str]) -> Set[str]:
    codes: Set[str] = set()
    for f in files:
        parsed = parse_proximition_filename(f)
        if parsed:
            codes.update(parsed)
    return codes


def round_to_5min_str(dt_series: pd.Series) -> pd.Series:
    return pd.to_datetime(dt_series, errors="coerce").dt.tz_localize(None).dt.round("5min").dt.strftime("%Y-%m-%d %H:%M:%S")


def build_gps_lookup(codes: Set[str]) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Build lookup: (gps_code, ts_5min_str) -> (lat, lon)
    - Loads only rows where gps_code in codes from all CSVs under GPS_DATA_DIR
    - Shifts timestamps by +8 hours (local time)
    - Rounds timestamps to 5 minutes to align with proximition timestamps
    - If multiple rows fall into the same 5-min bin, keep the last by original timestamp
    """
    if not codes:
        return {}

    lookup: Dict[Tuple[str, str], Tuple[float, float]] = {}
    csv_files = [os.path.join(GPS_DATA_DIR, f) for f in os.listdir(GPS_DATA_DIR) if f.lower().endswith(".csv")]
    usecols = ["gps_code", "timestamp", "lat", "lon"]

    for path in csv_files:
        # Read in chunks to limit memory usage
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=200000):
            # Filter by codes
            sub = chunk[chunk["gps_code"].isin(codes)].copy()
            if sub.empty:
                continue
            # Time to local (+8h) then bin to 5-min string
            sub["ts_local"] = pd.to_datetime(sub["timestamp"], errors="coerce") + pd.Timedelta(hours=8)
            sub.dropna(subset=["ts_local"], inplace=True)
            sub.sort_values(["gps_code", "ts_local"], inplace=True)
            sub["ts_5min"] = sub["ts_local"].dt.round("5min").dt.strftime("%Y-%m-%d %H:%M:%S")
            # Keep last per (gps_code, ts_5min)
            sub = sub.drop_duplicates(subset=["gps_code", "ts_5min"], keep="last")
            for _, r in sub.iterrows():
                key = (r["gps_code"], r["ts_5min"])
                lookup[key] = (float(r["lat"]), float(r["lon"]))

    return lookup


def process_one_file(
    fpath: str,
    lookup: Dict[Tuple[str, str], Tuple[float, float]],
    out_dir: str,
) -> Tuple[str, int, int]:
    """
    Compute pairwise distance for one proximition CSV.
    Returns (filename, total_rows, matched_rows)
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(fpath)

    pair = parse_proximition_filename(base)
    if not pair:
        return base, 0, 0
    ram_code, ewe_code = pair

    df = pd.read_csv(fpath)
    # Expect timestamp/value
    if df.empty or "timestamp" not in df.columns:
        df["distance"] = []
        df.to_csv(os.path.join(out_dir, base), index=False)
        return base, 0, 0

    # Map timestamps to 5-min grid (they should already align), but ensure consistent formatting
    df["ts5"] = round_to_5min_str(df["timestamp"])

    lats1: List[Optional[float]] = []
    lons1: List[Optional[float]] = []
    lats2: List[Optional[float]] = []
    lons2: List[Optional[float]] = []

    for ts in df["ts5"].tolist():
        p1 = lookup.get((ram_code, ts))
        p2 = lookup.get((ewe_code, ts))
        if p1 is None:
            lats1.append(None); lons1.append(None)
        else:
            lats1.append(p1[0]); lons1.append(p1[1])
        if p2 is None:
            lats2.append(None); lons2.append(None)
        else:
            lats2.append(p2[0]); lons2.append(p2[1])

    dist: List[object] = []
    matched = 0
    for a_lat, a_lon, b_lat, b_lon in zip(lats1, lons1, lats2, lons2):
        if a_lat is None or a_lon is None or b_lat is None or b_lon is None:
            dist.append("INF")
        else:
            matched += 1
            dist.append(round(haversine_m(a_lat, a_lon, b_lat, b_lon), 3))

    df["distance"] = dist
    df = df[["timestamp", "value", "distance"]]
    out_path = os.path.join(out_dir, base)
    df.to_csv(out_path, index=False)
    return base, len(df), matched


def main(limit_files: Optional[int] = 10, max_workers: int = 8) -> None:
    # List proximition files
    all_files = [os.path.join(PROXIMITION_DIR, f) for f in os.listdir(PROXIMITION_DIR) if f.lower().endswith(".csv")]
    all_files.sort()
    if limit_files is not None:
        files = all_files[:limit_files]
    else:
        files = all_files

    # Collect required GPS codes
    codes = collect_needed_codes(files)
    print(f"Files to process: {len(files)}; Unique GPS codes needed: {len(codes)}")

    # Build lookup map
    lookup = build_gps_lookup(codes)
    print(f"Lookup entries built: {len(lookup)}")

    # Process files in parallel
    results: List[Tuple[str, int, int]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(process_one_file, f, lookup, OUTPUT_DIR) for f in files]
        for fut in as_completed(futs):
            results.append(fut.result())

    # Report
    total_rows = sum(r[1] for r in results)
    matched_rows = sum(r[2] for r in results)
    print(f"Done. Files: {len(results)}, Total rows: {total_rows}, Rows with GPS for both: {matched_rows}")
    if results:
        print("Sample results:", sorted(r[0] for r in results)[:3])


if __name__ == "__main__":
    # Default: run 10 files for test
    main(limit_files=10, max_workers=8)

