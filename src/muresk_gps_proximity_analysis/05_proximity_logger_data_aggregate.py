from __future__ import annotations

from pathlib import Path

import pandas as pd


PROXIMITY_DIR = Path("data/muresk_farm_proximity_data/muresk_proximity_data")
INPUT_GLOB = "*_60secProxSummary.csv"
META_FILE = PROXIMITY_DIR / "Uwa_logger_sensors_2025.xlsx"
META_SHEET = "Link"
OUTPUT_FILE = PROXIMITY_DIR / "muresk_proximity_5min.csv"
GPS_RAW_FILE = Path("data/muresk_farm_proximity_data/muresk_gps_data/muresk_gps_raw_data.csv")
START_DATE = "2025-12-10"
END_DATE = "2026-01-03"


def normalize_text(value: object) -> str:
    """Normalize id-like values for robust matching."""
    if pd.isna(value):
        return ""
    return "".join(str(value).strip().upper().split())


def normalize_actigraph(value: object) -> str:
    """Normalize Actigraph IDs from metadata/file name."""
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return normalize_text(text)


def load_gps_valid_counts(gps_file: Path, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> dict[str, int]:
    """Count valid GPS points by code in the experiment period."""
    gps_df = pd.read_csv(gps_file, usecols=["gps_code", "timestamp", "latitude", "longitude"])
    gps_df["gps_norm"] = gps_df["gps_code"].apply(normalize_text)
    gps_df["timestamp"] = pd.to_datetime(gps_df["timestamp"], errors="coerce")
    gps_df["latitude"] = pd.to_numeric(gps_df["latitude"], errors="coerce")
    gps_df["longitude"] = pd.to_numeric(gps_df["longitude"], errors="coerce")
    gps_df = gps_df[(gps_df["timestamp"] >= start_ts) & (gps_df["timestamp"] <= end_ts)].copy()

    valid_mask = (
        gps_df["latitude"].notna()
        & gps_df["longitude"].notna()
        & (gps_df["latitude"] != 0)
        & (gps_df["longitude"] != 0)
        & gps_df["latitude"].between(-90, 90)
        & gps_df["longitude"].between(-180, 180)
    )
    counts = gps_df.loc[valid_mask].groupby("gps_norm").size().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def resolve_serial_map(
    serial_candidates: dict[str, list[dict]],
    gps_valid_counts: dict[str, int],
) -> dict[str, dict]:
    """Resolve duplicate serial->GPS mappings using GPS availability in experiment period."""
    serial_map: dict[str, dict] = {}
    for serial_key, items in serial_candidates.items():
        if len(items) == 1:
            serial_map[serial_key] = items[0]
            continue

        def score(item: dict) -> tuple[int, int]:
            gps_key = normalize_text(item.get("gps_id", ""))
            valid_num = gps_valid_counts.get(gps_key, 0)
            has_gps = 1 if gps_key else 0
            return valid_num, has_gps

        best = max(items, key=score)
        serial_map[serial_key] = best

        candidate_desc = ", ".join(
            f"{it.get('gps_id','?')}[valid={gps_valid_counts.get(normalize_text(it.get('gps_id','')), 0)}]"
            for it in items
        )
        print(
            f"Resolved duplicated serial {serial_key} -> {best.get('gps_id','')} "
            f"using GPS availability. Candidates: {candidate_desc}"
        )
    return serial_map


def load_link_metadata(
    meta_file: Path,
    sheet_name: str,
    gps_valid_counts: dict[str, int],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Build lookup maps for receiver (actigraph) and beacon (serial)."""
    link_df = pd.read_excel(meta_file, sheet_name=sheet_name, engine="openpyxl")
    required_cols = {"GPS", "Actigraph", "Serial", "Actigraph mode", "Coller design"}
    missing = required_cols - set(link_df.columns)
    if missing:
        raise ValueError(f"Missing columns in metadata sheet '{sheet_name}': {sorted(missing)}")

    df = link_df[list(required_cols)].copy()
    df["gps_norm"] = df["GPS"].astype("string").str.strip().str.lower().fillna("")
    df["actigraph_norm"] = df["Actigraph"].apply(normalize_actigraph)
    df["serial_norm"] = df["Serial"].apply(normalize_text)
    df["mode_norm"] = df["Actigraph mode"].astype("string").str.strip().str.lower().fillna("")
    df["coller_design_norm"] = df["Coller design"].astype("string").str.strip().fillna("")

    actigraph_map: dict[str, dict] = {}
    serial_candidates: dict[str, list[dict]] = {}

    for _, row in df.iterrows():
        item = {
            "gps_id": row["gps_norm"],
            "actigraph_id": row["actigraph_norm"],
            "serial_id": row["serial_norm"],
            "mode": row["mode_norm"],
            "collar_design": row["coller_design_norm"],
        }

        actigraph_key = row["actigraph_norm"]
        serial_key = row["serial_norm"]
        if actigraph_key and actigraph_key not in actigraph_map:
            actigraph_map[actigraph_key] = item
        if serial_key:
            serial_candidates.setdefault(serial_key, []).append(item)

    serial_map = resolve_serial_map(serial_candidates, gps_valid_counts)

    return actigraph_map, serial_map


def parse_receiver_from_file(file_path: Path) -> str:
    """Extract receiver Actigraph ID from '<actigraph>_60secProxSummary.csv'."""
    return normalize_actigraph(file_path.name.split("_", 1)[0])


def format_time_col(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def file_to_pair_rows(
    file_path: Path,
    actigraph_map: dict[str, dict],
    serial_map: dict[str, dict],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> tuple[list[dict], set[pd.Timestamp]]:
    receiver_actigraph = parse_receiver_from_file(file_path)
    receiver_meta = actigraph_map.get(receiver_actigraph, {})

    raw_df = pd.read_csv(file_path)
    if "Timestamp" not in raw_df.columns:
        raise ValueError(f"Missing 'Timestamp' column in file: {file_path}")

    raw_ts = raw_df["Timestamp"].copy()
    raw_df["Timestamp"] = pd.to_datetime(
        raw_ts,
        format="%d/%m/%Y %I:%M:%S %p",
        errors="coerce",
    )
    if raw_df["Timestamp"].isna().any():
        missing_mask = raw_df["Timestamp"].isna()
        raw_df["Timestamp"] = raw_df["Timestamp"].fillna(
            pd.to_datetime(raw_ts[missing_mask], format="mixed", dayfirst=True, errors="coerce")
        )
    raw_df = raw_df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    raw_df = raw_df[(raw_df["Timestamp"] >= start_ts) & (raw_df["Timestamp"] <= end_ts)]
    if raw_df.empty:
        return [], set()
    raw_df = raw_df.set_index("Timestamp")

    beacon_cols = [col for col in raw_df.columns if col.strip()]
    if not beacon_cols:
        return [], set()

    presence_df = raw_df[beacon_cols].notna().astype("int16")
    agg_df = presence_df.resample("5min", label="right", closed="right").sum().fillna(0).astype("int16")
    agg_df = agg_df[(agg_df.index >= start_ts) & (agg_df.index <= end_ts)]
    if agg_df.empty:
        return [], set()

    time_cols = set(agg_df.index.to_pydatetime())
    pair_rows: list[dict] = []

    for beacon_serial in agg_df.columns:
        beacon_key = normalize_text(beacon_serial)
        beacon_meta = serial_map.get(beacon_key, {})
        row: dict = {
            "receiver_gps_id": receiver_meta.get("gps_id", ""),
            "receiver_actigraph_id": receiver_actigraph,
            "receiver_serial_id": receiver_meta.get("serial_id", ""),
            "beacon_gps_id": beacon_meta.get("gps_id", ""),
            "beacon_actigraph_id": beacon_meta.get("actigraph_id", ""),
            "beacon_serial_id": beacon_key,
            "beacon_collar_design": beacon_meta.get("collar_design", ""),
        }
        series = agg_df[beacon_serial]
        for ts, value in series.items():
            row[format_time_col(ts)] = int(value)
        pair_rows.append(row)

    return pair_rows, set(pd.Timestamp(ts) for ts in time_cols)


def main() -> None:
    input_files = sorted(PROXIMITY_DIR.glob(INPUT_GLOB))
    if not input_files:
        raise FileNotFoundError(f"No files matched pattern {INPUT_GLOB!r} in {PROXIMITY_DIR}")

    start_ts = pd.Timestamp(f"{START_DATE} 00:00:00")
    end_ts = pd.Timestamp(f"{END_DATE} 23:59:59")
    gps_valid_counts = load_gps_valid_counts(GPS_RAW_FILE, start_ts, end_ts)
    actigraph_map, serial_map = load_link_metadata(META_FILE, META_SHEET, gps_valid_counts)

    all_rows: list[dict] = []
    all_time_slots: set[pd.Timestamp] = set()

    for file_path in input_files:
        rows, time_slots = file_to_pair_rows(file_path, actigraph_map, serial_map, start_ts, end_ts)
        all_rows.extend(rows)
        all_time_slots.update(time_slots)

    fixed_cols = [
        "receiver_gps_id",
        "receiver_actigraph_id",
        "receiver_serial_id",
        "beacon_gps_id",
        "beacon_actigraph_id",
        "beacon_serial_id",
        "beacon_collar_design",
    ]
    time_cols = [format_time_col(ts) for ts in sorted(all_time_slots)]

    out_df = pd.DataFrame(all_rows)
    if out_df.empty:
        out_df = pd.DataFrame(columns=fixed_cols + time_cols)
    else:
        for col in fixed_cols:
            if col not in out_df.columns:
                out_df[col] = ""
        for col in time_cols:
            if col not in out_df.columns:
                out_df[col] = pd.NA

        out_df = out_df[fixed_cols + time_cols]
        out_df = out_df.sort_values(
            by=["receiver_actigraph_id", "beacon_actigraph_id", "beacon_serial_id"],
            kind="stable",
        )
        out_df[time_cols] = out_df[time_cols].astype("Int64")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"Processed files: {len(input_files)}")
    print(f"Date filter: {start_ts} to {end_ts}")
    print(f"Output rows (receiver-beacon pairs): {len(out_df)}")
    print(f"Output time slots (5-min): {len(time_cols)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
