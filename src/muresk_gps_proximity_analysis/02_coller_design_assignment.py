from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_GPS_FILE = Path("data/muresk_farm_proximity_data/muresk_gps_data/muresk_gps_raw_data.csv")
MASTERFILE = Path("data/muresk_farm_proximity_data/muresk_proximity_data/Uwa_logger_sensors_2025.xlsx")
MASTER_SHEET = "Link"


def normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""
    return "".join(str(value).upper().split())


def load_coller_design_mapping(masterfile: Path, sheet_name: str) -> dict[str, str]:
    link_df = pd.read_excel(masterfile, sheet_name=sheet_name, engine="openpyxl")

    required_cols = {"GPS", "Coller design"}
    missing = required_cols - set(link_df.columns)
    if missing:
        raise ValueError(f"Missing columns in masterfile sheet '{sheet_name}': {sorted(missing)}")

    mapping_df = link_df[["GPS", "Coller design"]].copy()
    mapping_df["gps_norm"] = mapping_df["GPS"].apply(normalize_code)
    mapping_df = mapping_df[mapping_df["gps_norm"] != ""]
    mapping_df["coller_design"] = mapping_df["Coller design"].astype("string").str.strip()
    mapping_df = mapping_df.dropna(subset=["coller_design"])
    mapping_df = mapping_df.drop_duplicates(subset=["gps_norm"], keep="first")

    return dict(zip(mapping_df["gps_norm"], mapping_df["coller_design"], strict=False))


def assign_coller_design(raw_gps_file: Path, mapping: dict[str, str]) -> tuple[int, int, int]:
    gps_df = pd.read_csv(raw_gps_file)
    if "gps_code" not in gps_df.columns:
        raise ValueError("Column 'gps_code' not found in raw GPS file.")

    gps_df["gps_norm"] = gps_df["gps_code"].apply(normalize_code)
    gps_df["coller_design"] = gps_df["gps_norm"].map(mapping)
    gps_df = gps_df.drop(columns=["gps_norm"])
    gps_df.to_csv(raw_gps_file, index=False, encoding="utf-8")

    total_rows = len(gps_df)
    matched_rows = int(gps_df["coller_design"].notna().sum())
    unmatched_rows = total_rows - matched_rows
    return total_rows, matched_rows, unmatched_rows


def main() -> None:
    mapping = load_coller_design_mapping(MASTERFILE, MASTER_SHEET)
    total_rows, matched_rows, unmatched_rows = assign_coller_design(RAW_GPS_FILE, mapping)
    print(f"Updated file: {RAW_GPS_FILE}")
    print(f"Total rows: {total_rows}")
    print(f"Matched rows: {matched_rows}")
    print(f"Unmatched rows: {unmatched_rows}")


if __name__ == "__main__":
    main()
