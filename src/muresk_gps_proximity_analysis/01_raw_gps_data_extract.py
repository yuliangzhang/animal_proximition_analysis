from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT_DIR = Path("data/muresk_farm_proximity_data/muresk_gps_data")
OUTPUT_FILE = ROOT_DIR / "muresk_gps_raw_data.csv"
FILE_PATTERN = re.compile(r"^GPS\d{4}_(\d{6})\.txt$")
INPUT_COLUMNS = [
    "Hours",
    "minutes",
    "seconds",
    "latitude",
    "longitude",
    "altitude",
    "speed",
    "course",
    "HDOP",
    "satellites",
]
OUTPUT_COLUMNS = [
    "gps_code",
    "timestamp",
    "latitude",
    "longitude",
    "altitude",
    "speed",
    "course",
    "HDOP",
    "satellites",
]


def parse_file_date(file_name: str) -> pd.Timestamp | None:
    """
    Parse YYMMDD from filename.
    Business rule from requirement:
    GPS0000_250910.txt represents data date 2025-09-11 (file date + 1 day).
    """
    match = FILE_PATTERN.match(file_name)
    if not match:
        return None
    raw_date = pd.to_datetime(match.group(1), format="%y%m%d", errors="coerce")
    if pd.isna(raw_date):
        return None
    return raw_date + pd.Timedelta(days=1)


def load_single_file(file_path: Path) -> pd.DataFrame | None:
    file_date = parse_file_date(file_path.name)
    if file_date is None:
        print(f"[Skip] Invalid filename format: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:  # pragma: no cover
        print(f"[Skip] Failed to read {file_path}: {exc}")
        return None

    df.columns = [str(col).strip() for col in df.columns]
    missing = [col for col in INPUT_COLUMNS if col not in df.columns]
    if missing:
        # Some files have no header row. Re-read with fixed schema.
        try:
            df = pd.read_csv(file_path, header=None, names=INPUT_COLUMNS)
        except Exception as exc:  # pragma: no cover
            print(f"[Skip] Failed to parse headerless file {file_path}: {exc}")
            return None
    else:
        df = df[INPUT_COLUMNS].copy()
    for col in ["Hours", "minutes", "seconds"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build timestamp from file date and row-level h/m/s.
    timestamp = (
        file_date
        + pd.to_timedelta(df["Hours"], unit="h")
        + pd.to_timedelta(df["minutes"], unit="m")
        + pd.to_timedelta(df["seconds"], unit="s")
    )

    result = pd.DataFrame(
        {
            "gps_code": file_path.parent.name,
            "timestamp": timestamp.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": df["latitude"],
            "longitude": df["longitude"],
            "altitude": df["altitude"],
            "speed": df["speed"],
            "course": df["course"],
            "HDOP": df["HDOP"],
            "satellites": df["satellites"],
        }
    )
    result = result.dropna(subset=["timestamp"])
    return result


def collect_all_gps_data(root_dir: Path) -> pd.DataFrame:
    all_files = sorted(root_dir.glob("N*/GPS*.txt"))
    if not all_files:
        raise FileNotFoundError(f"No GPS text files found under: {root_dir}")

    dataframes: list[pd.DataFrame] = []
    for file_path in all_files:
        file_df = load_single_file(file_path)
        if file_df is not None and not file_df.empty:
            dataframes.append(file_df)

    if not dataframes:
        raise ValueError("No valid GPS files were parsed.")

    combined = pd.concat(dataframes, ignore_index=True)
    return combined[OUTPUT_COLUMNS]


def main() -> None:
    gps_df = collect_all_gps_data(ROOT_DIR)
    gps_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Saved {len(gps_df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
