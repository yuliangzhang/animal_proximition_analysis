import pandas as pd
import os


MASTERFILE_IN = "data/masterfile/masterfile_2024.csv"
GPS_MASTER_XLSX = "data/raw_data/2024-GPS-Summer-Recovery-MASTER.xlsx"
MASTERFILE_OUT = "data/masterfile/masterfile_2024_gps_code.csv"


def _norm_str(s):
    if pd.isna(s):
        return ""
    return str(s).strip()


def _norm_key(s):
    """Normalize EID-like keys: strip spaces and uppercase (no hyphen insertion here)."""
    s = _norm_str(s)
    return s.replace(" ", "").upper()


def _norm_code(s):
    """Normalize GPS code: remove spaces and slashes, keep as string."""
    s = _norm_str(s)
    return s.replace(" ", "").replace("/", "-")


def load_masterfile(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure expected columns exist
    for c in ["collar_vid_2024_on", "gps_2024"]:
        if c not in df.columns:
            df[c] = None
    # Normalize keys used for matching
    df["collar_vid_2024_on"] = df["collar_vid_2024_on"].astype(str).map(_norm_key)
    return df


def load_gps_mapping(xlsx_path: str) -> dict:
    """
    Build mapping from 'Collar EID' -> 'GPS #' using both column sets if present.
    """
    xl = pd.ExcelFile(xlsx_path)
    if "Sheet1" in xl.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name="Sheet1")
    else:
        # Fallback: first sheet
        df = pd.read_excel(xlsx_path)

    mapping = {}

    candidates = [
        ("Collar EID", "GPS #"),
        ("Collar EID.1", "GPS #.1"),
    ]

    for k_col, v_col in candidates:
        if k_col in df.columns and v_col in df.columns:
            sub = df[[k_col, v_col]].dropna(how="all")
            for _, row in sub.iterrows():
                key = _norm_key(row.get(k_col))
                val = _norm_code(row.get(v_col))
                if key and val:
                    mapping[key] = val

    return mapping


def apply_mapping(master_df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    before_filled = master_df["gps_2024"].notna().sum()

    # Fill only missing gps_2024 values
    def fill_gps(row):
        gps = row.get("gps_2024")
        if pd.notna(gps) and str(gps).strip() not in ("", "nan", "None"):
            return _norm_code(gps)
        key = row.get("collar_vid_2024_on")
        code = mapping.get(_norm_key(key), None)
        return _norm_code(code) if code else None

    master_df["gps_2024"] = master_df.apply(fill_gps, axis=1)
    after_filled = master_df["gps_2024"].notna().sum()

    print(f"gps_2024 non-empty before: {before_filled}, after: {after_filled}, newly filled: {after_filled - before_filled}")
    return master_df


def main(
    master_in: str = MASTERFILE_IN,
    gps_xlsx: str = GPS_MASTER_XLSX,
    master_out: str = MASTERFILE_OUT,
):
    print(f"Loading masterfile: {master_in}")
    mf = load_masterfile(master_in)
    print(f"Masterfile rows: {len(mf)}")

    print(f"Loading GPS master: {gps_xlsx}")
    mapping = load_gps_mapping(gps_xlsx)
    print(f"GPS mapping entries: {len(mapping)}")

    print("Applying mapping to masterfile...")
    mf2 = apply_mapping(mf, mapping)

    os.makedirs(os.path.dirname(master_out), exist_ok=True)
    mf2.to_csv(master_out, index=False)
    print(f"Written: {master_out}")


if __name__ == "__main__":
    main()

