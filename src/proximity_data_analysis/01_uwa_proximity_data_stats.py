"""Aggregate UWA ram-ewe proximity data by day and by 4-hour windows.

Input:
- Excel files in data/uwa_proximity_data/ matching *5min*UWA.xlsx

Output:
- data/uwa_proximity_data/HS_proximity_<YEAR>_day_UWA.csv
- data/uwa_proximity_data/HS_proximity_<YEAR>_4h_UWA.csv

Notes:
- The first five columns are treated as metadata columns.
- All remaining columns are treated as timestamp columns from 5-minute intervals.
- Empty values are interpreted as no proximity events.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple
from xml.etree import ElementTree as ET
from zipfile import ZipFile


DATA_DIR = Path("data/uwa_proximity_data")
INPUT_GLOB = "*5min*UWA.xlsx"
META_COL_COUNT = 5

NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def col_ref_to_index(cell_ref: str) -> int:
    """Convert an Excel cell reference (e.g. BC12) to 1-based column index."""
    col = 0
    for ch in cell_ref:
        if ch.isalpha():
            col = col * 26 + (ord(ch.upper()) - ord("A") + 1)
        else:
            break
    return col


def excel_serial_to_datetime(serial: float) -> datetime:
    """Convert Excel serial date to datetime (Excel 1900 date system)."""
    return datetime(1899, 12, 30) + timedelta(days=serial)


def parse_timestamp_header(raw_value: str) -> datetime:
    """Parse a timestamp column header from common text/serial formats."""
    text = str(raw_value).strip()

    # Numeric Excel serial date, as seen in these files.
    try:
        numeric = float(text)
    except ValueError:
        numeric = None
    if numeric is not None:
        return excel_serial_to_datetime(numeric)

    # Fallback for text timestamps if source format changes.
    for fmt in (
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d-%m-%Y %H:%M",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unsupported timestamp header format: {raw_value!r}")


def format_day_label(dt: datetime) -> str:
    """Use day/month/year labels matching the requested style."""
    return f"{dt.day}/{dt.month}/{dt.year}"


def format_4h_label(dt: datetime) -> str:
    """Use start time label for each 4-hour bucket."""
    return f"{dt.day}/{dt.month}/{dt.year} {dt.hour:02d}:00"


def parse_numeric_count(value: str) -> float:
    """Parse a proximity count value; empty strings mean zero."""
    text = str(value).strip()
    if not text:
        return 0.0
    return float(text)


def format_count(value: float) -> str:
    """Write empty string for zero; integer-like values as ints."""
    if abs(value) < 1e-12:
        return ""
    rounded = round(value)
    if abs(value - rounded) < 1e-12:
        return str(int(rounded))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def load_shared_strings(zf: ZipFile) -> List[str]:
    """Load shared strings table from workbook, if present."""
    try:
        with zf.open("xl/sharedStrings.xml") as fp:
            root = ET.parse(fp).getroot()
    except KeyError:
        return []

    shared: List[str] = []
    for si in root.findall(f"{{{NS_MAIN}}}si"):
        parts = [node.text or "" for node in si.findall(f".//{{{NS_MAIN}}}t")]
        shared.append("".join(parts))
    return shared


def get_first_sheet_path(zf: ZipFile) -> str:
    """Resolve the file path of the first worksheet in the workbook."""
    with zf.open("xl/workbook.xml") as fp:
        workbook_root = ET.parse(fp).getroot()
    sheets = workbook_root.find(f"{{{NS_MAIN}}}sheets")
    if sheets is None:
        raise ValueError("Workbook has no sheets")
    first_sheet = sheets.findall(f"{{{NS_MAIN}}}sheet")[0]
    rel_id = first_sheet.attrib[f"{{{NS_REL}}}id"]

    with zf.open("xl/_rels/workbook.xml.rels") as fp:
        rels_root = ET.parse(fp).getroot()
    for rel in rels_root.findall(f"{{{NS_PACKAGE_REL}}}Relationship"):
        if rel.attrib.get("Id") == rel_id:
            target = rel.attrib["Target"]
            if target.startswith("/"):
                return target.lstrip("/")
            if target.startswith("xl/"):
                return target
            return f"xl/{target}"

    raise ValueError(f"Could not resolve sheet relationship {rel_id}")


def extract_cell_value(cell: ET.Element, shared_strings: List[str]) -> str:
    """Extract string value from a worksheet cell element."""
    cell_type = cell.attrib.get("t")
    v_node = cell.find(f"{{{NS_MAIN}}}v")
    if cell_type == "s":
        if v_node is None or v_node.text is None:
            return ""
        idx = int(v_node.text)
        return shared_strings[idx] if 0 <= idx < len(shared_strings) else ""
    if cell_type == "inlineStr":
        is_node = cell.find(f"{{{NS_MAIN}}}is")
        if is_node is None:
            return ""
        text_parts = [node.text or "" for node in is_node.findall(f".//{{{NS_MAIN}}}t")]
        return "".join(text_parts)
    if v_node is None or v_node.text is None:
        return ""
    return v_node.text


def iter_sheet_rows(
    zf: ZipFile,
    sheet_path: str,
    shared_strings: List[str],
) -> Iterator[Tuple[int, Dict[int, str]]]:
    """Yield worksheet rows as (row_number, {column_index: string_value})."""
    with zf.open(sheet_path) as fp:
        context = ET.iterparse(fp, events=("end",))
        row_tag = f"{{{NS_MAIN}}}row"
        cell_tag = f"{{{NS_MAIN}}}c"

        for _, elem in context:
            if elem.tag != row_tag:
                continue

            row_number = int(elem.attrib.get("r", "0"))
            values: Dict[int, str] = {}
            for cell in elem.findall(cell_tag):
                ref = cell.attrib.get("r", "")
                col_idx = col_ref_to_index(ref)
                values[col_idx] = extract_cell_value(cell, shared_strings)

            yield row_number, values
            elem.clear()


def aggregate_workbook(input_path: Path, output_dir: Path) -> Tuple[Path, Path]:
    """Aggregate one workbook and write day-level and 4-hour-level CSV outputs."""
    year_match = re.search(r"(20\d{2})", input_path.name)
    if not year_match:
        raise ValueError(f"Could not infer year from file name: {input_path.name}")
    year = year_match.group(1)

    output_day = output_dir / f"HS_proximity_{year}_day_UWA.csv"
    output_4h = output_dir / f"HS_proximity_{year}_4h_UWA.csv"

    with ZipFile(input_path) as zf:
        shared_strings = load_shared_strings(zf)
        sheet_path = get_first_sheet_path(zf)

        rows_iter = iter_sheet_rows(zf, sheet_path, shared_strings)

        try:
            _, header_cells = next(rows_iter)
        except StopIteration as exc:
            raise ValueError(f"Worksheet is empty: {input_path.name}") from exc

        # Keep metadata headers in original order.
        meta_headers = [header_cells.get(i, f"meta_col_{i}") for i in range(1, META_COL_COUNT + 1)]

        # Build mapping of time columns from header row.
        time_col_to_dt: Dict[int, datetime] = {}
        for col_idx, raw_header in header_cells.items():
            if col_idx <= META_COL_COUNT:
                continue
            text = str(raw_header).strip()
            if not text:
                continue
            try:
                time_col_to_dt[col_idx] = parse_timestamp_header(text)
            except ValueError:
                # Non-timestamp columns after metadata are ignored.
                continue

        if not time_col_to_dt:
            raise ValueError(f"No timestamp columns found in {input_path.name}")

        day_keys = sorted({datetime(dt.year, dt.month, dt.day) for dt in time_col_to_dt.values()})
        zone_keys = sorted(
            {
                datetime(dt.year, dt.month, dt.day, (dt.hour // 4) * 4, 0, 0)
                for dt in time_col_to_dt.values()
            }
        )

        day_headers = [format_day_label(dt) for dt in day_keys]
        zone_headers = [format_4h_label(dt) for dt in zone_keys]

        day_rows: List[List[str]] = []
        zone_rows: List[List[str]] = []

        for _, cell_map in rows_iter:
            if not cell_map:
                continue

            meta_values = [str(cell_map.get(i, "")).strip() for i in range(1, META_COL_COUNT + 1)]
            if not any(meta_values):
                continue

            day_counts = defaultdict(float)
            zone_counts = defaultdict(float)

            for col_idx, raw_value in cell_map.items():
                if col_idx <= META_COL_COUNT:
                    continue
                ts = time_col_to_dt.get(col_idx)
                if ts is None:
                    continue
                count = parse_numeric_count(raw_value)
                if count == 0:
                    continue

                day_key = datetime(ts.year, ts.month, ts.day)
                zone_key = datetime(ts.year, ts.month, ts.day, (ts.hour // 4) * 4, 0, 0)

                day_counts[day_key] += count
                zone_counts[zone_key] += count

            day_values = [format_count(day_counts.get(day_key, 0.0)) for day_key in day_keys]
            zone_values = [format_count(zone_counts.get(zone_key, 0.0)) for zone_key in zone_keys]

            day_rows.append(meta_values + day_values)
            zone_rows.append(meta_values + zone_values)

    output_dir.mkdir(parents=True, exist_ok=True)

    with output_day.open("w", newline="", encoding="utf-8") as f_day:
        writer = csv.writer(f_day)
        writer.writerow(meta_headers + day_headers)
        writer.writerows(day_rows)

    with output_4h.open("w", newline="", encoding="utf-8") as f_4h:
        writer = csv.writer(f_4h)
        writer.writerow(meta_headers + zone_headers)
        writer.writerows(zone_rows)

    return output_day, output_4h


def find_input_files(data_dir: Path) -> Iterable[Path]:
    """Find valid input workbooks and skip temporary lock files."""
    return sorted(
        path
        for path in data_dir.glob(INPUT_GLOB)
        if path.is_file() and not path.name.startswith("~$")
    )


def main() -> None:
    input_files = list(find_input_files(DATA_DIR))
    if not input_files:
        raise FileNotFoundError(f"No input files found in {DATA_DIR} with pattern {INPUT_GLOB}")

    print(f"Found {len(input_files)} input workbook(s)")
    for input_file in input_files:
        print(f"Processing: {input_file}")
        out_day, out_4h = aggregate_workbook(input_file, DATA_DIR)
        print(f"  Wrote day-level output: {out_day}")
        print(f"  Wrote 4-hour output: {out_4h}")


if __name__ == "__main__":
    main()
