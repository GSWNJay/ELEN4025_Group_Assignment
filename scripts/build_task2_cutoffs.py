"""Build Task 2 row-level VLE cutoff files for the OULAD project.

This script prepares the handoff from Task 2 to Task 3. It extracts the raw
OULAD CSVs if needed, joins studentVle rows to the vle activity lookup, and
writes one cumulative row-level file per week cutoff. It deliberately does not
aggregate, impute, encode, or create model-ready features.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd


REQUIRED_RAW_FILES = ("studentInfo.csv", "studentVle.csv", "vle.csv")
STUDENT_INFO_KEYS = ("id_student", "code_module", "code_presentation")
STUDENT_VLE_COLUMNS = (
    "code_module",
    "code_presentation",
    "id_student",
    "id_site",
    "date",
    "sum_click",
)
VLE_LOOKUP_COLUMNS = ("id_site", "code_module", "code_presentation", "activity_type")
JOIN_KEYS = ["id_site", "code_module", "code_presentation"]
OUTPUT_COLUMNS = [
    "code_module",
    "code_presentation",
    "id_student",
    "id_site",
    "date",
    "sum_click",
    "activity_type",
]
CUTOFFS = {
    2: 13,
    4: 27,
    6: 41,
    8: 55,
}
EXPECTED_ROW_COUNTS = {
    2: 1_757_383,
    4: 2_801_311,
    6: 3_562_771,
    8: 4_302_651,
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build row-level studentVle cutoff files for Task 3."
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=repo_root / "anonymisedData.zip",
        help="Path to the OULAD anonymisedData.zip file.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=repo_root / "data" / "raw",
        help="Folder containing or receiving raw OULAD CSVs.",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=repo_root / "data" / "interim",
        help="Folder for Task 2 cutoff handoff files.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Number of studentVle rows to process per chunk.",
    )
    parser.add_argument(
        "--skip-count-check",
        action="store_true",
        help="Skip validation against the known OULAD cutoff row counts.",
    )
    return parser.parse_args()


def ensure_raw_files(zip_path: Path, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in REQUIRED_RAW_FILES if not (raw_dir / name).exists()]
    if not missing:
        print(f"Raw files already present in {raw_dir}")
        return

    if not zip_path.exists():
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing raw files ({missing_list}) and zip file was not found: {zip_path}"
        )

    print(f"Extracting {', '.join(missing)} from {zip_path} to {raw_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_names = set(zip_ref.namelist())
        for file_name in missing:
            if file_name not in zip_names:
                raise FileNotFoundError(f"{file_name} is not present in {zip_path}")
            zip_ref.extract(file_name, raw_dir)


def read_csv_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def require_columns(path: Path, required: tuple[str, ...] | list[str]) -> None:
    columns = set(read_csv_header(path))
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def load_vle_lookup(vle_path: Path) -> pd.DataFrame:
    require_columns(vle_path, VLE_LOOKUP_COLUMNS)
    lookup = pd.read_csv(vle_path, usecols=VLE_LOOKUP_COLUMNS)
    duplicate_count = lookup.duplicated(subset=JOIN_KEYS).sum()
    if duplicate_count:
        raise ValueError(
            f"{vle_path} has {duplicate_count} duplicate rows for join keys {JOIN_KEYS}"
        )
    return lookup


def validate_chunk(
    chunk: pd.DataFrame,
    merged: pd.DataFrame,
    sum_click_before: pd.Series,
    chunk_number: int,
) -> None:
    if len(merged) != len(chunk):
        raise ValueError(
            f"Chunk {chunk_number}: merge changed row count from {len(chunk)} "
            f"to {len(merged)}"
        )

    if not pd.api.types.is_numeric_dtype(merged["date"]):
        raise TypeError(f"Chunk {chunk_number}: date is not numeric after parsing")

    sum_click_after = merged["sum_click"].reset_index(drop=True)
    if not sum_click_after.equals(sum_click_before.reset_index(drop=True)):
        raise ValueError(f"Chunk {chunk_number}: sum_click changed during merge")

    missing_activity_type = merged["activity_type"].isna().sum()
    if missing_activity_type:
        raise ValueError(
            f"Chunk {chunk_number}: {missing_activity_type} rows have no activity_type"
        )


def build_cutoff_files(
    student_vle_path: Path,
    lookup: pd.DataFrame,
    interim_dir: Path,
    chunksize: int,
) -> dict[int, int]:
    interim_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        week: interim_dir / f"studentVle_week{week}.csv" for week in CUTOFFS
    }

    for path in output_paths.values():
        if path.exists():
            path.unlink()

    row_counts = {week: 0 for week in CUTOFFS}
    wrote_header = {week: False for week in CUTOFFS}
    rows_read = 0

    reader = pd.read_csv(
        student_vle_path,
        usecols=STUDENT_VLE_COLUMNS,
        chunksize=chunksize,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        rows_read += len(chunk)
        chunk["date"] = pd.to_numeric(chunk["date"], errors="raise")
        sum_click_before = chunk["sum_click"].copy()
        merged = chunk.merge(lookup, on=JOIN_KEYS, how="left", validate="many_to_one")
        validate_chunk(chunk, merged, sum_click_before, chunk_number)

        for week, cutoff in CUTOFFS.items():
            filtered = merged.loc[merged["date"] <= cutoff, OUTPUT_COLUMNS]
            if filtered.empty:
                continue

            filtered.to_csv(
                output_paths[week],
                index=False,
                mode="a" if wrote_header[week] else "w",
                header=not wrote_header[week],
            )
            wrote_header[week] = True
            row_counts[week] += len(filtered)

        print(f"Processed chunk {chunk_number} ({rows_read:,} rows read)")

    for week, path in output_paths.items():
        if not wrote_header[week]:
            pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(path, index=False)

    return row_counts


def validate_expected_counts(row_counts: dict[int, int]) -> None:
    mismatches = {
        week: (row_counts[week], expected)
        for week, expected in EXPECTED_ROW_COUNTS.items()
        if row_counts[week] != expected
    }
    if mismatches:
        formatted = ", ".join(
            f"week {week}: got {actual:,}, expected {expected:,}"
            for week, (actual, expected) in mismatches.items()
        )
        raise ValueError(f"Cutoff row count validation failed: {formatted}")


def main() -> int:
    args = parse_args()
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be a positive integer")

    ensure_raw_files(args.zip_path, args.raw_dir)

    student_info_path = args.raw_dir / "studentInfo.csv"
    student_vle_path = args.raw_dir / "studentVle.csv"
    vle_path = args.raw_dir / "vle.csv"

    require_columns(student_info_path, STUDENT_INFO_KEYS)
    require_columns(student_vle_path, STUDENT_VLE_COLUMNS)
    lookup = load_vle_lookup(vle_path)

    row_counts = build_cutoff_files(
        student_vle_path=student_vle_path,
        lookup=lookup,
        interim_dir=args.interim_dir,
        chunksize=args.chunksize,
    )

    if not args.skip_count_check:
        validate_expected_counts(row_counts)

    print("Task 2 cutoff files built successfully:")
    for week, row_count in row_counts.items():
        cutoff = CUTOFFS[week]
        output_path = args.interim_dir / f"studentVle_week{week}.csv"
        print(f"  week {week} (date <= {cutoff}): {row_count:,} rows -> {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
