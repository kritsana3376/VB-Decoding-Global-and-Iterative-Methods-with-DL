"""Tabulate the probability of decoding failure (Pfail) for each decoder.

Reads the simulator's raw per-trial CSVs, groups by input SER (prop_error),
and writes a summary CSV. No plotting; use replot_Propfail_Eng.py for the
figure.

    Pfail = (count_all - count_success) / count_all

Files whose first six columns do not match REQUIRED_HEADER_PREFIX are skipped.

Usage:
    python plot_tab_Pfail.py RESULTS_FOLDER [-o analysis_summary.csv]
"""

import argparse
import gc
import time

import numpy as np
import pandas as pd

from csv_tools import (
    DECODER_ORDER,
    collect_valid_files,
    read_header,
    write_summary_csv,
)

# =================================================================
# CONFIGURATION
# =================================================================
CHUNK_SIZE = 200_000
OUTPUT_NAME = "analysis_summary.csv"
FLOAT_FORMAT = "%.6e"

# None = auto-detect every success_* column in the file.
# Or give an explicit list, e.g. ["success_EhMP", "success_VSD_AFV"].
DECODER_COLUMNS = None


def pick_success_columns(header):
    if DECODER_COLUMNS is None:
        return [c for c in header if c.startswith("success_")]
    return [c for c in DECODER_COLUMNS if c in header]


_TRUE = {"true", "t", "yes", "y", "1", "1.0"}
_FALSE = {"false", "f", "no", "n", "0", "0.0", "", "nan", "none"}


def to_binary(series):
    """Convert a success_* column to 0/1.

    Handles real booleans, the strings 'True'/'False', and 1/0. The columns
    are written from bool(...), so if pandas reads them back as strings a
    plain to_numeric turns 'True' into NaN and every trial counts as a
    failure, which is why this is done explicitly.
    """
    if series.dtype == bool:
        return series.astype("int64")

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return (numeric != 0).astype("int64")

    text = series.astype(str).str.strip().str.lower()
    unknown = set(text.unique()) - _TRUE - _FALSE
    if unknown:
        raise ValueError(
            f"Column {series.name!r} holds values that are not 0/1: "
            f"{sorted(unknown)[:5]}"
        )
    return text.isin(_TRUE).astype("int64")


# =================================================================
def analyze_decoders(data_folder_path, output_name=OUTPUT_NAME):
    valid_files, _ = collect_valid_files(data_folder_path, "success_")

    agg = pd.DataFrame()
    all_decoders = []

    for path in valid_files:
        t0 = time.time()
        success_cols = pick_success_columns(read_header(path))
        if not success_cols:
            print(f"[skip] {path.name}   no matching success_* column")
            continue
        for column in success_cols:
            if column not in all_decoders:
                all_decoders.append(column)

        print(f"Processing: {path.name}   ({len(success_cols)} decoders)")

        for chunk in pd.read_csv(path, usecols=["prop_error"] + success_cols,
                                 chunksize=CHUNK_SIZE):
            for column in success_cols:
                chunk[column] = to_binary(chunk[column])

            grouped = chunk.groupby("prop_error").agg(["sum", "count"])
            grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
            grouped = grouped.reset_index()

            agg = grouped if agg.empty else (
                pd.concat([agg, grouped], ignore_index=True)
                  .groupby("prop_error", as_index=False).sum()
            )
            del chunk, grouped
            gc.collect()

        print(f"   done in {time.time() - t0:.1f}s")

    if agg.empty:
        print("No data to process")
        return None

    # ---------- Pfail = (count_all - count_success) / count_all ----------
    agg = agg.sort_values("prop_error").reset_index(drop=True)
    out = pd.DataFrame({"prop_error": agg["prop_error"]})

    count_cols = [c for c in agg.columns if c.endswith("_count")]
    out["count_all"] = agg[count_cols].max(axis=1).astype("int64")

    rank = {f"success_{name}": i for i, name in enumerate(DECODER_ORDER)}
    all_decoders = sorted(all_decoders, key=lambda c: (rank.get(c, len(rank)), c))

    found = []
    for column in all_decoders:
        sum_col, count_col = f"{column}_sum", f"{column}_count"
        if sum_col not in agg.columns:
            continue
        name = column[len("success_"):]
        n_all = agg[count_col]
        n_ok = agg[sum_col]
        out[f"pfail_{name}"] = (n_all - n_ok) / n_all.replace(0, np.nan)
        found.append((name, column))

    # Raw counts appended so the ratios can be audited later.
    for name, column in found:
        out[f"count_success_{name}"] = agg[f"{column}_sum"].astype("int64")
        out[f"count_all_{name}"] = agg[f"{column}_count"].astype("int64")

    summary_path = valid_files[0].parent / output_name
    write_summary_csv(out, summary_path,
                      [f"pfail_{name}" for name, _ in found], FLOAT_FORMAT)

    print(f"\nFound {len(found)} decoder(s): {', '.join(n for n, _ in found)}")
    print(f"Summary saved to: {summary_path}")

    with pd.option_context("display.width", 250, "display.max_columns", 100):
        cols = ["prop_error", "count_all"] + [f"pfail_{n}" for n, _ in found]
        print("\n" + out[cols].to_string(index=False))

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="folder holding the simulator's raw CSV files")
    parser.add_argument("-o", "--output", default=OUTPUT_NAME,
                        help=f"summary file name (default: {OUTPUT_NAME})")
    args = parser.parse_args()
    analyze_decoders(args.folder, args.output)


if __name__ == "__main__":
    main()
