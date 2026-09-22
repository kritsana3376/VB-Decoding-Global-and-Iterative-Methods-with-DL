"""Tabulate D-SER and D-USR for each decoder, grouped by input SER.

D-SER is the data symbol error rate after decoding: the fraction of the k
data symbols still wrong. D-USR is the data unverified symbol rate: the
fraction of data symbols the decoder could not verify.

Reads the simulator's raw per-trial CSVs and writes a summary CSV. No
plotting; use replot_SER_USR_ENG.py for the figure.

Files whose first six columns do not match REQUIRED_HEADER_PREFIX are skipped.

Usage:
    python plot_tab_output_SER_USR.py RESULTS_FOLDER [-o SER_plot_summary.csv]
"""

import argparse
import gc
import re
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from csv_tools import (
    DECODER_ORDER,
    collect_valid_files,
    read_code_parameters,
    resolve_verified,
    write_summary_csv,
)

# =================================================================
# CONFIGURATION
# =================================================================
CHUNK_SIZE = 50_000
OUTPUT_NAME = "SER_plot_summary.csv"
FLOAT_FORMAT = "%.5e"      # still parses back as a number
INCLUDE_RAW_SUMS = False   # True also writes the raw sum/count columns

# Column-name suffix for each decoder.
DECODER_SUFFIXES = DECODER_ORDER

_KEY_RE = re.compile(r"(\d+)\s*:")


# =================================================================
# Per-value helpers, cached because the same strings repeat constantly
# (an all-ones verified list, an empty error dict, and so on).
# =================================================================
@lru_cache(maxsize=1_000_000)
def _count_keys_below(dict_str, k):
    """Count keys in a dict repr whose index is < k, i.e. data symbols only."""
    if not dict_str or dict_str == "{}":
        return 0
    return sum(1 for key in _KEY_RE.findall(dict_str) if int(key) < k)


@lru_cache(maxsize=1_000_000)
def _count_zeros_prefix(list_str, k):
    """Count zeros in the first k entries of a verified list '[1 0 1 ...]'."""
    if not list_str:
        return 0
    cleaned = list_str.strip().strip("[]")
    if not cleaned:
        return 0
    return cleaned.split()[:k].count("0")


def count_keys_series(series, k):
    return series.fillna("{}").astype(str).map(lambda s: _count_keys_below(s, k))


def count_zeros_series(series, k):
    return series.fillna("").astype(str).map(lambda s: _count_zeros_prefix(s, k))


# =================================================================
def process_folder(folder_path, output_name=OUTPUT_NAME):
    valid_files, _ = collect_valid_files(folder_path, "unsolved_errors_")

    # Code parameters come from the data, so (31,21) and (31,16) both work.
    _, k_fallback = read_code_parameters(valid_files[0])

    agg_result = pd.DataFrame()

    for path in valid_files:
        t0 = time.time()
        print(f"Processing: {path.name}")

        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
            if "k" in chunk.columns:
                k_vals = pd.to_numeric(chunk["k"], errors="coerce").dropna().unique()
                k_const = int(k_vals[0]) if len(k_vals) == 1 else None
            else:
                k_const = k_fallback

            temp = pd.DataFrame({"prop_error": chunk["prop_error"].values})
            k_per_row = None
            if k_const is None:
                # k varies inside this chunk, so work row by row.
                k_per_row = (pd.to_numeric(chunk["k"], errors="coerce")
                               .fillna(k_fallback).astype(int))

            for suffix in DECODER_SUFFIXES:
                unsolved_col = f"unsolved_errors_{suffix}"
                verified_col = resolve_verified(chunk.columns, suffix)
                if unsolved_col not in chunk.columns:
                    continue

                if k_const is not None:
                    n_unsolved = count_keys_series(chunk[unsolved_col], k_const)
                    denom = k_const
                else:
                    n_unsolved = pd.Series(
                        [_count_keys_below(str(s) if pd.notna(s) else "{}", int(kk))
                         for s, kk in zip(chunk[unsolved_col], k_per_row)],
                        index=chunk.index)
                    denom = k_per_row

                temp[f"ser_ratio_{suffix}"] = (n_unsolved / denom).values

                if verified_col is not None and verified_col in chunk.columns:
                    if k_const is not None:
                        n_zeros = count_zeros_series(chunk[verified_col], k_const)
                    else:
                        n_zeros = pd.Series(
                            [_count_zeros_prefix(str(s) if pd.notna(s) else "", int(kk))
                             for s, kk in zip(chunk[verified_col], k_per_row)],
                            index=chunk.index)
                    temp[f"usr_ratio_{suffix}"] = (n_zeros / denom).values

            chunk_agg = temp.groupby("prop_error").agg(["sum", "count"])
            chunk_agg.columns = [f"{c[0]}_{c[1]}" for c in chunk_agg.columns]
            chunk_agg = chunk_agg.reset_index()

            if agg_result.empty:
                agg_result = chunk_agg
            else:
                agg_result = (pd.concat([agg_result, chunk_agg], ignore_index=True)
                                .groupby("prop_error", as_index=False).sum())

            del chunk, temp, chunk_agg
            gc.collect()

        print(f"   done in {time.time() - t0:.1f}s")

    if agg_result.empty:
        print("No data to process")
        return None

    # ---------- Mean = Sum / Count ----------
    agg_result = agg_result.sort_values("prop_error").reset_index(drop=True)
    summary_df = pd.DataFrame({"prop_error": agg_result["prop_error"]})

    count_cols = [c for c in agg_result.columns if c.endswith("_count")]
    if count_cols:
        summary_df["n_codeword"] = agg_result[count_cols].max(axis=1).astype("int64")

    found = []
    for suffix in DECODER_SUFFIXES:
        for kind in ("ser", "usr"):
            sum_col = f"{kind}_ratio_{suffix}_sum"
            count_col = f"{kind}_ratio_{suffix}_count"
            if sum_col in agg_result.columns:
                summary_df[f"{kind}_ratio_{suffix}"] = (
                    agg_result[sum_col] / agg_result[count_col].replace(0, np.nan))
                if INCLUDE_RAW_SUMS:
                    summary_df[f"{kind}_sum_{suffix}"] = agg_result[sum_col]
                    summary_df[f"{kind}_count_{suffix}"] = agg_result[count_col]
        if f"ser_ratio_{suffix}" in summary_df.columns:
            found.append(suffix)

    if not found:
        print("No decoder columns found in any readable file")
        return None

    summary_path = valid_files[0].parent / output_name
    ratio_cols = [c for c in summary_df.columns
                  if c.startswith(("ser_ratio_", "usr_ratio_"))]
    write_summary_csv(summary_df, summary_path, ratio_cols, FLOAT_FORMAT)

    print(f"\nFound {len(found)} decoder(s): {', '.join(found)}")
    print(f"Summary saved to: {summary_path}")

    with pd.option_context("display.width", 200, "display.max_columns", 100):
        print("\n" + summary_df[["prop_error", "n_codeword"]
                                + [f"ser_ratio_{s}" for s in found]].to_string(index=False))

    return summary_df


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="folder holding the simulator's raw CSV files")
    parser.add_argument("-o", "--output", default=OUTPUT_NAME,
                        help=f"summary file name (default: {OUTPUT_NAME})")
    args = parser.parse_args()
    process_folder(args.folder, args.output)


if __name__ == "__main__":
    main()
