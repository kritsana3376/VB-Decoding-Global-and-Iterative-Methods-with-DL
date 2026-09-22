#!/usr/bin/env python3
"""Summarise verified / false unverified / false verified symbols per decoder.

Results are split by input SER (prop_error) and by data vs parity symbol.

Input columns (from the simulator's raw p*.csv files):
  prop_error            input SER for that trial
  initial_errors_<DEC>  dict repr, "{5: 3294145190, 12: ...}"; keys only
  unsolved_errors_<DEC> dict repr, same treatment
  verified_list_<DEC>   numpy repr, "[1 0 1 1 ...]", length n

Definitions:
  verified         verified_list[i] == 1
  false verified   verified_list[i] == 1 and i is in unsolved_errors
  false unverified verified_list[i] == 0 and i is not in initial_errors
  data symbols are indices 0..k-1, parity symbols are k..n-1

n and k are read from the CSV, so BCH(31,21) and BCH(31,16) both work
without editing this file.

Usage:
    python plot_tab_FV_analysis.py RESULTS_FOLDER
    python plot_tab_FV_analysis.py RESULTS_FOLDER -o result.xlsx --pct
    python plot_tab_FV_analysis.py RESULTS_FOLDER --limit 5000   # quick test
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from collections import defaultdict

import pandas as pd
import numpy as np

from csv_tools import (
    check_header,
    read_code_parameters,
    order_decoders,
    resolve_verified,
)

# ===========================================================================
# CONFIG
# ===========================================================================
SER_COLUMN = "prop_error"
# Fallback when the column is absent. Matches both "p0.03_..." (what the
# simulator writes) and the older "p0_03_..." naming.
SER_FROM_FILENAME = r"p(\d)[._](\d+)"

csv.field_size_limit(10 ** 9)
_DICT_KEY_RE = re.compile(r"(\d+)\s*:")

# Parsed values repeat heavily across millions of rows (an all-ones verified
# list, an empty error dict), so both parsers memoize.
_bits_cache: dict = {}
_keys_cache: dict = {}
_CACHE_MAX = 300_000


def parse_bits(s, n_symbols):
    """'[1 0 1 ...]' -> uint8 ndarray of length n_symbols."""
    key = (s, n_symbols)
    cached = _bits_cache.get(key)
    if cached is not None:
        return cached
    body = s.strip().strip("[]").replace(" ", "").replace(",", "").replace("\n", "")
    arr = np.frombuffer(body.encode(), dtype=np.uint8) - 48
    if arr.size != n_symbols:
        out = np.zeros(n_symbols, dtype=np.uint8)
        out[: min(n_symbols, arr.size)] = arr[:n_symbols]
        arr = out
    if len(_bits_cache) < _CACHE_MAX:
        _bits_cache[key] = arr
    return arr


def parse_keys(s):
    """'{5: 3294145190, 12: 2339669200}' -> (5, 12); values are ignored."""
    if not s or s == "{}":
        return ()
    cached = _keys_cache.get(s)
    if cached is not None:
        return cached
    keys = tuple(int(x) for x in _DICT_KEY_RE.findall(s))
    if len(_keys_cache) < _CACHE_MAX:
        _keys_cache[s] = keys
    return keys


def ser_from_filename(path):
    match = re.search(SER_FROM_FILENAME, os.path.basename(path))
    return float(f"{match.group(1)}.{match.group(2)}") if match else None


def discover_decoders(header):
    """Map decoder name -> its unsolved / initial / verified columns.

    This walks unsolved_errors_* rather than verified_list_*, because the
    VSD-guided decoders' verified columns do not follow the pattern and share
    a prefix with VSD_AFV's.
    """
    hset = set(header)
    found = {}
    for column in header:
        if not column.startswith("unsolved_errors_"):
            continue
        name = column[len("unsolved_errors_"):]
        found[name] = {
            "unsolved": column,
            "initial": (f"initial_errors_{name}"
                        if f"initial_errors_{name}" in hset else None),
            "verified": resolve_verified(header, name),
        }
    return found


# ===========================================================================
class Acc:
    """Running counts for one decoder at one SER."""
    __slots__ = ("ver_d", "ver_p", "fv_d", "fv_p", "fu_d", "fu_p")

    def __init__(self):
        self.ver_d = self.ver_p = 0
        self.fv_d = self.fv_p = 0
        self.fu_d = self.fu_p = 0


def process_folder(folder, limit=None, verbose=True, strict=True):
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(".csv"))
    if not files:
        raise SystemExit(f"No .csv files in {folder}")

    # Read n and k from the first file that actually passes the header check,
    # so summary tables sitting in the same folder cannot be sampled.
    n_symbols = data_end = n_parity = None

    n_cw = defaultdict(int)                       # ser -> codeword count
    acc = defaultdict(lambda: defaultdict(Acc))   # ser -> decoder -> Acc
    all_decoders = []
    used, skipped = [], []

    for path in files:
        t0 = time.time()
        fallback_ser = ser_from_filename(path)
        fname = os.path.basename(path)

        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            try:
                header = [h.strip().lstrip("\ufeff") for h in next(reader)]
            except StopIteration:
                skipped.append((fname, "empty file"))
                print(f"[skip] {fname}   empty file")
                continue

            reason = check_header(header, "verified_list_") if strict else None
            if reason:
                skipped.append((fname, reason))
                print(f"[skip] {fname}   {reason}")
                continue

            if n_symbols is None:
                n_symbols, data_end = read_code_parameters(path)
                n_parity = n_symbols - data_end
                if verbose:
                    print(f"Code parameters from data: n={n_symbols}, k={data_end} "
                          f"({data_end} data + {n_parity} parity symbols)\n")

            decoders = discover_decoders(header)
            no_verified = [n for n, m in decoders.items() if not m["verified"]]
            decoders = {n: m for n, m in decoders.items() if m["verified"]}
            for name in decoders:
                if name not in all_decoders:
                    all_decoders.append(name)

            pos = {c: i for i, c in enumerate(header)}
            ser_i = pos.get(SER_COLUMN)
            plan = [(name,
                     pos[m["verified"]],
                     pos.get(m["unsolved"]) if m["unsolved"] else None,
                     pos.get(m["initial"]) if m["initial"] else None)
                    for name, m in decoders.items()]

            if verbose:
                src = SER_COLUMN if ser_i is not None else f"filename -> {fallback_ser}"
                print(f"[read] {fname}   decoders={len(plan)}   SER from {src}")
                for name in order_decoders(list(decoders)):
                    meta = decoders[name]
                    note = ("" if meta["verified"] == f"verified_list_{name}"
                            else "   <- non-standard column name")
                    print(f"        {name:<18} verified={meta['verified']}{note}")
                    if not meta["initial"]:
                        print(f"        {'':<18} [warn] no initial_errors_{name}")
                if no_verified:
                    print(f"        [warn] skipped decoders with no verified "
                          f"column: {', '.join(no_verified)}")

            nrow = 0
            for row in reader:
                if not row:
                    continue
                if ser_i is not None and row[ser_i]:
                    try:
                        ser = float(row[ser_i])
                    except ValueError:
                        ser = fallback_ser
                else:
                    ser = fallback_ser
                if ser is None:
                    raise SystemExit("Could not determine the input SER from the "
                                     "prop_error column or the file name")

                n_cw[ser] += 1
                bucket = acc[ser]

                for name, vi, ui, ii in plan:
                    verified = parse_bits(row[vi], n_symbols)
                    a = bucket[name]

                    vd = int(verified[:data_end].sum())
                    vp = int(verified[data_end:].sum())
                    a.ver_d += vd
                    a.ver_p += vp

                    # False verified: verified == 1 but the index is unsolved.
                    if ui is not None:
                        for i in parse_keys(row[ui]):
                            if i < n_symbols and verified[i]:
                                if i < data_end:
                                    a.fv_d += 1
                                else:
                                    a.fv_p += 1

                    # False unverified: verified == 0 and not an initial error.
                    zd = data_end - vd
                    zp = n_parity - vp
                    if ii is not None:
                        for i in parse_keys(row[ii]):
                            if i < n_symbols and not verified[i]:
                                if i < data_end:
                                    zd -= 1
                                else:
                                    zp -= 1
                    a.fu_d += zd
                    a.fu_p += zp

                nrow += 1
                if limit and nrow >= limit:
                    break

        used.append(fname)
        if verbose:
            print(f"        {nrow:,} codewords in {time.time() - t0:.1f}s")

    print(f"\nUsing {len(used)} file(s), skipped {len(skipped)}")
    if not used:
        raise SystemExit("No file matched the expected format. Check "
                         "REQUIRED_HEADER_PREFIX in csv_tools.py, or pass "
                         "--loose to disable the header check.")
    return acc, n_cw, order_decoders(all_decoders), n_symbols, data_end


# ===========================================================================
def build_table(acc, n_cw, decoders, n_symbols, data_end, pct=False):
    n_parity = n_symbols - data_end
    metrics = [("Verified symbol", "ver_d", "ver_p"),
               ("False unverified symbol", "fu_d", "fu_p"),
               ("False verified symbol", "fv_d", "fv_p")]

    cols = [("Number of decoded symbol", "", "Data"),
            ("Number of decoded symbol", "", "Parity")]
    for decoder in decoders:
        for label, _, _ in metrics:
            cols.append((decoder, label, "Data"))
            cols.append((decoder, label, "Parity"))

    rows, index = [], []
    for ser in sorted(acc):
        cw = n_cw[ser]
        tot_d, tot_p = cw * data_end, cw * n_parity
        row = [tot_d, tot_p]
        for decoder in decoders:
            a = acc[ser].get(decoder) or Acc()
            for _, field_d, field_p in metrics:
                vd, vp = getattr(a, field_d), getattr(a, field_p)
                if pct:
                    row += [100.0 * vd / tot_d if tot_d else 0.0,
                            100.0 * vp / tot_p if tot_p else 0.0]
                else:
                    row += [vd, vp]
        rows.append(row)
        index.append(ser)

    df = pd.DataFrame(
        rows,
        index=pd.Index(index, name="Input SER"),
        columns=pd.MultiIndex.from_tuples(cols, names=["Decoder", "Metric", "Type"]),
    )
    return df.round(4) if pct else df


# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="folder holding the simulator's raw CSV files")
    parser.add_argument("-o", "--output", default="decoder_stats.csv",
                        help="output file, .csv or .xlsx (default: decoder_stats.csv)")
    parser.add_argument("--pct", action="store_true",
                        help="also build a percentage table against the total symbol count")
    parser.add_argument("--limit", type=int, default=None,
                        help="read only the first N rows of each file (for testing)")
    parser.add_argument("--loose", action="store_true",
                        help="disable the header check that skips foreign-format files")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    acc, n_cw, decoders, n_symbols, data_end = process_folder(
        args.folder, args.limit, not args.quiet, strict=not args.loose)
    table = build_table(acc, n_cw, decoders, n_symbols, data_end)

    pd.set_option("display.width", 100000)
    pd.set_option("display.max_columns", 500)
    print("\n" + table.to_string())

    if args.output.lower().endswith(".xlsx"):
        with pd.ExcelWriter(args.output) as writer:
            table.to_excel(writer, sheet_name="counts")
            if args.pct:
                build_table(acc, n_cw, decoders, n_symbols, data_end,
                            pct=True).to_excel(writer, sheet_name="percent")
    else:
        table.to_csv(args.output)
        if args.pct:
            pct_path = args.output.rsplit(".", 1)[0] + "_percent.csv"
            build_table(acc, n_cw, decoders, n_symbols, data_end,
                        pct=True).to_csv(pct_path)
            print(f"\nSaved: {pct_path}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
