"""Shared helpers for reading the simulator's raw per-trial CSV files.

`run_all_for_Simulation.py` writes one CSV per symbol error rate. The three
analysis scripts all need the same things: list the CSVs in a folder, reject
files written in a different format, pull the code parameters out, and write a
summary table with a fixed float format. Those used to be copy-pasted into
each script; they live here now.
"""

import csv
from pathlib import Path

import pandas as pd

# The first six columns every raw file must have, in this order. A file whose
# header does not start with these is from a different format and is skipped.
REQUIRED_HEADER_PREFIX = [
    "n", "k", "symbol_size", "prop_error",
    "Number of Error Symbol", "Number of Error Bits",
]

# Display order: the algebraic baseline first, then the CNN and GNN of the
# same family. Names not listed here are appended alphabetically.
DECODER_ORDER = [
    "EhMP", "E_hMP_guid_CNN", "E_hMP_guid_GNN",
    "MhMP", "M_hMP_guid_CNN", "M_hMP_guid_GNN",
    "VSD_AFV", "VSD_guid_CNN", "VSD_guid_GNN",
]

# Raw rows hold long dict/array reprs, well past the default field limit.
csv.field_size_limit(10 ** 9)


def read_header(path):
    """Read just the header row. Returns None for an empty file."""
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        try:
            return [h.strip().lstrip("\ufeff") for h in next(csv.reader(fh))]
        except StopIteration:
            return None


def check_header(header, required_prefix_of=None):
    """Validate a header. Returns None if usable, else why the file is skipped.

    `required_prefix_of` names a column prefix the file must also carry, e.g.
    "success_" for the Pfail table or "unsolved_errors_" for the SER table.
    """
    if header is None:
        return "empty file"

    need = REQUIRED_HEADER_PREFIX
    if len(header) < len(need):
        return f"only {len(header)} columns (need at least {len(need)})"

    mismatches = [
        f"col{i + 1}: {got!r} != {want!r}"
        for i, (got, want) in enumerate(zip(header[:len(need)], need))
        if got != want
    ]
    if mismatches:
        return "header mismatch -> " + "; ".join(mismatches[:3])

    if required_prefix_of and not any(c.startswith(required_prefix_of) for c in header):
        return f"no {required_prefix_of}* column found"
    return None


def collect_valid_files(folder, required_prefix_of=None):
    """List the CSVs in `folder` whose header matches the expected format.

    Returns (valid_paths, skipped) and prints a line per skipped file.
    Raises SystemExit if the folder is missing, empty, or has no usable file.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files in {folder}")

    valid, skipped = [], []
    for path in csv_files:
        reason = check_header(read_header(path), required_prefix_of)
        if reason:
            skipped.append((path.name, reason))
            print(f"[skip] {path.name}   {reason}")
        else:
            valid.append(path)

    if not valid:
        raise SystemExit(
            "No file matched the expected format. Check REQUIRED_HEADER_PREFIX "
            "in csv_tools.py against the header your simulator actually writes."
        )
    print(f"\nUsing {len(valid)} file(s), skipped {len(skipped)}\n")
    return valid, skipped


def read_code_parameters(path, fallback_n=31, fallback_k=21):
    """Read n and k from the first data row rather than hardcoding them.

    The raw files carry the code parameters per row, so the analysis works for
    BCH(31,21), BCH(31,16) and anything else without editing a constant.
    """
    try:
        head = pd.read_csv(path, usecols=["n", "k"], nrows=1)
        return int(head["n"].iloc[0]), int(head["k"].iloc[0])
    except (ValueError, KeyError, IndexError, pd.errors.EmptyDataError):
        print(f"[warn] {Path(path).name}: no usable n/k columns, "
              f"falling back to n={fallback_n}, k={fallback_k}")
        return fallback_n, fallback_k


def resolve_verified(columns, name):
    """Find the verified_list column belonging to decoder `name`.

    1. `verified_list_<name>`, the normal case.
    2. Otherwise any `verified_list_*` column ending in `<name>`, which covers
       the VSD-guided decoders: the simulator writes their masks as
       `verified_list_VSD_AFV_make_success_VSD_guid_CNN`.

    Note that the fallback column is the locating vector that made VSD
    succeed, not the decoder's final verified mask. Treat those two D-USR
    figures accordingly.
    """
    exact = f"verified_list_{name}"
    if exact in columns:
        return exact
    candidates = [c for c in columns
                  if c.startswith("verified_list_") and c.endswith(name)]
    return candidates[0] if candidates else None


def order_decoders(names):
    """Sort decoder names into DECODER_ORDER, unknown ones last."""
    rank = {n: i for i, n in enumerate(DECODER_ORDER)}
    return sorted(names, key=lambda d: (rank.get(d, len(rank)), d))


def write_summary_csv(df, path, formatted_columns, float_format):
    """Write a summary table, formatting only the result columns.

    prop_error is left alone so it stays readable as 0.03 rather than
    3.00000e-02.
    """
    out = df.copy()
    for column in formatted_columns:
        out[column] = out[column].map(
            lambda x: "" if pd.isna(x) else float_format % x
        )
    out.to_csv(path, index=False)
