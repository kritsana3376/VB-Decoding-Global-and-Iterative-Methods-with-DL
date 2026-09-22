"""Plot probability of decoding failure against input SER, on a log y axis.

Reads the summary written by plot_tab_Pfail.py.

Usage:
    python replot_Propfail_Eng.py path/to/analysis_summary.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

import plot_style as ps

OUTPUT_FILENAME = "decoding_failure_ENG_fixed_plot.png"

LINE_WIDTH_DATA = 2.5

# Marker per decoder; colours and labels come from plot_style.DECODER_STYLES.
MARKERS = {
    "EhMP": "o", "E_hMP_guid_CNN": "X", "E_hMP_guid_GNN": "P",
    "MhMP": "o", "M_hMP_guid_CNN": "X", "M_hMP_guid_GNN": "p",
    "VSD_AFV": "o", "VSD_guid_CNN": "X", "VSD_guid_GNN": "P",
}


def get_fail_rate(df, key):
    """Fetch one decoder's Pfail, supporting both summary formats.

    Current: a precomputed `pfail_<name>` column.
    Older:   `count_success_<name>` / `count_all_<name>`, or
             `success_<name>_sum` / `success_<name>_count`, computed here.

    Returns (Series, source description), or (None, None) if not present.
    """
    name = key[len("success_"):] if key.startswith("success_") else key

    column = f"pfail_{name}"
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce"), column

    for sum_col, count_col in ((f"count_success_{name}", f"count_all_{name}"),
                               (f"success_{name}_sum", f"success_{name}_count")):
        if sum_col in df.columns and count_col in df.columns:
            n_ok = pd.to_numeric(df[sum_col], errors="coerce")
            n_all = pd.to_numeric(df[count_col], errors="coerce").replace(0, np.nan)
            return (n_all - n_ok) / n_all, f"{sum_col} / {count_col}"

    return None, None


def plot_and_save_graph(csv_path, output_name=OUTPUT_FILENAME):
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return

    csv_path = os.path.abspath(csv_path)
    csv_folder = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)
    if "prop_error" not in df.columns:
        print(f"Error: no 'prop_error' column in {csv_path}")
        return

    df = df.sort_values("prop_error").reset_index(drop=True)
    x = pd.to_numeric(df["prop_error"], errors="coerce").to_numpy(dtype=float)

    fig, ax = ps.new_figure()
    drawn, missing, n_dropped = [], [], 0

    for key, style in ps.DECODER_STYLES.items():
        fail_rate, source = get_fail_rate(df, key)
        if fail_rate is None:
            missing.append(key)
            continue

        y, dropped = ps.positive_only(fail_rate.to_numpy(dtype=float))
        n_dropped += dropped

        ax.plot(x, y, marker=MARKERS.get(key, "o"), linestyle="-",
                color=style["color"], label=style["label"],
                linewidth=LINE_WIDTH_DATA, markersize=ps.MARKER_SIZE_DATA)
        drawn.append((key, source))

    # Report rather than silently saving an empty figure.
    if missing:
        print("These decoders were not found in the CSV:")
        for key in missing:
            print(f"   - {key}")
    if not drawn:
        print("\nNothing to plot, so no figure was saved.")
        print("Columns actually present:")
        for column in df.columns:
            print(f"   {column}")
        return

    print(f"Plotted {len(drawn)} line(s):")
    for key, source in drawn:
        print(f"   - {key}   (using {source})")

    ps.finish_axes(ax, x,
                   "Input Symbol Error Rate (Input SER)",
                   "Probability of Decoding Failure")
    fig.tight_layout()
    ps.save_figure(fig, os.path.join(csv_folder, output_name), n_dropped)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_path", help="analysis_summary.csv from plot_tab_Pfail.py")
    parser.add_argument("-o", "--output", default=OUTPUT_FILENAME,
                        help=f"figure file name (default: {OUTPUT_FILENAME})")
    args = parser.parse_args()
    plot_and_save_graph(args.csv_path, args.output)


if __name__ == "__main__":
    main()
