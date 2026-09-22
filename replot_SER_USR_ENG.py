"""Plot D-SER and D-USR against input SER, on a log y axis.

Reads the summary written by plot_tab_output_SER_USR.py. Each decoder gets
two lines in the same colour: D-SER solid, D-USR dashed.

Usage:
    python replot_SER_USR_ENG.py path/to/SER_plot_summary.csv
    python replot_SER_USR_ENG.py SER_plot_summary.csv -d EhMP MhMP VSD_AFV
"""

import argparse
import os

import pandas as pd

import plot_style as ps

OUTPUT_FILENAME = "SER_USR_Replot_ENG.png"

LINE_WIDTH_DATA = 3.0

# The paper's figure shows only the three algebraic baselines. Pass -d to
# plot a different set; any name in plot_style.DECODER_STYLES works.
DEFAULT_DECODERS = ["EhMP", "MhMP", "VSD_AFV"]

MARKERS = {
    "EhMP": "^", "E_hMP_guid_CNN": "v", "E_hMP_guid_GNN": "<",
    "MhMP": "d", "M_hMP_guid_CNN": "s", "M_hMP_guid_GNN": "p",
    "VSD_AFV": "*", "VSD_guid_CNN": "X", "VSD_guid_GNN": "P",
}


def replot_summary_data(csv_path, decoders=None, output_name=OUTPUT_FILENAME):
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return

    csv_path = os.path.abspath(csv_path)
    csv_folder = os.path.dirname(csv_path)
    decoders = decoders or DEFAULT_DECODERS

    df = pd.read_csv(csv_path)
    if "prop_error" not in df.columns:
        print(f"Error: no 'prop_error' column in {csv_path}")
        return

    df = df.sort_values("prop_error").reset_index(drop=True)
    x = pd.to_numeric(df["prop_error"], errors="coerce").to_numpy(dtype=float)

    fig, ax = ps.new_figure()
    drawn, missing, n_dropped = [], [], 0

    for name in decoders:
        style = ps.DECODER_STYLES.get(name)
        if style is None:
            print(f"[warn] unknown decoder {name!r}, skipping")
            continue

        for kind, linestyle, alpha, tag in (("ser", "-", 1.0, "D-SER"),
                                            ("usr", "--", 0.7, "D-USR")):
            column = f"{kind}_ratio_{name}"
            if column not in df.columns:
                missing.append(column)
                continue

            y, dropped = ps.positive_only(
                pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float))
            n_dropped += dropped

            ax.plot(x, y, marker=MARKERS.get(name, "o"), linestyle=linestyle,
                    color=style["color"], alpha=alpha,
                    linewidth=LINE_WIDTH_DATA, markersize=ps.MARKER_SIZE_DATA,
                    label=f"{style['label']} ({tag})")
            drawn.append(column)

    # Report rather than silently saving an empty figure.
    if missing:
        print("These columns were not found in the CSV:")
        for column in missing:
            print(f"   - {column}")
    if not drawn:
        print("\nNothing to plot, so no figure was saved.")
        print("Columns actually present:")
        for column in df.columns:
            print(f"   {column}")
        return

    print(f"Plotted {len(drawn)} line(s): {', '.join(drawn)}")

    ps.finish_axes(ax, x,
                   "Input Symbol Error Rate (Input SER)",
                   "Data Symbol Error Rate & Data Unverified Symbol Rate")
    fig.tight_layout()
    ps.save_figure(fig, os.path.join(csv_folder, output_name), n_dropped)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_path",
                        help="SER_plot_summary.csv from plot_tab_output_SER_USR.py")
    parser.add_argument("-d", "--decoders", nargs="+", default=None,
                        help=f"decoders to plot (default: {' '.join(DEFAULT_DECODERS)})")
    parser.add_argument("-o", "--output", default=OUTPUT_FILENAME,
                        help=f"figure file name (default: {OUTPUT_FILENAME})")
    args = parser.parse_args()
    replot_summary_data(args.csv_path, args.decoders, args.output)


if __name__ == "__main__":
    main()
