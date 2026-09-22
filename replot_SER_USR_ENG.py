import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =================================================================
# โซนตั้งค่ากราฟ (CONFIGURATION BLOCK)
# =================================================================
TARGET_FONT = 'Tahoma'

FONT_LABEL = 24          # ขนาดฟอนต์ชื่อแกน X และ Y
FONT_TICKS = 20          # ขนาดฟอนต์ตัวเลขบนแกน X และ Y
FONT_LEGEND = 18         # ขนาดฟอนต์ของคำอธิบายเส้นกราฟ (Legend)
FONT_TITLE = 14          # ขนาดฟอนต์ชื่อหัวกราฟ

GRID_THICKNESS = 0.6
GRID_STYLE = '--'
GRID_COLOR = '#757575'

AXIS_THICKNESS = 1.2     # ความหนาของเส้นแกน X และ Y
AXIS_COLOR = '#333333'
TICK_LENGTH = 6
TICK_THICKNESS = 1.2

FIG_SIZE = (14, 10)
LINE_WIDTH_DATA = 3.0
MARKER_SIZE_DATA = 12

# แกน X: X_AUTO=True ใช้ช่วงของข้อมูลจริง, False บังคับตาม X_START/X_END/X_STEP
X_AUTO = False
X_START = 0.01
X_END = 0.16
X_STEP = 0.01

OUTPUT_FILENAME = 'SER_USR_Replot_ENG.png'

# key = ชื่อ decoder ตามที่ปรากฏในคอลัมน์ ser_ratio_<name> / usr_ratio_<name>
plot_configs = [
    # --- กลุ่ม EhMP (โทนสีแดง) ---
    {'name': 'EhMP',           'label': 'E-hMP',     'color': 'red',            'marker': '^'},
    # {'name': 'E_hMP_guid_CNN', 'label': 'E-hMP CNN', 'color': 'darkred',        'marker': 'v'},
    # {'name': 'E_hMP_guid_GNN', 'label': 'E-hMP GNN', 'color': 'salmon',         'marker': '<'},

    # --- กลุ่ม M (โทนสีน้ำเงิน) ---
    {'name': 'MhMP',           'label': 'M-hMP',     'color': 'blue',           'marker': 'd'},
    # {'name': 'M_hMP_guid_CNN', 'label': 'M-hMP CNN', 'color': 'darkblue',       'marker': 's'},
    # {'name': 'M_hMP_guid_GNN', 'label': 'M-hMP GNN', 'color': 'dodgerblue',     'marker': 'p'},

    # --- กลุ่ม VSD (โทนสีเขียว) ---
    {'name': 'VSD_AFV',        'label': 'VSD-pv',    'color': 'limegreen',      'marker': '*'},
    # {'name': 'VSD_guid_CNN',   'label': 'VSD CNN',   'color': 'darkgreen',      'marker': 'X'},
    # {'name': 'VSD_guid_GNN',   'label': 'VSD GNN',   'color': 'mediumseagreen', 'marker': 'P'},
]
# =================================================================


def replot_summary_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: ไม่พบไฟล์ CSV ที่ตำแหน่ง: {csv_path}")
        return

    absolute_csv_path = os.path.abspath(csv_path)
    csv_folder = os.path.dirname(absolute_csv_path)

    df = pd.read_csv(absolute_csv_path)
    if 'prop_error' not in df.columns:
        print(f"Error: ไฟล์นี้ไม่มีคอลัมน์ 'prop_error' -> {absolute_csv_path}")
        return

    df = df.sort_values('prop_error').reset_index(drop=True)
    x = pd.to_numeric(df['prop_error'], errors='coerce').to_numpy(dtype=float)

    sns.set_theme(style="whitegrid", rc={"font.family": TARGET_FONT})
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    drawn, missing, n_zero = [], [], 0

    for cfg in plot_configs:
        name = cfg['name']
        for kind, style, alpha, tag in (('ser', '-', 1.0, 'D-SER'),
                                        ('usr', '--', 0.7, 'D-USR')):
            col = f'{kind}_ratio_{name}'
            if col not in df.columns:
                missing.append(col)
                continue

            y = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
            # log scale วาดค่า <= 0 ไม่ได้ -> ทำเป็น NaN ให้เส้นขาดแทนที่จะหายเงียบ
            n_zero += int(np.sum(~np.isnan(y) & (y <= 0)))
            y = np.where(y > 0, y, np.nan)

            ax.plot(x, y, marker=cfg['marker'], linestyle=style, color=cfg['color'],
                    alpha=alpha, linewidth=LINE_WIDTH_DATA, markersize=MARKER_SIZE_DATA,
                    label=f"{cfg['label']} ({tag})")
            drawn.append(col)

    # ---------- แจ้งเตือนแทนที่จะเซฟรูปเปล่า ----------
    if missing:
        print("ไม่พบคอลัมน์เหล่านี้ในไฟล์ CSV:")
        for c in missing:
            print(f"   - {c}")
    if not drawn:
        print("\nไม่มีเส้นกราฟให้วาดเลย จึงไม่เซฟรูป")
        print("คอลัมน์ที่มีอยู่จริงในไฟล์:")
        for c in df.columns:
            print(f"   {c}")
        plt.close(fig)
        return

    print(f"วาดกราฟ {len(drawn)} เส้น: {', '.join(drawn)}")

    ax.set_yscale('log')

    ticks = np.unique(x[~np.isnan(x)]) if X_AUTO else np.arange(X_START, X_END + 1e-9, X_STEP)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:.2f}" for v in ticks], fontsize=FONT_TICKS)
    ax.tick_params(axis='y', labelsize=FONT_TICKS)

    ax.set_xlabel('Input Symbol Error Rate (Input SER)', fontsize=FONT_LABEL, labelpad=12)
    ax.set_ylabel('Data Symbol Error Rate & Data Unverified Symbol Rate',
                  fontsize=FONT_LABEL, labelpad=12)
    ax.set_title('', fontsize=FONT_TITLE, pad=15)

    ax.legend(fontsize=FONT_LEGEND, loc='best', frameon=True,
              facecolor='white', edgecolor='none')
    ax.grid(True, which='both', linestyle=GRID_STYLE,
            linewidth=GRID_THICKNESS, color=GRID_COLOR, alpha=0.7)

    for side in ('left', 'bottom'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(AXIS_COLOR)
        ax.spines[side].set_linewidth(AXIS_THICKNESS)
    ax.tick_params(axis='both', which='major', direction='out',
                   length=TICK_LENGTH, width=TICK_THICKNESS, colors=AXIS_COLOR,
                   left=True, bottom=True)
    ax.tick_params(axis='y', which='minor', left=False)

    fig.tight_layout()

    save_path = os.path.join(csv_folder, OUTPUT_FILENAME)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nบันทึกรูปภาพกราฟเรียบร้อยแล้วที่:\n   {save_path}")
    if n_zero:
        print(f"หมายเหตุ: มีค่า <= 0 จำนวน {n_zero} จุด ที่วาดบนสเกล log ไม่ได้ จึงเว้นช่วงไว้")

    plt.show()


if __name__ == '__main__':
    csv_target = r"C:\Users\user\Desktop\ProjectY4\pre_master\new_journal_run\Add_3121\SER_plot_summary.csv"
    replot_summary_data(sys.argv[1] if len(sys.argv) > 1 else csv_target)