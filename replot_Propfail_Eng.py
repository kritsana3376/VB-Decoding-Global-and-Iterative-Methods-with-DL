import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =================================================================
# โซนตั้งค่า (CONFIGURATION BLOCK)
# =================================================================
TARGET_FONT = 'Tahoma'

FONT_LABEL = 24          # ขนาดฟอนต์ชื่อแกน X และ Y
FONT_TICKS = 20          # ขนาดฟอนต์ตัวเลขบนแกน X และ Y
FONT_LEGEND = 18         # ขนาดฟอนต์ของคำอธิบายเส้นกราฟ (Legend)

GRID_THICKNESS = 0.6
GRID_STYLE = '--'
GRID_COLOR = "#757575"

AXIS_THICKNESS = 1.2     # ความหนาของเส้นแกน X และ Y
AXIS_COLOR = '#333333'
TICK_LENGTH = 6
TICK_THICKNESS = 1.2

FIG_SIZE = (14, 10)
LINE_WIDTH_DATA = 2.5
MARKER_SIZE_DATA = 12

# แกน X: X_AUTO=True ใช้ช่วงของข้อมูลจริง, False บังคับตาม X_START/X_END/X_STEP
X_AUTO = False
X_START = 0.01
X_END = 0.16
X_STEP = 0.01

OUTPUT_FILENAME = 'decoding_failure_ENG_fixed_plot.png'

# key = ชื่อ decoder (จะใส่ 'success_' นำหน้าหรือไม่ก็ได้ รองรับทั้งคู่)
plot_styles = {
    # --- กลุ่ม EhMP (โทนสีแดง) ---
    'EhMP':           {'marker': 'o', 'linestyle': '-', 'color': 'red',            'label': 'E-hMP'},
    'E_hMP_guid_CNN': {'marker': 'X', 'linestyle': '-', 'color': 'darkred',        'label': 'E-hMP CNN'},
    'E_hMP_guid_GNN': {'marker': 'P', 'linestyle': '-', 'color': 'salmon',         'label': 'E-hMP GNN'},

    # --- กลุ่ม M (โทนสีน้ำเงิน) ---
    'MhMP':           {'marker': 'o', 'linestyle': '-', 'color': 'blue',           'label': 'M-hMP'},
    'M_hMP_guid_CNN': {'marker': 'X', 'linestyle': '-', 'color': 'darkblue',       'label': 'M-hMP CNN'},
    'M_hMP_guid_GNN': {'marker': 'p', 'linestyle': '-', 'color': 'dodgerblue',     'label': 'M-hMP GNN'},

    # --- กลุ่ม VSD (โทนสีเขียว) ---
    'VSD_AFV':        {'marker': 'o', 'linestyle': '-', 'color': 'limegreen',      'label': 'VSD-pv'},
    'VSD_guid_CNN':   {'marker': 'X', 'linestyle': '-', 'color': 'darkgreen',      'label': 'VSD CNN'},
    'VSD_guid_GNN':   {'marker': 'P', 'linestyle': '-', 'color': 'mediumseagreen', 'label': 'VSD GNN'},
}
# =================================================================


def get_fail_rate(df, key):
    """
    ดึงค่า Pfail ของ decoder ตัวหนึ่ง รองรับ analysis_summary.csv ทั้ง 2 รุ่น
      รุ่นใหม่ : pfail_<name>                      (คำนวณมาแล้ว)
      รุ่นเก่า : success_<name>_sum / _count       (ต้องคำนวณเอง)
    คืน (Series, ที่มา) หรือ (None, None) ถ้าไม่พบ
    """
    name = key[len('success_'):] if key.startswith('success_') else key

    col = f'pfail_{name}'
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce'), col

    for s_col, c_col in ((f'count_success_{name}', f'count_all_{name}'),
                         (f'success_{name}_sum', f'success_{name}_count')):
        if s_col in df.columns and c_col in df.columns:
            n_ok = pd.to_numeric(df[s_col], errors='coerce')
            n_all = pd.to_numeric(df[c_col], errors='coerce').replace(0, np.nan)
            return (n_all - n_ok) / n_all, f'{s_col} / {c_col}'

    return None, None


def plot_and_save_graph(csv_path):
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

    for key, style in plot_styles.items():
        fail_rate, src = get_fail_rate(df, key)
        if fail_rate is None:
            missing.append(key)
            continue

        y = fail_rate.to_numpy(dtype=float)
        # log scale วาดค่า 0 หรือค่าติดลบไม่ได้ -> ทำเป็น NaN ให้เส้นขาดแทนที่จะหายเงียบ
        n_zero += int(np.sum(~np.isnan(y) & (y <= 0)))
        y = np.where(y > 0, y, np.nan)

        ax.plot(x, y, linewidth=LINE_WIDTH_DATA, markersize=MARKER_SIZE_DATA, **style)
        drawn.append((key, src))

    # ---------- แจ้งเตือนแทนที่จะเซฟรูปเปล่า ----------
    if missing:
        print("ไม่พบคอลัมน์ของ decoder เหล่านี้ในไฟล์ CSV:")
        for k in missing:
            print(f"   - {k}")
    if not drawn:
        print("\nไม่มีเส้นกราฟให้วาดเลย จึงไม่เซฟรูป")
        print("คอลัมน์ที่มีอยู่จริงในไฟล์:")
        for c in df.columns:
            print(f"   {c}")
        plt.close(fig)
        return

    print(f"วาดกราฟ {len(drawn)} เส้น:")
    for k, src in drawn:
        print(f"   - {k}   (ใช้คอลัมน์ {src})")

    ax.set_yscale('log')

    ticks = np.unique(x[~np.isnan(x)]) if X_AUTO else np.arange(X_START, X_END + 1e-9, X_STEP)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:.2f}" for v in ticks], fontsize=FONT_TICKS)
    ax.tick_params(axis='y', labelsize=FONT_TICKS)

    ax.set_xlabel('Input Symbol Error Rate (Input SER)', fontsize=FONT_LABEL, labelpad=12)
    ax.set_ylabel('Probability of Decoding Failure', fontsize=FONT_LABEL, labelpad=12)

    ax.legend(fontsize=FONT_LEGEND, loc='best', frameon=True,
              facecolor='white', edgecolor='none')
    ax.grid(True, which='both', linestyle=GRID_STYLE,
            linewidth=GRID_THICKNESS, color=GRID_COLOR, alpha=0.7)

    # เส้นแกนทึบ + ขีดบอกระดับยื่นออกด้านนอก
    for side in ('left', 'bottom'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(AXIS_COLOR)
        ax.spines[side].set_linewidth(AXIS_THICKNESS)
    ax.tick_params(axis='both', which='major', direction='out',
                   length=TICK_LENGTH, width=TICK_THICKNESS, colors=AXIS_COLOR,
                   left=True, bottom=True)
    ax.tick_params(axis='y', which='minor', left=False)

    fig.tight_layout()

    output_plot_path = os.path.join(csv_folder, OUTPUT_FILENAME)
    # bbox_inches='tight' กันชื่อแกนฟอนต์ใหญ่โดนตัดขอบ
    fig.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nบันทึกรูปภาพกราฟเรียบร้อยแล้วที่:\n   {output_plot_path}")
    if n_zero:
        print(f"หมายเหตุ: มีค่า <= 0 จำนวน {n_zero} จุด ที่วาดบนสเกล log ไม่ได้ จึงเว้นช่วงไว้")

    plt.show()


# =================================================================
if __name__ == '__main__':
    csv_target = r"pre_master/new_journal_run/Add_3121/analysis_summary.csv"
    plot_and_save_graph(sys.argv[1] if len(sys.argv) > 1 else csv_target)